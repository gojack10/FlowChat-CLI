import os
import sys
import json
import subprocess
import traceback
import pty
import select
import re
import shutil

from openai import OpenAI, OpenAIError
from dotenv import load_dotenv
import tiktoken
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import PathCompleter, Completer, Completion
from prompt_toolkit.key_binding import KeyBindings

# --- fzf check ---
def check_fzf():
    """checks if fzf is installed and executable."""
    return shutil.which("fzf")

# --- end fzf check ---

# --- FZF Path Completer ---
class FzfPathCompleter:
    """Handles path completion by running fzf externally and updating the buffer."""

    def run_fzf_and_update_buffer(self, event):
        """Runs fzf and directly updates the prompt_toolkit buffer."""
        global fzf_path
        buffer = event.cli.current_buffer
        if not fzf_path:
            # maybe add a visual bell or temp message?
            print("\n[fzf completer] fzf not found.", file=sys.stderr)
            return

        document = buffer.document
        text_before_cursor = document.text_before_cursor
        original_cursor_position = document.cursor_position

        # determine search directory and initial query for fzf
        search_dir = "."
        fzf_query = ""
        path_fragment_len = 0 # how much of the original text represents the path fragment

        if text_before_cursor:
            # expand ~ manually first
            if text_before_cursor.startswith("~/"):
                 base_path = os.path.expanduser(text_before_cursor)
                 is_tilde_path = True
            else:
                 base_path = text_before_cursor
                 is_tilde_path = False

            # check if it looks like a path
            if '/' in base_path:
                 potential_dir = os.path.dirname(base_path)
                 if os.path.isdir(potential_dir):
                      search_dir = potential_dir
                      fzf_query = os.path.basename(base_path)
                      path_fragment_len = len(text_before_cursor) # whole thing is path
                 elif not potential_dir and is_tilde_path: # edge case: just "~/"
                      search_dir = os.path.expanduser("~")
                      fzf_query = ""
                      path_fragment_len = len(text_before_cursor)
                 elif not potential_dir: # input is likely just a filename part relative to cwd
                      search_dir = "."
                      fzf_query = base_path
                      path_fragment_len = len(text_before_cursor)
                 else: # dirname doesn't exist, treat as query in cwd
                    search_dir = "."
                    fzf_query = base_path
                    path_fragment_len = len(text_before_cursor)
            else:
                 # assume it's just a filename part in the current dir
                 search_dir = "."
                 fzf_query = text_before_cursor
                 path_fragment_len = len(text_before_cursor)
        else:
             # no text, search current dir
             search_dir = "."
             fzf_query = ""
             path_fragment_len = 0

        # ensure search_dir is valid before proceeding
        if not os.path.isdir(search_dir):
             print(f"\n[fzf completer] search dir '{search_dir}' not found.", file=sys.stderr)
             return

        # construct the command pipeline: find | fzf
        # use find to list files/dirs, pipe to fzf
        # -maxdepth 1 to avoid excessive recursion initially?
        # use portable -print
        # escaping search_dir just in case it has spaces/special chars
        find_cmd = ["find", search_dir, "-maxdepth", "1", "-mindepth", "1", "-print"]
        # Add hidden file filtering? Maybe later.
        fzf_cmd = [fzf_path, "--height=40%", "--layout=reverse", "--border", "--query", fzf_query]

        selected_path_full = None
        try:
            # run find, capture its output
            find_process = subprocess.run(find_cmd, capture_output=True, text=True, check=False)
            if find_process.returncode != 0 and find_process.stderr:
                if "permission denied" not in find_process.stderr.lower():
                    print(f"\n[fzf completer] find error: {find_process.stderr.strip()}", file=sys.stderr)
                find_output = ""
            else:
                find_output = find_process.stdout

            # run fzf, passing find's output as stdin
            # crucial: run fzf without capturing output directly, let it interact with terminal
            # we need to temporarily give control to fzf
            # THIS IS COMPLEX - using subprocess.run directly might not work well for interactive fzf
            # A better approach might involve pty or direct terminal manipulation

            # --- Simplified approach for now: capture output --- >>>> TODO: REVISIT THIS
            fzf_process = subprocess.run(fzf_cmd, input=find_output, capture_output=True, text=True, check=False) # allow non-zero exit for cancellation

            if fzf_process.returncode == 0: # successful selection
                 selected_path_raw = fzf_process.stdout.strip()
                 if selected_path_raw: # ensure selection is not empty
                      # fzf output using find -print gives full path from find's start dir
                      selected_path_full = selected_path_raw
            elif fzf_process.returncode == 130: # cancelled
                 # print(f"\n[fzf completer] fzf cancelled by user.") # debug - removed
                 selected_path_full = None # ensure no path is set
                 # reset renderer even on cancellation to fix terminal state
                 event.cli.renderer.reset()
            else: # other error
                 print(f"\n[fzf completer] fzf error (code {fzf_process.returncode}): {fzf_process.stderr}", file=sys.stderr)
                 selected_path_full = None

            # --- End Simplified approach --- #

        except FileNotFoundError:
            print(f"\n[fzf completer] Error: {fzf_path} or find command not found.", file=sys.stderr)
        except Exception as e:
             print(f"\n[fzf completer] Unexpected error during fzf execution: {e}", file=sys.stderr)
             traceback.print_exc()

        # update buffer only if fzf returned a valid path
        if selected_path_full is not None:
            # determine the text to insert
            completion_text = selected_path_full
            # append a space if it's a directory for convenience
            if os.path.isdir(selected_path_full):
                 completion_text += "/"

            # calculate start position relative to current cursor
            # we need to delete the part of the text that was used as the query
            buffer.delete_before_cursor(path_fragment_len)
            buffer.insert_text(completion_text)
            # explicitly invalidate/request redraw after update
            # event.app.invalidate()
            # try a more forceful redraw
            event.cli.renderer.reset()

# --- End FZF Path Completer ---

# --- Tilde Workaround Completer ---
class TildeWorkaroundCompleter(Completer):
    """wraps pathcompleter to manually handle '~/...' paths because
       the default expanduser=true isn't working reliably here.
    """
    def __init__(self):
        # underlying completer for non-'~/...' cases
        self.path_completer = PathCompleter()

    def get_completions(self, document, complete_event):
        text_before_cursor = document.text_before_cursor

        if text_before_cursor.startswith('~/'):
            try:
                home_dir = os.path.expanduser('~')
                path_fragment = text_before_cursor[2:] # part after '~/'
                full_path_prefix = os.path.join(home_dir, path_fragment)
                dir_to_list = home_dir
                partial_item = path_fragment

                if os.path.sep in path_fragment:
                    dir_to_list = os.path.dirname(full_path_prefix)
                    partial_item = os.path.basename(full_path_prefix)

                if os.path.isdir(dir_to_list):
                    for item in os.listdir(dir_to_list):
                        if item.lower().startswith(partial_item.lower()):
                            full_item_path = os.path.join(dir_to_list, item)
                            completion_suffix = item
                            display_suffix = item
                            if os.path.isdir(full_item_path):
                                display_suffix += os.path.sep
                                completion_suffix += os.path.sep

                            start_pos = -len(partial_item)

                            yield Completion(
                                text=completion_suffix,
                                start_position=start_pos,
                                display='~/ ' + path_fragment[:-len(partial_item)] + display_suffix,
                                display_meta='manual'
                            )
            except Exception:
                pass # ignore errors during manual completion
        else:
            # delegate to the standard pathcompleter
            try:
                yield from self.path_completer.get_completions(document, complete_event)
            except Exception:
                pass # ignore errors during standard completion
# --- End Tilde Workaround Completer ---

console = Console()
fzf_completer = FzfPathCompleter()

# set up explicit key bindings
kb = KeyBindings()

# @kb.add('c-i') # Tab binding - keep commented out unless needed again
@kb.add('c-i') # tab binding
def _(event):
    """force completion on tab."""
    # this should now trigger our FzfPathCompleter directly
    fzf_completer.run_fzf_and_update_buffer(event)

@kb.add('escape', eager=True)
def _(event):
    """close completion menu on escape."""
    if event.cli.current_buffer.complete_state:
        event.cli.current_buffer.cancel_completion()
        # --- debug ---
        # print the type and available attributes of the renderer object
        # print(f"\\n--- DEBUG ---")
        # try:
        #     print(f"type(event.cli.renderer) = {type(event.cli.renderer)}")
        #     print(f"dir(event.cli.renderer) = {dir(event.cli.renderer)}")
        # except Exception as e:
        #     print(f"error inspecting renderer: {e}")
        # print(f"--- END DEBUG ---")
        # --- end debug ---
        # request redraw after cancelling to potentially speed up ui update
        # event.cli.renderer.request_redraw() # original error line
        # print("\\n redraw requested (commented out)") # placeholder
        # attempt a different way to redraw if needed after debugging
        event.app.invalidate() # use invalidate() on the application object
    else:
        # default escape action if not completing (e.g., clear input?)
        # or just pass to potentially let other bindings handle it.
        pass # or implement other escape behavior if desired

# Pass the TildeWorkaroundCompleter and other settings
session = PromptSession(complete_while_typing=False, key_bindings=kb)

# --- check fzf installation ---
fzf_path = check_fzf()
if not fzf_path:
    console.print("[bold red]Error:[/bold red] `fzf` command not found in PATH. FZF-based autocompletion will be disabled.")
    # we might set a flag here later to disable the fzf completer
    # for now, just printing the error is sufficient for task 1
# --- end check ---

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    console.print("[bold red]Error:[/bold red] OPENROUTER_API_KEY environment variable not set.")
    sys.exit(1)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

MODEL = "google/gemini-2.5-flash-preview:thinking"

try:
    encoding = tiktoken.encoding_for_model(MODEL.split(':')[0])
except KeyError:
    encoding = tiktoken.get_encoding("cl100k_base")

write_file_tool = {
    "type": "function",
    "function": {
        "name": "write_file",
        "description": "Write content to a specified file path. Use this when the user explicitly asks to save or write something to a file. The user will confirm the final path.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The relative or absolute path for the file to be written, as suggested by the user or inferred."
                },
                "content": {
                    "type": "string",
                    "description": "The full content to write into the file."
                }
            },
            "required": ["path", "content"]
        }
    }
}

tools = [write_file_tool]
tool_map = {"write_file": write_file_tool}

history = [
    {
        "role": "system",
        "content": (
            "You are a helpful assistant named TARS. You can propose writing files using the `write_file` tool. "
            "When asked to write a file, suggest a path and provide the content using the tool. The user will confirm/correct the final path before writing. "
            "Be concise in your regular responses while retaining as much information as possible."
        )
    }
]

def estimate_tokens(text):
    """estimates the number of tokens in a string."""
    return len(encoding.encode(text))

def estimate_prompt_tokens(messages):
    """estimates the total tokens in the message history (prompt)."""
    num_tokens = 0
    for message in messages:
        num_tokens += 4
        for key, value in message.items():
            if value:
                if isinstance(value, list) and key == 'tool_calls':
                    for tool_call in value:
                        if tool_call.get('function'):
                           num_tokens += estimate_tokens(json.dumps(tool_call['function']))
                elif isinstance(value, dict):
                     num_tokens += estimate_tokens(json.dumps(value))
                else:
                    num_tokens += estimate_tokens(str(value))
            if key == "name":
                num_tokens += -1
    num_tokens += 2
    return num_tokens

def run_interactive_script(script_path):
    """
    Runs a script expecting interactive use or TTY using pty.openpty.
    Shows script output to user and captures the final line printed by realpath.
    Returns the final line (the selected path) or None on error/cancellation via script.
    """
    master_fd, slave_fd = pty.openpty()
    process = None
    output_buffer = b""

    try:
        process = subprocess.Popen(
            [script_path],
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            close_fds=True
        )
        os.close(slave_fd)

        inputs = [master_fd]
        while inputs:
            readable, _, _ = select.select(inputs, [], [], 0.05)
            for fd in readable:
                try:
                    data = os.read(fd, 1024)
                    if not data:
                        inputs.remove(fd)
                        break
                    sys.stdout.write(data.decode('utf-8', errors='ignore'))
                    sys.stdout.flush()
                    output_buffer += data
                except OSError:
                     inputs.remove(fd)

        process.wait()

        full_output = output_buffer.decode('utf-8', errors='ignore')
        lines = [line for line in full_output.strip().splitlines() if line.strip()]
        
        # Clean the potential path: remove ANSI escape codes and strip whitespace
        raw_path = lines[-1] if lines else ""
        # Regex to remove ANSI escape codes (like color codes, cursor movements)
        # Corrected regex pattern:
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        selected_path = ansi_escape.sub('', raw_path).strip()

        if process.returncode != 0:
             console.print(f"\n[yellow]Script exited with non-zero status {process.returncode}.[/yellow]")
             return None

        return selected_path

    except Exception as e:
        console.print(f"\n[bold red]Error during interactive script execution:[/bold red] {e}")
        if process and process.poll() is None:
             try:
                 process.kill()
             except OSError:
                 pass
        return None
    finally:
        if 'master_fd' in locals() and master_fd is not None and master_fd >= 0:
             try:
                 os.close(master_fd)
             except OSError:
                 pass

def main():
    console.print(Panel(f"Starting chat with {MODEL}. Type [bold]'!add_context [<path>]'[/], [bold]'exit'[/], or [bold]'quit'[/].", title="Welcome", border_style="green"))
    console.print("--- ")

    while True:
        try:
            user_input = session.prompt("You: ")

            if user_input.lower() in ["exit", "quit"]:
                console.print("[yellow]Exiting chat.[/yellow]")
                break

            if user_input.lower().startswith("!add_context"):
                command_part = "!add_context"
                file_path_arg = user_input[len(command_part):].strip()
                selected_path_for_context = None

                helper_script = "./select_path.sh"

                if not file_path_arg:
                    if not os.path.exists(helper_script) or not os.access(helper_script, os.X_OK):
                         console.print(f"[bold red]Error:[/bold red] Script '{helper_script}' not found or not executable.")
                         selected_path_for_context = None
                    else:
                        selected_path_for_context = run_interactive_script(helper_script)
                        if selected_path_for_context is None:
                             console.print("[red]Path selection failed due to script error or cancellation.[/red]")
                             continue
                        elif not selected_path_for_context:
                            console.print("[yellow]No path selected or selection cancelled.[/yellow]")
                            continue
                        else:
                             console.print(f"Selected path: [cyan]{selected_path_for_context}[/cyan]")

                else:
                    expanded_direct_path = os.path.expanduser(file_path_arg)
                    if os.path.exists(expanded_direct_path):
                         selected_path_for_context = expanded_direct_path
                    else:
                        console.print(f"[bold red]Error:[/bold red] Provided path '[cyan]{expanded_direct_path}[/cyan]' does not exist.")
                        continue

                if selected_path_for_context:
                    combined_content = ""
                    context_source_description = ""
                    content_items_count = 0

                    try:
                        if os.path.isfile(selected_path_for_context):
                            context_source_description = f"file: {os.path.basename(selected_path_for_context)}"
                            with open(selected_path_for_context, 'r', errors='ignore') as f:
                                combined_content = f.read()
                            content_items_count = 1

                        elif os.path.isdir(selected_path_for_context):
                            context_source_description = f"folder: {os.path.basename(selected_path_for_context)}"
                            content_parts = []
                            for root, dirs, files in os.walk(selected_path_for_context):
                                files = [f for f in files if not f.startswith('.')]
                                dirs[:] = [d for d in dirs if not d.startswith('.')]

                                for filename in files:
                                    file_path = os.path.join(root, filename)
                                    relative_path = os.path.relpath(file_path, selected_path_for_context)
                                    try:
                                        with open(file_path, 'r', errors='ignore') as f:
                                            content = f.read()
                                            content_parts.append(f"### File: {relative_path}\n\n{content}\n\n---\n")
                                            content_items_count += 1
                                    except Exception as e:
                                        console.print(f"[yellow]Warning: Could not read file {file_path}: {e}[/yellow]")
                            combined_content = "".join(content_parts)
                        else:
                             console.print(f"[red]Internal Error:[/red] Path '[cyan]{selected_path_for_context}[/cyan]' is neither a file nor a directory after existence check.")
                             combined_content = None

                        if combined_content is not None and combined_content.strip():
                            estimated_context_tokens = estimate_tokens(combined_content)
                            confirm = session.prompt(
                                f"Add context from {context_source_description} ({content_items_count} item(s), approx. {estimated_context_tokens} tokens)? (y/N): "
                            ).lower()

                            if confirm == 'y':
                                context_msg = f"Context from {context_source_description}:\n\n{combined_content}"
                                history.append({"role": "user", "content": context_msg})
                                prompt_tokens_est = estimate_prompt_tokens(history)
                                console.print(f"[dim]Added context. Total prompt tokens: ~{prompt_tokens_est}[/dim]")
                            else:
                                console.print("[yellow]Context adding cancelled by user.[/yellow]")
                        elif combined_content is not None:
                             console.print("[yellow]Selected source contains no readable text content. Context not added.[/yellow]")

                    except Exception as e:
                        console.print(f"[bold red]Error processing context path {selected_path_for_context}:[/bold red] {e}")

                continue

            history.append({"role": "user", "content": user_input})

            prompt_tokens = estimate_prompt_tokens(history)
            console.print(f"[dim](Prompt Tokens: ~{prompt_tokens})[/dim]")

            console.print(f"[bold blue]TARS:[/bold blue] ", end="")

            response = client.chat.completions.create(
                model=MODEL,
                messages=history,
                tools=tools,
                tool_choice="auto",
                stream=True
            )

            stream_content = ""
            tool_calls = []
            finish_reason = None
            response_message = None

            for chunk in response:
                delta = chunk.choices[0].delta
                current_finish_reason = chunk.choices[0].finish_reason
                if current_finish_reason:
                     finish_reason = current_finish_reason

                if delta and delta.content:
                    content_part = delta.content
                    console.print(content_part, end="")
                    stream_content += content_part

                if delta and delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        if tc_delta.index >= len(tool_calls):
                            tool_calls.append({"id": None, "type": "function", "function": {"name": None, "arguments": ""}})
                        current_tc = tool_calls[tc_delta.index]
                        if tc_delta.function and tc_delta.function.arguments:
                            current_tc["function"]["arguments"] += tc_delta.function.arguments
                        if tc_delta.id:
                            current_tc["id"] = tc_delta.id
                        if tc_delta.type:
                            current_tc["type"] = tc_delta.type
                        if tc_delta.function and tc_delta.function.name:
                             current_tc["function"]["name"] = tc_delta.function.name

                if finish_reason is not None:
                     if stream_content or tool_calls:
                         response_message = {"role": "assistant", "content": stream_content if stream_content else None}
                         if tool_calls:
                             final_tool_calls = []
                             for tc in tool_calls:
                                 if tc["id"] and tc["type"] and tc["function"]["name"] and tc["function"]["arguments"] is not None:
                                     final_tool_calls.append({
                                          "id": tc["id"],
                                          "type": tc["type"],
                                          "function": {
                                              "name": tc["function"]["name"],
                                              "arguments": tc["function"]["arguments"]
                                          }
                                     })
                                 else:
                                     console.print(f"[yellow]Warning: Incomplete tool call delta received: {tc}[/yellow]")
                             if final_tool_calls:
                                response_message["tool_calls"] = final_tool_calls

                     break

            console.print()

            completion_tokens = estimate_tokens(stream_content) if stream_content else 0

            if finish_reason == "tool_calls" and response_message and response_message.get("tool_calls"):
                history.append(response_message)

                tool_results = []
                for tool_call in response_message["tool_calls"]:
                    function_name = tool_call["function"]["name"]
                    tool_call_id = tool_call["id"]
                    tool_output = ""

                    if function_name == "write_file":
                        try:
                            args = json.loads(tool_call["function"]["arguments"])
                            llm_suggested_path = args.get("path", "")
                            content = args.get("content")

                            if content is None:
                                console.print(f"[bold red]Error:[/bold red] Missing content for write_file tool.")
                                tool_output = json.dumps({"error": "missing required argument: content"})
                            else:
                                try:
                                    lexer = Syntax.guess_lexer(llm_suggested_path)
                                except Exception:
                                    lexer = "text"
                                console.print(Panel(Syntax(content, lexer=lexer, theme="monokai", line_numbers=True), title="File Content", border_style="blue"))

                                prompt_display_text = (
                                    f"LLM suggested path: '[cyan]{llm_suggested_path}[/cyan]'\n"
                                    f"Enter the correct full path, type [bold yellow]!browse[/bold yellow] to select interactively, or leave blank to use suggestion: "
                                )
                                console.print(prompt_display_text, end="")
                                user_path_input = session.prompt("").strip()

                                final_path = llm_suggested_path

                                if user_path_input == "!browse":
                                    console.print("[yellow]Launching interactive path selector...[/yellow]")
                                    helper_script = "./select_path.sh"
                                    if not os.path.exists(helper_script) or not os.access(helper_script, os.X_OK):
                                        console.print(f"[bold red]Error:[/bold red] Script '{helper_script}' not found or not executable.")
                                        final_path = None
                                    else:
                                        selected_path = run_interactive_script(helper_script)
                                        if selected_path is None:
                                            console.print("[red]Path selection failed due to script error or cancellation.[/red]")
                                            final_path = None
                                        elif selected_path:
                                            final_path = selected_path
                                            console.print(f"Selected path: [cyan]{final_path}[/cyan]")
                                        else:
                                            console.print("[yellow]No path selected or selection cancelled.[/yellow]")

                                elif user_path_input:
                                    final_path = user_path_input

                                if final_path is None:
                                     console.print("[yellow]File writing skipped due to invalid path selection/input.[/yellow]")
                                     tool_output = json.dumps({"success": False, "message": "write path selection failed or cancelled"})
                                     tool_results.append({
                                        "role": "tool",
                                        "tool_call_id": tool_call_id,
                                        "name": function_name,
                                        "content": tool_output,
                                    })
                                     continue

                                expanded_path = os.path.expanduser(final_path)

                                path_to_write = None
                                write_mode_description = ""

                                if os.path.isdir(expanded_path):
                                    console.print(f"Target '[cyan]{expanded_path}[/cyan]' is a directory.")
                                    suggested_filename = os.path.basename(llm_suggested_path) if llm_suggested_path and not os.path.isdir(llm_suggested_path) else "new_file.txt"
                                    filename_prompt_text = f"Enter filename to save inside this directory (default: '[yellow]{suggested_filename}[/yellow]'): "
                                    console.print(prompt_display_text, end="")
                                    filename = session.prompt("").strip()
                                    if not filename:
                                        filename = suggested_filename
                                    path_to_write = os.path.join(expanded_path, filename)
                                    if os.path.exists(path_to_write):
                                         write_mode_description = f"Overwrite existing file '[cyan]{path_to_write}[/cyan]' inside directory?"
                                    else:
                                         write_mode_description = f"Write new file '[cyan]{path_to_write}[/cyan]' inside directory?"

                                elif os.path.isfile(expanded_path):
                                    path_to_write = expanded_path
                                    write_mode_description = f"Overwrite existing file '[cyan]{path_to_write}[/cyan]'?"
                                else:
                                    path_to_write = expanded_path
                                    write_mode_description = f"Write new file '[cyan]{path_to_write}[/cyan]'?"

                                if path_to_write:
                                    confirm_prompt_text = f"{write_mode_description} (y/N): "
                                    console.print(prompt_display_text, end="")
                                    confirm = session.prompt("").lower()

                                    if confirm == 'y':
                                        try:
                                            parent_dir = os.path.dirname(path_to_write)
                                            if parent_dir:
                                                 os.makedirs(parent_dir, exist_ok=True)

                                            with open(path_to_write, 'w') as f:
                                                f.write(content)
                                            console.print(f"[green]File '{path_to_write}' written successfully.[/green]")
                                            tool_output = json.dumps({"success": True, "message": f"File {path_to_write} written."})
                                        except Exception as e:
                                            console.print(f"[bold red]Error writing file {path_to_write}:[/bold red] {e}")
                                            tool_output = json.dumps({"success": False, "error": str(e)})
                                    else:
                                        console.print("[yellow]Write operation cancelled by user.[/yellow]")
                                        tool_output = json.dumps({"success": False, "message": "user denied request"})
                                else:
                                     console.print("[red]Error: Failed to determine final write path from selected/input path.[/red]")
                                     tool_output = json.dumps({"success": False, "message": "failed to determine final write path"})
                                tool_results.append({
                                    "role": "tool",
                                    "tool_call_id": tool_call_id,
                                    "name": function_name,
                                    "content": tool_output,
                                })

                        except json.JSONDecodeError:
                            console.print("[bold red]Error:[/bold red] Could not decode tool arguments.")
                            tool_output = json.dumps({"error": "invalid json arguments"})
                            tool_results.append({
                                "role": "tool",
                                "tool_call_id": tool_call_id,
                                "name": function_name,
                                "content": tool_output,
                            })
                        except Exception as e:
                            console.print(f"[bold red]Error processing tool call:[/bold red] {e}")
                            tool_output = json.dumps({"error": f"failed to process tool: {str(e)}"})
                            tool_results.append({
                                "role": "tool",
                                "tool_call_id": tool_call_id,
                                "name": function_name,
                                "content": tool_output,
                            })
                    else:
                        console.print(f"[bold red]Error:[/bold red] Unknown tool function '{function_name}'")
                        tool_output = json.dumps({"error": f"unknown tool {function_name}"})
                        tool_results.append({
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "name": function_name,
                            "content": tool_output,
                        })


                if tool_results:
                    history.extend(tool_results)

                    console.print("[bold blue]Assistant (after tool execution):[/bold blue] ", end="")

                    prompt_tokens_tool = estimate_prompt_tokens(history)
                    console.print(f"\n[dim](Prompt Tokens: ~{prompt_tokens_tool})[/dim]")

                    second_response = client.chat.completions.create(
                        model=MODEL,
                        messages=history,
                        stream=True
                    )

                    final_content = ""
                    for chunk in second_response:
                        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                            content_part = chunk.choices[0].delta.content
                            console.print(content_part, end="")
                            final_content += content_part

                    console.print()
                    completion_tokens_tool = estimate_tokens(final_content)
                    if final_content:
                        history.append({"role": "assistant", "content": final_content})

                    console.print(f"[dim](Completion Tokens: ~{completion_tokens_tool}, Total: ~{prompt_tokens_tool + completion_tokens_tool})[/dim]")
                else:
                     console.print("[yellow]Skipping second API call as no tool outputs were generated (tool execution likely failed early or was cancelled).[/yellow]")

            elif stream_content:
                history.append({"role": "assistant", "content": stream_content})
                console.print(f"[dim](Completion Tokens: ~{completion_tokens}, Total: ~{prompt_tokens + completion_tokens})[/dim]")
            elif finish_reason != "stop" and finish_reason != "tool_calls":
                 console.print(f"[yellow][Stream finished unexpectedly with reason: {finish_reason}][/yellow]")
            elif not stream_content and not response_message:
                console.print("[yellow][No content or tool call received from assistant][/yellow]")

        except OpenAIError as e:
            console.print(f"\n[bold red]An API error occurred:[/bold red] {e}")
        except KeyboardInterrupt:
            console.print("\n[yellow]Exiting chat.[/yellow]")
            break
        except Exception as e:
            console.print(f"\n[bold red]An unexpected error occurred:[/bold red] {e}")
            console.print_exception(show_locals=True)
            break

if __name__ == "__main__":
    main()
