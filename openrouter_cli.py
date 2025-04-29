import os
import sys
import json
import subprocess # needed to run the helper script
import traceback # for debugging
from openai import OpenAI, OpenAIError
from dotenv import load_dotenv
import tiktoken
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax

# --- Rich Console Initialization ---
console = Console()

# load environment variables from .env file
load_dotenv()

# load the api key from environment variables
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    console.print("[bold red]Error:[/bold red] OPENROUTER_API_KEY environment variable not set.")
    sys.exit(1)

# initialize the openai client configured for openrouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

# default model (can be changed)
MODEL = "google/gemini-2.5-flash-preview:thinking" # using a model known to support tool calling well

# get encoding for the model (or a close approximation)
try:
    encoding = tiktoken.encoding_for_model(MODEL.split(':')[0]) # handle potential :thinking suffix
except KeyError:
    encoding = tiktoken.get_encoding("cl100k_base") # default for many recent models

# --- tool definition ---
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
tool_map = {"write_file": write_file_tool} # map name to definition for easy lookup if needed

# maintain conversation history
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
                # handle tool call dicts specifically
                if isinstance(value, list) and key == 'tool_calls':
                    for tool_call in value:
                        if tool_call.get('function'):
                           num_tokens += estimate_tokens(json.dumps(tool_call['function']))
                elif isinstance(value, dict):
                     num_tokens += estimate_tokens(json.dumps(value)) # generic dict handling
                else:
                    num_tokens += estimate_tokens(str(value))
            if key == "name":
                num_tokens += -1
    num_tokens += 2
    return num_tokens

console.print(Panel(f"Starting chat with [bold cyan]{MODEL}[/]. Type [bold]'/add_context <path>'[/], [bold]'exit'[/], or [bold]'quit'[/].", title="Welcome", border_style="green"))
console.print("--- ")

while True:
    try:
        # get user input using rich console
        user_input = console.input("[bold green]You:[/bold green] ")
        if user_input.lower() in ["exit", "quit"]:
            console.print("[yellow]Exiting chat.[/yellow]")
            break

        # check for special commands
        if user_input.lower().startswith("/add_context"):
            file_path_arg = user_input[len("/add_context"):].strip()
            selected_context_path = None

            if not file_path_arg:
                # no path provided, use the browser
                console.print("[yellow]Launching interactive path selector for context file...[/yellow]")
                try:
                    helper_script = "./select_path.sh"
                    if not os.path.exists(helper_script):
                         raise FileNotFoundError("select_path.sh not found in current directory.")
                    if not os.access(helper_script, os.X_OK):
                         raise PermissionError("select_path.sh is not executable (run chmod +x select_path.sh)")

                    process = subprocess.run([helper_script], capture_output=True, text=True, check=False)

                    if process.returncode != 0:
                        console.print(f"[red]Error running path selector script:[/red]")
                        console.print(f"[dim]Stderr: {process.stderr}[/dim]")
                    else:
                        selected_path_output = process.stdout.strip()
                        if selected_path_output:
                            # check if the selected path is a file
                            if os.path.isfile(selected_path_output):
                                selected_context_path = selected_path_output
                                console.print(f"Selected context file: [cyan]{selected_context_path}[/cyan]")
                            else:
                                console.print(f"[red]Error:[/red] Selected path '[cyan]{selected_path_output}[/cyan]' is not a file.")
                        else:
                            console.print("[yellow]No path selected or selection cancelled.[/yellow]")

                except FileNotFoundError as e:
                    console.print(f"[bold red]Error:[/bold red] {e}")
                except PermissionError as e:
                     console.print(f"[bold red]Error:[/bold red] {e}")
                except Exception as e:
                    console.print(f"[bold red]Error running browse script:[/bold red] {e}")
            else:
                # path provided directly, use existing logic
                expanded_direct_path = os.path.expanduser(file_path_arg)
                if os.path.isfile(expanded_direct_path):
                     selected_context_path = expanded_direct_path
                else:
                    console.print(f"[bold red]Error:[/bold red] Provided path '[cyan]{expanded_direct_path}[/cyan]' is not a valid file.")

            # if we have a valid file path (either selected or provided)
            if selected_context_path:
                try:
                    with open(selected_context_path, 'r') as f:
                        context_content = f.read()
                    context_msg = f"Context from {os.path.basename(selected_context_path)}:\n\n{context_content}"
                    history.append({"role": "user", "content": context_msg})
                    prompt_tokens_est = estimate_prompt_tokens(history)
                    console.print(f"[dim]Added context from {selected_context_path}. Estimated prompt tokens: ~{prompt_tokens_est}[/dim]")
                except Exception as e:
                    console.print(f"[bold red]Error reading context file {selected_context_path}:[/bold red] {e}")
            
            continue # continue loop whether context was added or error occurred

        # append user message to history
        history.append({"role": "user", "content": user_input})

        prompt_tokens = estimate_prompt_tokens(history)
        console.print(f"[dim](Prompt Tokens: ~{prompt_tokens})[/dim]")

        console.print(f"[bold blue]Assistant:[/bold blue] ", end="")
        # sys.stdout.flush() # not strictly needed with rich print?

        # --- first api call --- 
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

        # process the stream from the first call
        for chunk in response:
            delta = chunk.choices[0].delta
            # Handle potential missing fields gracefully
            current_finish_reason = chunk.choices[0].finish_reason
            if current_finish_reason:
                 finish_reason = current_finish_reason # update finish_reason if available

            # accumulate content delta
            if delta and delta.content:
                content_part = delta.content
                console.print(content_part, end="") # stream directly to rich console
                stream_content += content_part

            # accumulate tool calls delta
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
                 # construct the message object from accumulated data
                 if stream_content or tool_calls:
                     response_message = {"role": "assistant", "content": stream_content if stream_content else None} # content can be null if only tool call
                     if tool_calls:
                         # ensure arguments are fully formed before adding
                         final_tool_calls = []
                         for tc in tool_calls:
                             # check if all required parts are present
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
                     
                 break # exit stream loop

        console.print() # ensure newline after streaming

        # --- handle response (tool call or regular message) ---
        completion_tokens = estimate_tokens(stream_content) if stream_content else 0

        if finish_reason == "tool_calls" and response_message and response_message.get("tool_calls"):
            # console.print("[dim](Tool Call Requested)[/dim]")
            history.append(response_message)

            tool_results = []
            for tool_call in response_message["tool_calls"]:
                function_name = tool_call["function"]["name"]
                tool_call_id = tool_call["id"]
                tool_output = ""
                
                if function_name == "write_file":
                    try:
                        args = json.loads(tool_call["function"]["arguments"])
                        llm_suggested_path = args.get("path", "") # default to empty string
                        content = args.get("content")
                        
                        if content is None: # path can be empty initially, but content must exist
                            console.print(f"[bold red]Error:[/bold red] Missing content for write_file tool.")
                            tool_output = json.dumps({"error": "missing required argument: content"})
                        else:
                            # determine lexer based on suggested path extension
                            # lexer = Syntax.guess_lexer(llm_suggested_path, default="text") # Error: default is not a valid arg
                            try:
                                # guess_lexer might raise ClassNotFound if no lexer is found
                                lexer = Syntax.guess_lexer(llm_suggested_path)
                            except Exception: # Catch potential errors from pygments
                                lexer = "text" # fallback to plain text
                            console.print(Panel(Syntax(content, lexer=lexer, theme="monokai", line_numbers=True), title="File Content", border_style="blue"))
                            
                            # --- Path Confirmation --- 
                            prompt_text = (
                                f"LLM suggested path: '[cyan]{llm_suggested_path}[/cyan]'\n"
                                f"Enter the correct full path, type [bold yellow]!browse[/bold yellow] to select interactively, or leave blank to use suggestion: "
                            )
                            user_path_input = console.input(prompt_text).strip()
                            
                            final_path = llm_suggested_path # default
                            
                            if user_path_input == "!browse":
                                console.print("[yellow]Launching interactive path selector...[/yellow]")
                                try:
                                    # ensure helper script exists and is executable
                                    helper_script = "./select_path.sh"
                                    if not os.path.exists(helper_script):
                                         raise FileNotFoundError("select_path.sh not found in current directory.")
                                    if not os.access(helper_script, os.X_OK):
                                         raise PermissionError("select_path.sh is not executable (run chmod +x select_path.sh)")

                                    # use subprocess.run for better error handling & capture
                                    process = subprocess.run([helper_script], capture_output=True, text=True, check=False) # check=false allows us to see stderr
                                    
                                    if process.returncode != 0:
                                        console.print(f"[red]Error running path selector script:[/red]")
                                        console.print(f"[dim]Stderr: {process.stderr}[/dim]")
                                        final_path = None # Indicate failure
                                    else:
                                        selected_path = process.stdout.strip()
                                        if selected_path:
                                            final_path = selected_path
                                            console.print(f"Selected path: [cyan]{final_path}[/cyan]")
                                        else:
                                            console.print("[yellow]No path selected or selection cancelled.[/yellow]")
                                            final_path = None # Indicate cancellation
                                            
                                except FileNotFoundError as e:
                                    console.print(f"[bold red]Error:[/bold red] {e}")
                                    final_path = None
                                except PermissionError as e:
                                     console.print(f"[bold red]Error:[/bold red] {e}")
                                     final_path = None
                                except Exception as e:
                                    console.print(f"[bold red]Error running browse script:[/bold red] {e}")
                                    final_path = None
                            
                            elif user_path_input: # user typed a path directly
                                final_path = user_path_input
                            
                            # Proceed only if we have a valid path selection/input
                            if final_path:
                                expanded_path = os.path.expanduser(final_path)
                                
                                # --- Check if path is file, directory, or doesn't exist --- 
                                path_to_write = None
                                write_mode_description = ""
                                
                                if os.path.isdir(expanded_path):
                                    # target is a directory
                                    console.print(f"Target '[cyan]{expanded_path}[/cyan]' is a directory.")
                                    suggested_filename = os.path.basename(llm_suggested_path) if llm_suggested_path and not os.path.isdir(llm_suggested_path) else "new_file.txt"
                                    filename = console.input(f"Enter filename to save inside this directory (default: '[yellow]{suggested_filename}[/yellow]'): ").strip()
                                    if not filename:
                                        filename = suggested_filename
                                    path_to_write = os.path.join(expanded_path, filename)
                                    if os.path.exists(path_to_write):
                                         write_mode_description = f"Overwrite existing file '[cyan]{path_to_write}[/cyan]' inside directory?"
                                    else:
                                         write_mode_description = f"Write new file '[cyan]{path_to_write}[/cyan]' inside directory?"
                                
                                elif os.path.isfile(expanded_path):
                                    # target is an existing file
                                    path_to_write = expanded_path
                                    write_mode_description = f"Overwrite existing file '[cyan]{path_to_write}[/cyan]'?"
                                else:
                                    # target doesn't exist, treat as a new file path
                                    path_to_write = expanded_path
                                    # check if parent dir exists, create if not (handled later)
                                    write_mode_description = f"Write new file '[cyan]{path_to_write}[/cyan]'?"

                                # --- Final Confirmation --- 
                                if path_to_write:
                                    confirm = console.input(f"{write_mode_description} (y/N): ").lower()
                                    
                                    if confirm == 'y':
                                        try:
                                            # Ensure parent directory exists before writing
                                            parent_dir = os.path.dirname(path_to_write)
                                            if parent_dir: # Handle case where path is just filename in cwd
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
                                     # this case should ideally not be reached if initial path selection was valid
                                     console.print("[red]Error determining final write path.[/red]")
                                     tool_output = json.dumps({"success": False, "message": "failed to determine valid write path"})
                            else:
                                # path selection failed or was cancelled (!browse returned nothing, or initial path was invalid)
                                console.print("[yellow]File writing skipped due to invalid path or cancellation.[/yellow]")
                                tool_output = json.dumps({"success": False, "message": "path selection failed or cancelled"})

                    except json.JSONDecodeError:
                        console.print("[bold red]Error:[/bold red] Could not decode tool arguments.")
                        tool_output = json.dumps({"error": "invalid json arguments"})
                    except Exception as e:
                        console.print(f"[bold red]Error processing tool call:[/bold red] {e}")
                        # traceback.print_exc() # uncomment for debugging
                        tool_output = json.dumps({"error": f"failed to process tool: {str(e)}"})
                else:
                    console.print(f"[bold red]Error:[/bold red] Unknown tool function '{function_name}'")
                    tool_output = json.dumps({"error": f"unknown tool {function_name}"})

                tool_results.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": function_name,
                    "content": tool_output,
                })
            
            # only extend history if tool results were generated
            if tool_results:
                history.extend(tool_results)

                # --- second api call (after tool execution) --- 
                console.print("[bold blue]Assistant (after tool execution):[/bold blue] ", end="")
                # sys.stdout.flush()
                
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
                
                console.print() # newline after streaming
                completion_tokens_tool = estimate_tokens(final_content)
                if final_content:
                    history.append({"role": "assistant", "content": final_content})
                
                console.print(f"[dim](Completion Tokens: ~{completion_tokens_tool}, Total: ~{prompt_tokens_tool + completion_tokens_tool})[/dim]")
            else:
                 console.print("[yellow]Skipping second API call as tool execution failed or was cancelled.[/yellow]")

        elif stream_content: # regular message without tool calls
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
        # Use rich traceback printing
        console.print_exception(show_locals=True)
        break 