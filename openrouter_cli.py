import os
import sys
import json # needed for tool calling argument parsing
from openai import OpenAI, OpenAIError
from dotenv import load_dotenv
import tiktoken # for token counting

# load environment variables from .env file
load_dotenv()

# load the api key from environment variables
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    print("Error: OPENROUTER_API_KEY environment variable not set.")
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
        "description": "Write content to a specified file path. Use this when the user explicitly asks to save or write something to a file.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "the relative or absolute path for the file to be written."
                },
                "content": {
                    "type": "string",
                    "description": "the full content to write into the file."
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
            "You are a helpful assistant named 'TARS;. You can write files using the `write_file` tool. "
            "When asked to write a file, confirm the path and content clearly using the tool. "
            "Be concise in your regular responses while retaining as much information as possible."
        )
    }
]

def estimate_tokens(text):
    """estimates the number of tokens in a string."""
    return len(encoding.encode(text))

def estimate_prompt_tokens(messages):
    """estimates the total tokens in the message history (prompt)."""
    # approximation based on openai cookbook, accounting for message structure
    num_tokens = 0
    for message in messages:
        num_tokens += 4 # every message follows <im_start>{role/name}\n{content}<im_end>\n
        for key, value in message.items():
            if value: # ensure value is not none
                num_tokens += estimate_tokens(str(value))
            if key == "name": # if there's a name, the role is omitted
                num_tokens += -1 # role is always required and always 1 token
    num_tokens += 2 # every reply is primed with <im_start>assistant

    return num_tokens

print(f"Starting chat with {MODEL}. Type '/add_context <path>', 'exit', or 'quit'.")
print("---")

while True:
    try:
        # get user input
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting chat.")
            break

        # check for special commands
        if user_input.lower().startswith("/add_context "):
            try:
                file_path = user_input[len("/add_context "):].strip()
                if not os.path.exists(file_path):
                    print(f"Error: File not found: {file_path}")
                    continue # ask for input again
                with open(file_path, 'r') as f:
                    context_content = f.read()
                context_msg = f"Context from {os.path.basename(file_path)}:\n\n{context_content}"
                history.append({"role": "user", "content": context_msg})
                print(f"Added context from {file_path}. Estimated prompt tokens: {estimate_prompt_tokens(history)}")
                continue # skip api call for this turn, just added context
            except Exception as e:
                print(f"Error reading context file {file_path}: {e}")
                continue

        # append user message to history
        history.append({"role": "user", "content": user_input})

        # estimate and print prompt tokens before the call
        prompt_tokens = estimate_prompt_tokens(history)
        print(f"(Prompt Tokens: ~{prompt_tokens})")

        print(f"Assistant: ", end="")
        sys.stdout.flush() # ensure 'assistant:' prints immediately

        # --- first api call --- 
        response = client.chat.completions.create(
            model=MODEL,
            messages=history,
            tools=tools,
            tool_choice="auto", # let the model decide if it needs tools
            stream=True
        )

        stream_content = ""
        tool_calls = []
        finish_reason = None
        response_message = None # store the final message object from the stream

        # process the stream from the first call
        for chunk in response:
            delta = chunk.choices[0].delta
            finish_reason = chunk.choices[0].finish_reason

            # accumulate content delta
            if delta and delta.content:
                content_part = delta.content
                print(content_part, end="")
                sys.stdout.flush()
                stream_content += content_part

            # accumulate tool calls delta
            if delta and delta.tool_calls:
                # important: tool call deltas arrive incrementally!
                for tc_delta in delta.tool_calls:
                    # initialize tool call object if it's the first delta for this index
                    if tc_delta.index >= len(tool_calls):
                        tool_calls.append({"id": None, "type": "function", "function": {"name": None, "arguments": ""}})
                    
                    current_tc = tool_calls[tc_delta.index]
                    # append arguments incrementally
                    if tc_delta.function and tc_delta.function.arguments:
                        current_tc["function"]["arguments"] += tc_delta.function.arguments
                    # set id, name, type on first appearance
                    if tc_delta.id:
                        current_tc["id"] = tc_delta.id
                    if tc_delta.type:
                        current_tc["type"] = tc_delta.type
                    if tc_delta.function and tc_delta.function.name:
                         current_tc["function"]["name"] = tc_delta.function.name

            # store the message object when the stream finishes
            if finish_reason is not None:
                # the message object can sometimes be found in the chunk when finished
                # but it's more reliable to construct it from accumulated deltas
                if stream_content or tool_calls: # only construct if we got something
                     response_message = {
                         "role": "assistant",
                         "content": stream_content,
                         # construct tool_calls list from accumulated deltas
                         "tool_calls": [
                             {
                                 "id": tc["id"],
                                 "type": tc["type"],
                                 "function": {
                                     "name": tc["function"]["name"],
                                     "arguments": tc["function"]["arguments"]
                                 }
                             }
                             for tc in tool_calls
                         ] if tool_calls else None
                     }
                break # exit the loop once stream is finished

        print() # newline after streaming content or tool call indicators

        # --- handle response (tool call or regular message) ---
        completion_tokens = estimate_tokens(stream_content) if stream_content else 0

        if finish_reason == "tool_calls" and response_message and response_message["tool_calls"]:
            print("(Tool Call Requested)")
            # append the assistant's request message to history
            history.append(response_message)

            # execute tools and collect results
            tool_results = []
            for tool_call in response_message["tool_calls"]:
                function_name = tool_call["function"]["name"]
                tool_call_id = tool_call["id"]
                
                if function_name == "write_file":
                    try:
                        args = json.loads(tool_call["function"]["arguments"])
                        path = args.get("path")
                        content = args.get("content")

                        if not path or content is None: # check for missing args
                            print(f"Error: Missing path or content for write_file tool.")
                            tool_output = json.dumps({"error": "missing required arguments: path and content"})
                        else:
                            # expand the tilde to the user's home directory
                            expanded_path = os.path.expanduser(path)

                            print(f"\nTool Request: Write file '{expanded_path}' (Original: '{path}')")
                            print("Content:")
                            print("---------")
                            print(content)
                            print("---------")
                            confirm = input("Allow write? (y/N): ")

                            if confirm.lower() == 'y':
                                try:
                                    # create directory if it doesn't exist
                                    os.makedirs(os.path.dirname(expanded_path), exist_ok=True)
                                    with open(expanded_path, 'w') as f:
                                        f.write(content)
                                    print(f"File '{expanded_path}' written successfully.")
                                    tool_output = json.dumps({"success": True, "message": f"File {expanded_path} written."}) 
                                except Exception as e:
                                    print(f"Error writing file {expanded_path}: {e}")
                                    tool_output = json.dumps({"success": False, "error": str(e)})
                            else:
                                print("Write operation cancelled by user.")
                                tool_output = json.dumps({"success": False, "message": "user denied request"})

                    except json.JSONDecodeError:
                        print("Error: Could not decode tool arguments.")
                        tool_output = json.dumps({"error": "invalid json arguments"})
                    except Exception as e:
                        print(f"Error processing tool call: {e}")
                        tool_output = json.dumps({"error": f"failed to process tool: {str(e)}"})
                else:
                    print(f"Error: Unknown tool function '{function_name}'")
                    tool_output = json.dumps({"error": f"unknown tool {function_name}"})

                # append tool result message to history
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": function_name,
                    "content": tool_output, 
                })
                history.extend(tool_results)

            # --- second api call (after tool execution) --- 
            print("Assistant (after tool execution): ", end="")
            sys.stdout.flush()
            
            prompt_tokens_tool = estimate_prompt_tokens(history)
            print(f"\n(Prompt Tokens: ~{prompt_tokens_tool})")

            second_response = client.chat.completions.create(
                model=MODEL,
                messages=history,
                stream=True # stream the final response too
            )

            final_content = ""
            for chunk in second_response:
                 if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    content_part = chunk.choices[0].delta.content
                    print(content_part, end="")
                    sys.stdout.flush()
                    final_content += content_part
            
            print() # newline after streaming
            completion_tokens_tool = estimate_tokens(final_content)
            if final_content:
                 history.append({"role": "assistant", "content": final_content})
            
            print(f"(Completion Tokens: ~{completion_tokens_tool}, Total: ~{prompt_tokens_tool + completion_tokens_tool})")

        elif stream_content: # regular message without tool calls
            # append the full assistant response to history
            history.append({"role": "assistant", "content": stream_content})
            print(f"(Completion Tokens: ~{completion_tokens}, Total: ~{prompt_tokens + completion_tokens})")
        elif finish_reason != "stop" and finish_reason != "tool_calls":
             print(f"[Stream finished unexpectedly with reason: {finish_reason}]")
        elif not stream_content and not tool_calls:
            print("[No content or tool call received from assistant]") 

    except OpenAIError as e:
        print(f"\nAn API error occurred: {e}")
        # decide if you want to clear history or retry
        # for now, we just print the error and continue the loop
    except KeyboardInterrupt:
        print("\nExiting chat.")
        break
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        # traceback.print_exc() # uncomment for debugging
        break # exit on unexpected errors 