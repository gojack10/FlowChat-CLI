# FlowChat CLI

A simple command-line interface to interact with OpenRouter LLMs, featuring streaming, conversation history, token tracking, file writing via Tool Calling, and enhanced UI with `rich`.

## Features

*   Interactive chat with conversation history.
*   Streaming responses.
*   Loads API key from `.env` file.
*   `!add_context [<path>]` command to load file or folder content into the chat (prompts for confirmation with token count).
*   Token usage estimation (`tiktoken`).
*   File writing capability via LLM Tool Calling (with user confirmation).
*   Interactive file path selection using `fzf` (optional) via `!browse` command during file write confirmation.
*   Enhanced terminal UI using `rich`.

## Setup

1.  **Clone the repository (if applicable)**
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Install `fzf` (Optional, for `!browse`):**
    *   Follow instructions at [https://github.com/junegunn/fzf#installation](https://github.com/junegunn/fzf#installation)
4.  **Make helper script executable:**
    ```bash
    chmod +x select_path.sh
    ```
5.  **Create `.env` file:**
    Create a file named `.env` in the same directory and add your API key:
    ```
    OPENROUTER_API_KEY='your-api-key'
    ```

## Usage

```bash
python openrouter_cli.py
```

*   Chat normally.
*   Use `!add_context <path>` to add a specific file or folder.
*   Use `!add_context` (with no path) to open an interactive selector (`fzf` if installed) to choose a file or folder.
*   The script will estimate the tokens and ask for confirmation before adding the context.
*   When asked to write a file, the script will display the content and prompt for path confirmation:
    *   **LLM Suggestion:** Shows the path the LLM suggested.
    *   **Enter Path:** You can type a full path (e.g., `/path/to/your/file.txt`) or a path relative to the current directory (e.g., `my_folder/file.py`). Tilde `~` is expanded.
    *   **Use Suggestion:** Leave blank and press Enter to accept the LLM's suggested path.
    *   **Interactive Browse:** Type `!browse` and press Enter to use `fzf` (if installed) for interactive path selection.
    *   **Directory Handling:** If the confirmed/selected path is a directory, you will be prompted to enter a filename to save inside that directory (defaults based on LLM suggestion).
    *   **Overwrite Confirmation:** If the final path points to an existing file, you will be asked to confirm overwriting it.
*   Type 'exit' or 'quit' to end the session. 