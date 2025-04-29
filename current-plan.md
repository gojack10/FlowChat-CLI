# OpenRouter CLI Tool Plan

## Idea Evaluation Tournament (Initial)

- [X] Evaluate six potential approaches for the CLI tool using a round-robin tournament, prioritizing robustness (weight 2) and then ease of implementation (weight 1).
- [X] **Chosen Idea**: `openai` SDK + Loop.

## Implementation Outline

1.  **Setup**:
    *   [X] Create `openrouter_cli.py`.
    *   [X] Create `requirements.txt` with `openai`.
    *   [X] Create `README.md`.
    *   [X] Create `changelog.md`.
    *   [X] Create `current-plan.md`.
2.  **Core Logic (`openrouter_cli.py`)**:
    *   [X] Import necessary libraries (`openai`, `os`, `sys`).
    *   [X] Load `OPENROUTER_API_KEY` from environment variables. Exit if not found.
    *   [X] Initialize `OpenAI` client pointing to OpenRouter's `base_url`.
    *   [X] Implement a chat history list.
    *   [X] Start a `while True` loop for user interaction.
    *   [X] Get user input. Handle exit condition.
    *   [X] Append user message to history.
    *   [X] Call `client.chat.completions.create` with `model`, `messages` (history), and `stream=True`.
    *   [X] Iterate through the stream response chunks.
    *   [X] Print content deltas to `stdout` immediately and flush.
    *   [X] Accumulate the full response from deltas.
    *   [X] Append the full assistant response to history.
    *   [X] Include basic error handling (e.g., `try...except` around the API call).
3.  **Documentation**:
    *   [X] Write basic setup and usage instructions in `README.md`.
    *   [X] Add the first entry to `changelog.md`.
4.  **.env Support**:
    *   [X] Add `python-dotenv` to `requirements.txt`.
    *   [X] Modify `openrouter_cli.py` to load `.env`.
    *   [X] Update `changelog.md`.
5.  **Feature Enhancements**:
    *   [X] Added `/add_context <path>` command.
    *   [X] Integrated `tiktoken` for token estimation.
    *   [X] Implemented Tool Calling for `write_file`.
    *   [X] Fixed `write_file` path handling (`~` expansion, directory creation).

## UI/UX Enhancements

1.  **Path Selection (Tournament - User Driven)**:
    *   [X] Evaluate approaches for user-driven path selection (Confirmation w/ Edit, Explicit Prompting, Pause & Prompt, etc.).
    *   [X] Evaluate approaches for integrating shell tools (`prompt_toolkit`, `fzf` subprocess, external script, etc.).
    *   [X] **Chosen Idea**: External Helper Script (`select_path.sh` using `fzf`).
    *   [ ] Create `select_path.sh`.
    *   [ ] Modify `write_file` tool confirmation in `openrouter_cli.py` to optionally call `select_path.sh`.
2.  **Rich Integration**:
    *   [ ] Add `rich` to `requirements.txt`.
    *   [ ] Integrate `rich` for console output (e.g., `Console`, `Panel`, `Markdown`).
    *   [ ] Replace `input()` with `Console.input()` if needed for better integration.
3.  **Documentation & Changelog**:
    *   [ ] Update `README.md` for `rich` and `fzf` usage.
    *   [ ] Update `changelog.md` for UI/UX enhancements.

**Ideas:**

1.  **Basic `requests` + Loop**: Direct API calls with manual streaming handling in a loop.
2.  **`openai` SDK + Loop**: Use the `openai` SDK (configured for OpenRouter) with its built-in streaming in a loop.
3.  **`langchain` + `ChatOpenAI`**: Use LangChain for potentially better chat management, configured for OpenRouter.
4.  **`pydantic-ai` + Loop**: Use `pydantic-ai` configured for OpenRouter.
5.  **`typer` + `openai` SDK**: Build a structured CLI using `typer` on top of the `openai` SDK.
6.  **`rich` + `openai` SDK**: Enhance the UI using `rich` alongside the `openai` SDK.

**Tournament Results (Simulated):**

1.  Idea 2 (`openai` SDK + Loop): 10 points 