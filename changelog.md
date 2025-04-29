# Changelog

1. Initial project setup: Created `openrouter_cli.py`, `requirements.txt`, `README.md`, `changelog.md`, and `current-plan.md`. Implemented basic CLI structure with streaming chat using the `openai` SDK configured for OpenRouter.
2. Added `.env` file support: Integrated `python-dotenv` to automatically load the `OPENROUTER_API_KEY` from a `.env` file.
3. Feature Enhancements:
    - Added `/add_context <path>` command to load file content into history mid-conversation.
    - Integrated `tiktoken` for estimating and displaying prompt/completion token counts.
    - Refactored file writing to use OpenAI/OpenRouter Tool Calling:
        - Defined `write_file` tool schema.
        - Updated API calls and response handling for tool calls.
        - Implemented user confirmation prompt for file writes.
        - Updated system prompt for tool guidance.
4. Fixed `write_file` tool path handling: Added `os.path.expanduser` to correctly resolve `~` and `os.makedirs` to create parent directories if needed. 