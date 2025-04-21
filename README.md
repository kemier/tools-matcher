# Tools Matcher Embedding Server

This project provides an MCP (Model Context Protocol) server that offers a tool for calculating the semantic relevance of other tools based on user input.

## Description

The server implements a single MCP tool endpoint:

*   **`get-relevant-tools`**: Takes a user prompt, conversation history, and a list of available tools (with descriptions) as input. It uses a sentence embedding model (currently `sentence-transformers/LaBSE`) to calculate the semantic similarity between the user's context and each tool's description. It returns a ranked list of the most relevant tool names as a JSON string array.

This allows an MCP client or agent to determine which tools might be most appropriate to call for a given user request.

## Setup

1.  **Prerequisites:**
    *   Python >= 3.12 (as specified in `pyproject.toml`)
    *   A tool like `pip` or `uv` for package management.

2.  **Environment:** It is highly recommended to use a virtual environment:
    ```bash
    # Using venv
    python -m venv .venv
    source .venv/bin/activate # On Windows: .venv\Scripts\activate
    
    # Or using uv
    uv venv
    source .venv/bin/activate # On Windows: .venv\Scripts\activate
    ```

3.  **Installation:** Install the project and its dependencies:
    ```bash
    # Using uv (recommended)
    uv sync
    
    # Or using pip
    pip install .
    ```

    **Note on Memory:** The default embedding model (`sentence-transformers/LaBSE`) requires a significant amount of RAM and virtual memory (page file). If you encounter `OSError: [WinError 1455] The paging file is too small for this operation to complete` during startup (check `embedding_server.log`), you may need to:
    *   Increase your Windows paging file size.
    *   Switch to a smaller model (see Configuration below).

## Running the Server

The server communicates via standard input/output (stdio) as expected by MCP clients.

You can run the server using `uv`:

*   **Directly (Recommended to bypass potential build issues):**
    ```bash
    uv run python src/embedding/server.py
    ```

*   **Using the installed script (defined in `pyproject.toml`):**
    ```bash
    uv run embedding-server
    ```

*   **With the MCP Inspector:**
    ```bash
    npx @modelcontextprotocol/inspector uv run python src/embedding/server.py
    ```

## Functionality: `get-relevant-tools` Tool

*   **Purpose:** Ranks provided tools based on semantic similarity to the user's context.
*   **Input Arguments (`arguments` dictionary):**
    *   `prompt` (string): The latest user query.
    *   `history` (list): A list of previous conversation messages (usually dictionaries with 'role' and 'content').
    *   `tools` (list): A list of available tool definition objects. Each object must contain at least a `name` (string) and `description` (string).
    *   `max_tools` (integer, optional, default: 5): The maximum number of relevant tool names to return.
*   **Output:** A `TextContent` object containing a JSON string representation of a list of the top `max_tools` relevant tool names (e.g., `"["tool1", "tool2"]"`).

## Logging

Server activity and errors are logged to `embedding_server.log` in the project root directory.

## Configuration

*   **Embedding Model:** To change the Sentence Transformer model, modify the model name string in the `SentenceTransformer(...)` call within `src/embedding/server.py`. Consider smaller models like `all-MiniLM-L6-v2` (English) or `paraphrase-multilingual-MiniLM-L12-v2` if you encounter memory issues.
*   **Logging:** Logging level and file rotation settings can be adjusted in the "Logging Setup" section of `src/embedding/server.py`.
