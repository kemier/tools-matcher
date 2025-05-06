import logging.handlers # Import handlers
import numpy as np
from sentence_transformers import SentenceTransformer
# Use user-provided correct imports
from mcp.server.models import InitializationOptions
import mcp.types as types
from mcp.server import NotificationOptions, Server
import mcp.server.stdio as stdio
import asyncio
import json # Added json import

# --- Logging Setup ---
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log_file = 'embedding_server.log' # Log filename

# Create file handler
file_handler = logging.handlers.RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5) # Example: 10MB per file, 5 backups
file_handler.setFormatter(log_formatter)

# Get the logger and add the handler
# Configure the root logger or a specific logger
logger = logging.getLogger() # Get root logger to capture logs from dependencies too (like sentence_transformers)
logger.setLevel(logging.INFO) # Set desired level (INFO, DEBUG, etc.)
logger.addHandler(file_handler)

# Optional: Add a StreamHandler back if you want logs on console *and* file
# stream_handler = logging.StreamHandler()
# stream_handler.setFormatter(log_formatter)
# logger.addHandler(stream_handler)

# --- MCP Server Instance ---
server = Server("embedding-server") # Changed server name

# --- Tool Definition Handler ---
@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """
    List the single tool provided by this server.
    """
    # Define the schema for the get-relevant-tools input
    input_schema = {
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "description": "The latest user query."},
            "history": {
                "type": "array", 
                "items": {"type": "object"}, # Assuming ChatMessage structure
                "description": "The conversation history."
            },
            "tools": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "description": {"type": "string"}
                    },
                    "required": ["name", "description"]
                },
                "description": "List of available tools with names and descriptions."
            },
            "max_tools": {"type": "integer", "default": 5, "description": "Max number of tools to return."}
        },
        "required": ["prompt", "history", "tools"]
    }
    
    return [
        types.Tool(
            name="get-relevant-tools",
            description="Calculates relevance of tools based on prompt and history using embeddings.",
            inputSchema=input_schema # Use the defined schema
        ),
    ]

# --- Global Model Loading ---
try:
    logger.info("Loading SentenceTransformer model...")
    model = SentenceTransformer('sentence-transformers/LaBSE')
    logger.info("SentenceTransformer model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load SentenceTransformer model: {e}", exc_info=True)
    model = None

# --- Helper Functions ---
def _combine_context(prompt: str, history: list) -> str:
    history_text = "\n".join([msg.get('content', '') for msg in history[-3:]])
    return f"{history_text}\n\nUser: {prompt}"

def _cosine_similarity(vec_a, vec_b):
    if vec_a is None or vec_b is None: return 0.0
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0: return 0.0
    vec_a = vec_a.flatten()
    vec_b = vec_b.flatten()
    return np.dot(vec_a, vec_b) / (norm_a * norm_b)

# --- Tool Call Handler ---
@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict | None
) -> list[types.TextContent]: # Return list of TextContent
    """
    Handle execution requests for the get-relevant-tools tool.
    """
    global model
    if name != "get-relevant-tools":
        raise ValueError(f"Unknown tool: {name}")

    if not model:
        logger.error("Model not loaded, cannot process tool call.")
        return [types.TextContent(type="text", text="Error: Embedding model is not loaded.")]

    if not arguments:
        # Use default values or raise error if arguments are essential
        logger.warning("Missing arguments for get-relevant-tools.")
        return [types.TextContent(type="text", text="Error: Missing required arguments.")]

    # Extract arguments for get-relevant-tools
    prompt = arguments.get('prompt')
    history = arguments.get('history', [])
    tools_info = arguments.get('tools', [])
    max_tools = arguments.get('max_tools', 5)

    if not prompt or not tools_info:
        logger.warning("Missing prompt or tools info in arguments.")
        return [types.TextContent(type="text", text="Error: Missing 'prompt' or 'tools' in arguments.")]

    try:
        # --- Core Embedding Logic --- 
        context_text = _combine_context(prompt, history)
        context_embedding = model.encode(context_text)
        
        tool_descriptions = [t.get('description', t.get('name', '')) for t in tools_info]
        tool_names = [t.get('name', '') for t in tools_info]
        
        if not tool_descriptions:
            logger.warning("No tool descriptions provided.")
            return [types.TextContent(type="text", text="[]")] # Return empty list as text

        tool_embeddings = model.encode(tool_descriptions)
        similarities = [_cosine_similarity(context_embedding, tool_emb) for tool_emb in tool_embeddings]
        sorted_indices = np.argsort(similarities)[::-1]
        ranked_tool_names = [tool_names[i] for i in sorted_indices if i < len(tool_names)]
        final_tools = ranked_tool_names[:max_tools]
        
        logger.info(f"Calculated relevant tools: {final_tools}")
        # Return the result as a VALID JSON string representation of the list
        return [types.TextContent(type="text", text=json.dumps(final_tools))] # Use json.dumps

    except Exception as e:
        logger.error(f"Error processing get-relevant-tools: {e}", exc_info=True)
        # Return error message (or potentially a JSON error object?)
        return [types.TextContent(type="text", text=f"Error: Internal server error during tool execution.")]

# Removed weather helper functions and constants
# Removed standalone handle_get_relevant_tools function

# --- Main Execution ---
async def main_async(): # Renamed from main
    global server

    if not model:
        logger.critical("Model failed to load. Server cannot start.")
        return

    logger.info("Starting MCP server...")
    async with stdio.stdio_server() as (read_stream, write_stream):
        try:
            # Get capabilities using the decorators now
            # Add experimental_capabilities back as it seems required
            server_caps = server.get_capabilities(
                notification_options=NotificationOptions(),
                experimental_capabilities={} # Added back as required
            )

            init_options = InitializationOptions(
                server_name="embedding-server",
                server_version="0.1.0",
                capabilities=server_caps
            )
            
            logger.info("Prepared init_options. Calling server.run...")
            await server.run(
                read_stream,
                write_stream,
                init_options
            )
            # If server.run finishes without error, this will be logged
            logger.info("server.run completed.") 
        except Exception as e:
            # Log the specific error from server.run if it raises one
            logger.error(f"Server run failed: {e}", exc_info=True)
            
    logger.info("MCP server finished.")

def main(): # New synchronous wrapper
    """Synchronous entry point for the server."""
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        logger.info("Server shutdown requested via KeyboardInterrupt.")
    except Exception as e:
        logger.critical(f"Unhandled exception in main: {e}", exc_info=True)

if __name__ == "__main__":
    main() # Call the synchronous wrapper