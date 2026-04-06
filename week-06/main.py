import asyncio
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.tools.tool_context import ToolContext
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.genai import types

import os
# 1. Load your .env from the root directory
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
load_dotenv(dotenv_path)

# ======================================================================
# WEEK 6 NEW FEATURE: MCP POSTGRES INTEGRATION
# ======================================================================
import threading
import time
import os
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession

_mcp_session = None
_mcp_loop = None

def start_mcp_background():
    global _mcp_session, _mcp_loop
    _mcp_loop = asyncio.new_event_loop()
    
    async def run_mcp():
        global _mcp_session
        server_params = StdioServerParameters(
            command="npx",
            args=[
                "-y", 
                "@modelcontextprotocol/server-postgres", 
                os.getenv("DATABASE_URL", "")
            ],
        )
        try:
            async with stdio_client(server_params) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    _mcp_session = session
                    # Keep thread alive to serve the session pipe
                    while True:
                        await asyncio.sleep(1)
        except Exception as e:
            print(f"MCP Connection Error: {e}")
                    
    def t_func():
        asyncio.set_event_loop(_mcp_loop)
        _mcp_loop.run_until_complete(run_mcp())
        
    t = threading.Thread(target=t_func, daemon=True)
    t.start()

_mcp_brave_session = None
_mcp_brave_loop = None

def start_mcp_brave_background():
    global _mcp_brave_session, _mcp_brave_loop
    _mcp_brave_loop = asyncio.new_event_loop()
    
    async def run_mcp_brave():
        global _mcp_brave_session
        brave_key = os.getenv("BRAVE_API_KEY", "")
        if not brave_key:
            print("WARNING: BRAVE_API_KEY is not set. Web search will fail.")
            
        server_params = StdioServerParameters(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-brave-search"],
            env={**os.environ, "BRAVE_API_KEY": brave_key}
        )
        try:
            async with stdio_client(server_params) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    _mcp_brave_session = session
                    while True:
                        await asyncio.sleep(1)
        except Exception as e:
            print(f"Brave MCP Error: {e}")
            
    def t_func_brave():
        asyncio.set_event_loop(_mcp_brave_loop)
        _mcp_brave_loop.run_until_complete(run_mcp_brave())
        
    t_brave = threading.Thread(target=t_func_brave, daemon=True)
    t_brave.start()

def execute_db_query(query: str) -> str:
    """Executes a PostgreSQL query against the team's Supabase DB.
    
    Args:
        query: The raw SQL string to execute. (e.g. "SELECT tablename FROM pg_catalog.pg_tables")
    Returns:
        The results of the query or an error string.
    """
    if not _mcp_session:
        return "Error: MCP session is not connected yet. Tell the user to wait a moment."
    
    future = asyncio.run_coroutine_threadsafe(
        _mcp_session.call_tool("query", arguments={"sql": query}), 
        _mcp_loop
    )
    try:
        res = future.result(timeout=15)
        
        # Depending on the server response structure, extract text safely
        if res.isError:
            return f"Query returned an error from server."
            
        output = ""
        for content in res.content:
            if content.type == "text":
                output += content.text
        return output
    except Exception as e:
        return f"Query failed: {e}"

def execute_web_search(query: str) -> str:
    """Executes a web search via the Brave Search MCP to find the latest real-time NFL news.
    
    Args:
        query: The search query string (e.g. "Justin Jefferson latest injury update 2025")
    Returns:
        The text results of the web search.
    """
    if not _mcp_brave_session:
        return "Error: Brave MCP session is not connected. Missing API key?"
    
    future = asyncio.run_coroutine_threadsafe(
        _mcp_brave_session.call_tool("brave_web_search", arguments={"query": query}), 
        _mcp_brave_loop
    )
    try:
        res = future.result(timeout=15)
        if res.isError:
            return f"Brave search returned an error from server."
            
        output = ""
        for content in res.content:
            if content.type == "text":
                output += content.text
        return output
    except Exception as e:
        return f"Web search failed: {e}"

# ======================================================================
# WEEK 5 NEW FEATURE: LONG-TERM STATE (Session Memory)
# ======================================================================

def add_player_to_draft_board(tool_context: ToolContext, player_name: str, position: str) -> str:
    """Adds a player to the Draft Board session state.

    Args:
        tool_context: The ADK ToolContext (Injected automatically, do not pass manually).
        player_name: The name of the player to add.
        position: The position of the player (e.g., 'QB', 'WR').
        
    Returns:
        Confirmation message that the player was added.
    """
    if "draft_board" not in tool_context.state:
        tool_context.state["draft_board"] = []
    
    draft_board = tool_context.state["draft_board"]
    
    # Check if already added
    for p in draft_board:
        if p["name"].lower() == player_name.lower():
            return f"{player_name} is already on the Draft Board!"
            
    draft_board.append({"name": player_name, "position": position})
    tool_context.state["draft_board"] = draft_board
    return f"Successfully added {player_name} ({position}) to the Draft Board."

def remove_player_from_draft_board(tool_context: ToolContext, player_name: str) -> str:
    """Removes a player from the Draft Board session state.

    Args:
        tool_context: The ADK ToolContext (Injected automatically, do not pass manually).
        player_name: The name of the player to remove.
        
    Returns:
        Confirmation message.
    """
    draft_board = tool_context.state.get("draft_board", [])
    
    for i, p in enumerate(draft_board):
        if p["name"].lower() == player_name.lower():
            draft_board.pop(i)
            # Re-assign back to state
            tool_context.state["draft_board"] = draft_board
            return f"{player_name} was removed from the Draft Board."
            
    return f"Could not find {player_name} on the Draft Board."

def view_draft_board(tool_context: ToolContext) -> str:
    """Retrieves the full list of players currently on the Draft Board.

    Args:
        tool_context: The ADK ToolContext (Injected automatically, do not pass manually).
        
    Returns:
        A list of drafted players or a message if empty.
    """
    draft_board = tool_context.state.get("draft_board", [])
    if not draft_board:
        return "The Draft Board is currently empty."
    
    formatted_board = "\n".join([f"- {p['name']} ({p['position']})" for p in draft_board])
    return f"Current Draft Board:\n{formatted_board}"

# ======================================================================

retry_config=types.HttpRetryOptions(
    attempts=5, 
    exp_base=7, 
    initial_delay=1, 
    http_status_codes=[429, 500, 503, 504] 
)

model = Gemini(
    model="gemini-2.5-flash",
    retry_options=retry_config
)

# 2. Define the Agent with Memory Tools
draft_scout = Agent(
    name="Draft_Scout",
    model=model,
    description="A scout that manages the persistent draft room board and queries the database.",
    tools=[add_player_to_draft_board, remove_player_from_draft_board, view_draft_board, execute_db_query, execute_web_search],
    instruction="""You are 'YourGM,' an elite NFL Scout managing the team's Draft Board.
Your goal is to maintain the Draft Room's memory and use the database when requested.
- When you need breaking NFL news, draft rankings, or real-time injury updates, MUST use the `execute_web_search` tool.
- When you are asked to check for stats or players, use the `execute_db_query` tool. Use standard PostgreSQL queries.
- IMPORTANT: You don't know what tables exist inside the database! You MUST query the database first to find out what tables/schemas exist (e.g., SELECT * FROM information_schema.tables WHERE table_schema='public'). 
- When a user asks you to add someone to the draft board, use the `add_player_to_draft_board` tool.
- When a user asks you to remove someone, use `remove_player_from_draft_board`.
- When the user asks who is on the board, MUST call `view_draft_board` rather than trying to guess from chat history.
Always be professional and confirm their actions."""
)

def main():
    # Start Cloud Postgres & Brave Search connection
    print("Connecting to Supabase Database & Brave Search MCP...")
    start_mcp_background()
    start_mcp_brave_background()
    time.sleep(2) # Give node time to start
    
    # 3. Initialize the Runner
    runner = InMemoryRunner(agent=draft_scout)
    
    # Session Initialization for google-adk <= 1.8.0
    USER_ID = "user_123"
    SESSION_ID = "session_123"
    asyncio.run(runner.session_service.create_session(
        app_name=runner.app_name,
        user_id=USER_ID,
        session_id=SESSION_ID
    ))

    print("--- Digital NFL GM: System Online (Week 5 Edition - Draft Room Memory) ---")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "q", "x", "bye", "goodbye"]:
            break
        
        # 4. Query the Pipeline
        events = runner.run(
            user_id=USER_ID,
            session_id=SESSION_ID,
            new_message=types.Content(
                parts=[types.Part.from_text(text=user_input)],
                role="user"
            )
        )
        
        print("\nYourGM: ", end="", flush=True)
        for event in events:
            agent_label = getattr(event, 'agent_id', getattr(event, 'source', 'Draft_Scout'))
            if getattr(event, 'content', None) and getattr(event.content, 'parts', None):
                for part in event.content.parts:
                    if getattr(part, 'function_call', None):
                        print(f"\n[System: {agent_label} is calling tool '{part.function_call.name}'...]", end="", flush=True)
                    if getattr(part, 'text', None):
                        print(f"\n[{agent_label}]: {part.text}", end="", flush=True)
        print("\n")

if __name__ == "__main__":
    main()
