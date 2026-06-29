import asyncio
import os
import sys
import threading
import time
from dotenv import load_dotenv

# Ensure the week-09 directory is in sys.path to handle all execution context path differences
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

# Load env from root
dotenv_path = os.path.join(os.path.dirname(script_dir), '.env')
load_dotenv(dotenv_path)

# Map key if GEMINI_API_KEY is not set but GOOGLE_API_KEY is
if "GOOGLE_API_KEY" in os.environ and "GEMINI_API_KEY" not in os.environ:
    os.environ["GEMINI_API_KEY"] = os.environ["GOOGLE_API_KEY"]

from google.adk.agents import Agent
from google.adk.tools.tool_context import ToolContext
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.genai import types

# Import RAG functions
from rag_utils import retrieve_chunks, add_scout_note, load_vector_db

# ======================================================================
# MCP POSTGRES INTEGRATION
# ======================================================================
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession

_mcp_session = None
_mcp_loop = None
_mcp_connected = threading.Event()

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
                    _mcp_connected.set()
                    while True:
                        await asyncio.sleep(1)
        except Exception as e:
            print(f"MCP Connection Error: {e}")
                    
    def t_func():
        asyncio.set_event_loop(_mcp_loop)
        _mcp_loop.run_until_complete(run_mcp())
        
    t = threading.Thread(target=t_func, daemon=True)
    t.start()

# ======================================================================
# MCP BRAVE SEARCH INTEGRATION
# ======================================================================
_mcp_brave_session = None
_mcp_brave_loop = None
_mcp_brave_connected = threading.Event()

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
                    _mcp_brave_connected.set()
                    while True:
                        await asyncio.sleep(1)
        except Exception as e:
            print(f"Brave MCP Error: {e}")
            
    def t_func_brave():
        asyncio.set_event_loop(_mcp_brave_loop)
        _mcp_brave_loop.run_until_complete(run_mcp_brave())
        
    t_brave = threading.Thread(target=t_func_brave, daemon=True)
    t_brave.start()

# ======================================================================
# TOOL WRAPPERS FOR DATABASE & WEB
# ======================================================================
def execute_db_query(query: str) -> str:
    """Executes a PostgreSQL query against the team's Supabase DB.
    
    Args:
        query: The raw SQL string to execute. (e.g. "SELECT tablename FROM pg_catalog.pg_tables")
    Returns:
        The results of the query or an error string.
    """
    if not _mcp_connected.is_set():
        if not _mcp_connected.wait(timeout=3.0):
            return "Error: MCP Database session is not connected yet. Please try again in a few seconds."
    
    future = asyncio.run_coroutine_threadsafe(
        _mcp_session.call_tool("query", arguments={"sql": query}), 
        _mcp_loop
    )
    try:
        res = future.result(timeout=15)
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
    if not _mcp_brave_connected.is_set():
        if not _mcp_brave_connected.wait(timeout=3.0):
            return "Error: Brave MCP session is not connected. Missing API key or connection delayed."
    
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
# LONG-TERM STATE (Session Memory)
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
# WEEK 9 NEW FEATURE: MM-RAG UTILITY TOOLS
# ======================================================================
def query_draft_packet(query: str, source_filter: str = None) -> str:
    """Queries the vector database for 2025/2026 San Francisco 49ers Draft Packets, prospect profiles, scouting notes, or season reviews.
    
    Args:
        query: The specific search question or keywords (e.g. '2026 draft top prospects' or 'draft trade rules').
        source_filter: Optional source PDF filename to filter the search (e.g., '2025-San-Francisco-49ers-Draft-Packet.pdf', '2026-San-Francisco-49ers-Draft-Packet.pdf', or 'San-Francisco-49ers-2025-Season-Review.pdf').
    Returns:
        The matching text excerpts and confidence scores from the database.
    """
    try:
        results = retrieve_chunks(query, top_k=3, source_filter=source_filter)
        if not results:
            return "No matching records found in the database."
            
        formatted = "Here are the most relevant excerpts found in the vector database:\n"
        for idx, r in enumerate(results):
            formatted += f"\n[Excerpt {idx+1}] Source: {r['metadata']['source']} (Page {r['metadata'].get('page', 'N/A')}) | Confidence: {r['score']:.4f}\n"
            formatted += f"Content: {r['text']}\n"
            formatted += "-" * 60 + "\n"
        return formatted
    except Exception as e:
        return f"Error querying vector database: {e}"

def add_note_to_draft_database(note_text: str, source: str = "Scout Note") -> str:
    """Saves a custom text note or scouting report to the vector database for future RAG searches.
    
    Args:
        note_text: The info snippet or note you want to save.
        source: The name of the source or category for this note.
    Returns:
        Confirmation message.
    """
    try:
        msg = add_scout_note(note_text, source)
        return msg
    except Exception as e:
        return f"Error adding note to draft database: {e}"

# ======================================================================
# MODEL AND AGENT DEFINITION
# ======================================================================
retry_config = types.HttpRetryOptions(
    attempts=5, 
    exp_base=7, 
    initial_delay=1, 
    http_status_codes=[429, 500, 503, 504] 
)

model = Gemini(
    model="gemini-2.5-flash",
    retry_options=retry_config
)

draft_scout = Agent(
    name="Draft_Scout",
    model=model,
    description="A professional scout that queries database stats, browses the web, manages the draft board session state, and utilizes a multimodal RAG pipeline for the draft packet.",
    tools=[
        add_player_to_draft_board, 
        remove_player_from_draft_board, 
        view_draft_board, 
        execute_db_query, 
        execute_web_search,
        query_draft_packet,
        add_note_to_draft_database
    ],
    instruction="""You are 'YourGM,' an elite NFL Scout managing the team's Draft Board.
Your goal is to maintain the Draft Room's memory, use the database when requested, and provide scouting intelligence and draft facts.

Guidelines for querying the vector database:
1. When asked about draft dates, rules/limits, prospect notes, player scouts, team reviews, or stats from the 2025 or 2026 Draft Packets, or the 2025 Season Review, you MUST search the vector database using the `query_draft_packet` tool first.
2. Do NOT use `execute_web_search` for queries regarding draft prospects, rankings, dates, or rules if they can be answered by querying the draft packet database. Always search the vector database first!
3. Cite the source document, page number, and confidence score of the excerpts you retrieve in your final answer.
4. If the user shares new facts, scout insights, or updates about players (e.g. Pro Day results) that you should remember for future retrieval, you MUST use the `add_note_to_draft_database` tool to save them.
5. If the user specifies which document they want (e.g. 2025 Draft Packet, 2026 Draft Packet, or 2025 Season Review), you MUST set the `source_filter` parameter of the `query_draft_packet` tool to the corresponding filename: '2025-San-Francisco-49ers-Draft-Packet.pdf', '2026-San-Francisco-49ers-Draft-Packet.pdf', or 'San-Francisco-49ers-2025-Season-Review.pdf'. Otherwise, leave it as null.

Other guidelines:
- Use the `execute_web_search` tool ONLY when you need real-time, current-day breaking news (e.g., today's injuries, trade rumors, signings) or information explicitly NOT covered by the team's draft packets or season reviews.
- When you are asked to check for historical stats or players in our local records, use the `execute_db_query` tool to search standard PostgreSQL tables.
- IMPORTANT: You don't know what tables exist inside the database! You MUST query the database first to find out what tables/schemas exist (e.g., SELECT * FROM information_schema.tables WHERE table_schema='public'). 
- When a user asks you to add someone to the draft board, use the `add_player_to_draft_board` tool.
- When a user asks you to remove someone, use `remove_player_from_draft_board`.
- When the user asks who is on the board, you MUST call `view_draft_board` rather than trying to guess from chat history.
Always be professional and confirm actions clearly."""
)

def main():
    print("Initializing RAG Vector Database cache...")
    try:
        # Pre-load or index on startup so database is ready
        load_vector_db()
    except Exception as e:
        print(f"Warning/Error loading vector database: {e}")
        
    print("Connecting to Supabase Database & Brave Search MCP...")
    start_mcp_background()
    start_mcp_brave_background()
    time.sleep(2) # Give node time to start
    
    # Initialize the Runner
    runner = InMemoryRunner(agent=draft_scout)
    
    # Session Initialization
    USER_ID = "user_123"
    SESSION_ID = "session_123"
    asyncio.run(runner.session_service.create_session(
        app_name=runner.app_name,
        user_id=USER_ID,
        session_id=SESSION_ID
    ))

    print("--- Digital NFL GM: System Online (Week 9 Edition - MM-RAG Model) ---")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "q", "x", "bye", "goodbye"]:
            break
        
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
