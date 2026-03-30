import asyncio
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.tools.tool_context import ToolContext
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.genai import types

# 1. Load your .env from the root directory
load_dotenv()

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
    description="A scout that manages the persistent draft room board.",
    tools=[add_player_to_draft_board, remove_player_from_draft_board, view_draft_board],
    instruction="""You are 'YourGM,' an elite NFL Scout managing the team's Draft Board.
Your goal is to maintain the Draft Room's memory.
- When a user asks you to add someone to the draft board, use the `add_player_to_draft_board` tool.
- When a user asks you to remove someone, use `remove_player_from_draft_board`.
- When the user asks who is on the board, MUST call `view_draft_board` rather than trying to guess from chat history.
Always be professional and confirm their actions."""
)

def main():
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
