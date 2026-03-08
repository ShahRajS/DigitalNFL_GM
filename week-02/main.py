import asyncio
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.genai import types

# 1. Load your .env from the root directory
load_dotenv()

# ======================================================================
# WEEK 2 NEW FEATURE: TOOLS (GIVING YOUR GM "HANDS")
# ======================================================================

import nfl_data_py as nfl
import json
import ssl

def get_player_stats(player_name: str, year: int = 2023) -> str:
    """Gets the stats for a given NFL player using nfl_data_py.

    Args:
        player_name: The first and last name of the NFL player (e.g., 'Christian McCaffrey').
        year: The NFL season year to retrieve stats for (defaults to 2023).
        
    Returns:
        Summary of the player's statistics for the requested year.
    """
    # Bypass SSL verification issues depending on the local machine's python certs
    ssl._create_default_https_context = ssl._create_unverified_context
    
    try:
        # Import the requested season
        df = nfl.import_weekly_data([year])
        
        # Filter down to the specific player
        player_df = df[df['player_display_name'].str.lower() == player_name.lower()]
        
        if player_df.empty:
            return f"Error: Could not find any {year} stats for {player_name} in nfl_data_py."
        
        # Aggregate their season stats
        stats = player_df[['passing_yards', 'passing_tds', 'rushing_yards', 'rushing_tds', 'receptions', 'receiving_yards', 'receiving_tds']].sum()
        
        # Format as a nice dictionary for the LLM to read
        result = {
            "player": player_name,
            "season": year,
            "stats": stats.to_dict()
        }
        
        return json.dumps(result)
        
    except Exception as e:
        return f"Error retrieving data for {player_name} in the year {year} using nfl_data_py: {e}"

# ======================================================================

# 2. Define YourGM with Tools capability
retry_config=types.HttpRetryOptions(
    attempts=5, 
    exp_base=7, 
    initial_delay=1, 
    http_status_codes=[429, 500, 503, 504] 
)

root_agent = Agent(
    name="NFL_GM",
    model=Gemini(
        model="gemini-2.5-flash",
        retry_options=retry_config
    ),
    description="A simple agent that can answer general NFL based questions and lookup stats.",
    # === NEW: Notice how we pass the tool down to the Agent ===
    tools=[get_player_stats],
    # === NEW: Updated instructions to enforce tool usage ===
    instruction="""You are 'YourGM,' an elite NFL scout and Fantasy Football expert.
Your goal is to provide data-driven insights on player performance and draft strategy.
Be concise, professional, and use scouting terminology. Examples such as: ('ADP', 'Target Share', 'Red Zone Efficiency').

CRITICAL: If a user asks about a player's stats or performance, you MUST use the `get_player_stats` tool to find their statistics before answering. Do not guess or hallucinate stats.
""",
)

def main():
    # 3. Initialize the Runner
    runner = InMemoryRunner(agent=root_agent)
    
    # Session Initialization for google-adk <= 1.8.0
    USER_ID = "user_123"
    SESSION_ID = "session_123"
    asyncio.run(runner.session_service.create_session(
        app_name=runner.app_name,
        user_id=USER_ID,
        session_id=SESSION_ID
    ))

    print("--- Digital NFL GM: System Online (Week 2 Edition) ---")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "q", "x", "bye", "goodbye"]:
            break
        
        # 4. Query the Agent
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
            if event.content and event.content.parts:
                for part in event.content.parts:
                    # === NEW: Check if the Agent is using a tool ===
                    if part.function_call:
                        print(f"\n[System: YourGM is calling tool '{part.function_call.name}'...]\nYourGM: ", end="", flush=True)
                    if part.text:
                        print(part.text, end="", flush=True)
        print("\n")

if __name__ == "__main__":
    main()
