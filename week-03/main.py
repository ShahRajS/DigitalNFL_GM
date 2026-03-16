import asyncio
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.genai import types, Client

# 1. Load your .env from the root directory
load_dotenv()

# ======================================================================
# WEEK 3 NEW FEATURE: AGENTIC TOOLS (Unlocking your GM Potential)
# ======================================================================

import nflreadpy as nfl
import json
import ssl

def get_player_stats(query: str, player_name: str, year: int = 2025) -> str:
    """Gets the stats for a given NFL player using nflreadpy.

    Args:
        query: The original question the user asked.
        player_name: The first and last name of the NFL player (e.g., 'Christian McCaffrey').
        year: The NFL season year to retrieve stats for (defaults to 2025).
        
    Returns:
        Summary of the player's statistics for the requested year.
    """
    # Bypass SSL verification issues depending on the local machine's python certs
    ssl._create_default_https_context = ssl._create_unverified_context
    
    try:
        # Import the requested season
        df = nfl.load_player_stats([year])
        
        # Filter down to the specific player using Polars syntax
        player_df = df.filter(df['player_display_name'].str.to_lowercase() == player_name.lower())
        
        if player_df.height == 0:
            return f"Error: Could not find any {year} stats for {player_name} in nflreadpy."
        
        # Aggregate their season stats
        stats = player_df.select(['passing_yards', 'passing_tds', 'rushing_yards', 'rushing_tds', 'receptions', 'receiving_yards', 'receiving_tds']).sum()
        
        # Format as a nice dictionary for the LLM to read
        result = {
            "player": player_name,
            "season": year,
            "stats": stats.to_dicts()[0]
        }
        
        # === NEW: Use an LLM to process the raw data and answer the query ===
        client = Client()
        prompt = f"""
        You are a seasoned NFL scout. You have been given raw JSON statistics for a player.
        Answer the following user query based ONLY on the provided data.
        
        User Query: {query}
        
        Raw Data: {json.dumps(result)}
        """
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        return response.text
        
    except Exception as e:
        return f"Error retrieving data for {player_name} in the year {year} using nflreadpy: {e}"

def get_upcoming_schedule(query: str, team_abbr: str, year: int = 2025) -> str:
    """Gets the schedule for a given NFL team using nflreadpy.

    Args:
        query: The original question the user asked.
        team_abbr: The 2-3 letter abbreviation of the NFL team (e.g., 'SF', 'KC').
        year: The NFL season year (defaults to 2025).
        
    Returns:
        Summary of the team's schedule for the requested year, including opponents and results.
    """
    ssl._create_default_https_context = ssl._create_unverified_context
    
    try:
        df = nfl.load_schedules([year])
        
        # Filter where team is either home or away
        team_games = df.filter((df['home_team'] == team_abbr.upper()) | (df['away_team'] == team_abbr.upper()))
        
        if team_games.height == 0:
            return f"Error: Could not find any {year} schedule for team {team_abbr}."
            
        # Select relevant columns
        schedule = team_games.select(['week', 'game_type', 'away_team', 'home_team', 'away_score', 'home_score'])
        
        result = {
            "team": team_abbr.upper(),
            "season": year,
            "schedule": schedule.to_dicts()
        }
        
        # === NEW: Use an LLM to process the raw data and answer the query ===
        client = Client()
        prompt = f"""
        You are an NFL scheduling expert. You have been given raw JSON schedule data for a team.
        Answer the following user query based ONLY on the provided data.
        
        User Query: {query}
        
        Raw Data: {json.dumps(result)}
        """
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        return response.text
        
    except Exception as e:
        return f"Error retrieving schedule for {team_abbr} in {year}: {e}"

def get_depth_chart(query: str, team_abbr: str, position: str, year: int = 2025) -> str:
    """Gets the depth chart for a specific position on a specific NFL team using nflreadpy.

    Args:
        query: The original question the user asked.
        team_abbr: The abbreviation of the NFL team (e.g., 'SF', 'KC').
        position: The position to check (e.g., 'QB', 'RB', 'WR', 'TE').
        year: The NFL season year (defaults to 2025).
        
    Returns:
        The depth chart for that position.
    """
    ssl._create_default_https_context = ssl._create_unverified_context
    
    try:
        df = nfl.load_depth_charts([year])
        
        # nflreadpy uses an ESPN schema for the current season, and an Ourlads schema for historical
        if 'club_code' in df.columns:
            # Historical schema (<= 2024)
            team_dc = df.filter((df['club_code'] == team_abbr.upper()) & (df['position'] == position.upper()) & (df['week'] == 1))
            if team_dc.height == 0:
                return f"Error: Could not find depth chart for {team_abbr} {position} in {year}."
            dc = team_dc.select(['full_name', 'depth_team']).sort('depth_team')
        else:
            # Current schema (>= 2025)
            team_dc = df.filter((df['team'] == team_abbr.upper()) & (df['pos_abb'] == position.upper()))
            if team_dc.height == 0:
                return f"Error: Could not find depth chart for {team_abbr} {position} in {year}."
            dc = team_dc.select(['player_name', 'pos_rank']).rename({'player_name': 'full_name', 'pos_rank': 'depth_team'}).sort('depth_team')
        
        result = {
            "team": team_abbr.upper(),
            "position": position.upper(),
            "season": year,
            "depth_chart": dc.to_dicts()
        }
        
        # === NEW: Use an LLM to process the raw data and answer the query ===
        client = Client()
        prompt = f"""
        You are a professional NFL personnel scout. You have been given raw JSON depth chart data for a team positional group.
        Answer the following user query based ONLY on the provided data. Give concise answers.
        
        User Query: {query}
        
        Raw Data: {json.dumps(result)}
        """
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        return response.text
        
    except Exception as e:
        return f"Error retrieving depth chart for {team_abbr} {position} in {year}: {e}"

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
    tools=[get_player_stats, get_upcoming_schedule, get_depth_chart],
    # === NEW: Updated instructions to enforce tool usage ===
    instruction="""You are 'YourGM,' an elite NFL scout and Fantasy Football expert.
Your goal is to provide data-driven insights on player performance and draft strategy.
Be concise, professional, and use scouting terminology. Examples such as: ('ADP', 'Target Share', 'Red Zone Efficiency').

CRITICAL: 
- If a user asks about a player's stats or performance, you MUST use the `get_player_stats` tool.
- If a user asks about a team's schedule or upcoming games, you MUST use the `get_upcoming_schedule` tool.
- If a user asks about a team's depth chart, roster spot, or who the starter/backup is, you MUST use the `get_depth_chart` tool.
Do not guess or hallucinate data. If the tools don't return data, just say that the data is UNAVAILABLE.
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

    print("--- Digital NFL GM: System Online (Week 3 Edition) ---")
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
