import asyncio
from dotenv import load_dotenv
from google.adk.agents import Agent, ParallelAgent, SequentialAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.genai import types, Client

import nflreadpy as nfl
import json
import ssl

# 1. Load your .env from the root directory
load_dotenv()

# ======================================================================
# WEEK 4 NEW FEATURE: SEQUENTIAL AND PARALLEL AGENTS
# ======================================================================

def get_player_stats(query: str, player_name: str, year: int = 2024) -> str:
    """Gets the stats for a given NFL player using nflreadpy.

    Args:
        query: The original question the user asked.
        player_name: The first and last name of the NFL player (e.g., 'Christian McCaffrey').
        year: The NFL season year to retrieve stats for (defaults to 2024).
        
    Returns:
        Summary of the player's statistics for the requested year.
    """
    ssl._create_default_https_context = ssl._create_unverified_context
    try:
        df = nfl.load_player_stats([year])
        player_df = df.filter(df['player_display_name'].str.to_lowercase() == player_name.lower())
        if player_df.height == 0:
            return f"Error: Could not find any {year} stats for {player_name} in nflreadpy."
        stats = player_df.select(['passing_yards', 'passing_tds', 'rushing_yards', 'rushing_tds', 'receptions', 'receiving_yards', 'receiving_tds']).sum()
        result = {
            "player": player_name,
            "season": year,
            "stats": stats.to_dicts()[0]
        }
        client = Client()
        prompt = f"You are a seasoned NFL scout. You have been given raw JSON statistics for a player.\nAnswer the following user query based ONLY on the provided data.\n\nUser Query: {query}\n\nRaw Data: {json.dumps(result)}"
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        return response.text
    except Exception as e:
        return f"Error retrieving data for {player_name} in the year {year} using nflreadpy: {e}"

def get_upcoming_schedule(query: str, team_abbr: str, year: int = 2024) -> str:
    """Gets the schedule for a given NFL team using nflreadpy.

    Args:
        query: The original question the user asked.
        team_abbr: The 2-3 letter abbreviation of the NFL team (e.g., 'SF', 'KC').
        year: The NFL season year (defaults to 2024).
        
    Returns:
        Summary of the team's schedule for the requested year, including opponents and results.
    """
    ssl._create_default_https_context = ssl._create_unverified_context
    try:
        df = nfl.load_schedules([year])
        team_games = df.filter((df['home_team'] == team_abbr.upper()) | (df['away_team'] == team_abbr.upper()))
        if team_games.height == 0:
            return f"Error: Could not find any {year} schedule for team {team_abbr}."
        schedule = team_games.select(['week', 'game_type', 'away_team', 'home_team', 'away_score', 'home_score'])
        result = {
            "team": team_abbr.upper(),
            "season": year,
            "schedule": schedule.to_dicts()
        }
        client = Client()
        prompt = f"You are an NFL scheduling expert. You have been given raw JSON schedule data for a team.\nAnswer the following user query based ONLY on the provided data.\n\nUser Query: {query}\n\nRaw Data: {json.dumps(result)}"
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        return response.text
    except Exception as e:
        return f"Error retrieving schedule for {team_abbr} in {year}: {e}"

def get_depth_chart(query: str, team_abbr: str, position: str, year: int = 2024) -> str:
    """Gets the depth chart for a specific position on a specific NFL team using nflreadpy.

    Args:
        query: The original question the user asked.
        team_abbr: The abbreviation of the NFL team (e.g., 'SF', 'KC').
        position: The position to check (e.g., 'QB', 'RB', 'WR', 'TE').
        year: The NFL season year (defaults to 2024).
        
    Returns:
        The depth chart for that position.
    """
    ssl._create_default_https_context = ssl._create_unverified_context
    try:
        df = nfl.load_depth_charts([year])
        if 'club_code' in df.columns:
            team_dc = df.filter((df['club_code'] == team_abbr.upper()) & (df['position'] == position.upper()) & (df['week'] == 1))
            if team_dc.height == 0:
                return f"Error: Could not find depth chart for {team_abbr} {position} in {year}."
            dc = team_dc.select(['full_name', 'depth_team']).sort('depth_team')
        else:
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
        client = Client()
        prompt = f"You are a professional NFL personnel scout. You have been given raw JSON depth chart data for a team positional group.\nAnswer the following user query based ONLY on the provided data. Give concise answers.\n\nUser Query: {query}\n\nRaw Data: {json.dumps(result)}"
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        return response.text
    except Exception as e:
        return f"Error retrieving depth chart for {team_abbr} {position} in {year}: {e}"


# ======================================================================
# 2. Define the BaseModel
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

# 3. Define the Specialized Level 1 Scouts (To be run in parallel)
offensive_scout = Agent(
    name="Offensive_Scout",
    model=model,
    description="Analyzes offensive performance and player statistics.",
    tools=[get_player_stats],
    instruction="""You are the Offensive Scout for an NFL team.
Your goal is to lookup player stats using get_player_stats to provide a detailed breakdown of the player's performance. Focus only on offense.
If the tool doesn't apply to the query, skip or ignore."""
)

defensive_scout = Agent(
    name="Defensive_Scout",
    model=model,
    description="Analyzes defensive matchups, recent scores, and upcoming schedule.",
    tools=[get_upcoming_schedule],
    instruction="""You are the Defensive Scout and Schedule Analyst.
Your goal is to use get_upcoming_schedule to look up the team's opponents. Focus on recent game scores and the toughness of upcoming opponents."""
)

roster_scout = Agent(
    name="Roster_Scout",
    model=model,
    description="Analyzes team depth charts and positional groups.",
    tools=[get_depth_chart],
    instruction="""You are the Roster and Personnel Scout.
Your goal is to use get_depth_chart to check positional starters, backups, and depth charts."""
)

# 4. Group into a Parallel Agent
scouting_department = ParallelAgent(
    name="Scouting_Department",
    description="Runs specialized scouts in parallel to gather intel.",
    sub_agents=[offensive_scout, defensive_scout, roster_scout]
)

# 5. Define the Top-Level Synthesizer
head_coach = Agent(
    name="Head_Coach",
    model=model,
    description="Synthesizes all the scouting reports into a final game plan.",
    instruction="""You are 'YourGM,' an elite NFL Head Coach and General Manager.
You have a Scouting Department that just gathered intelligence (Offense, Defense, and Roster).
Review all of the context provided by your scouts in this conversation and provide a final Weekly Preview and unified Game Plan to the user.
Be decisive, professional, and highlight the key to victory or the most important fantasy football insight. Do not call any tools yourself, just summarize."""
)

# 6. Build the Sequential Pipeline
weekly_preview_pipeline = SequentialAgent(
    name="Weekly_Preview_Pipeline",
    description="A pipeline to gather stats and then synthesize a game plan.",
    sub_agents=[scouting_department, head_coach]
)

def main():
    # 7. Initialize the Runner with the pipeline root
    runner = InMemoryRunner(agent=weekly_preview_pipeline)
    
    # Session Initialization for google-adk <= 1.8.0
    USER_ID = "user_123"
    SESSION_ID = "session_123"
    asyncio.run(runner.session_service.create_session(
        app_name=runner.app_name,
        user_id=USER_ID,
        session_id=SESSION_ID
    ))

    print("--- Digital NFL GM: System Online (Week 4 Edition - The Weekly Preview) ---")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "q", "x", "bye", "goodbye"]:
            break
        
        # 8. Query the Pipeline
        events = runner.run(
            user_id=USER_ID,
            session_id=SESSION_ID,
            new_message=types.Content(
                parts=[types.Part.from_text(text=user_input)],
                role="user"
            )
        )
        
        print("\nPipeline Output: ", end="", flush=True)
        for event in events:
            agent_label = getattr(event, 'agent_id', getattr(event, 'source', 'Agent'))
            if getattr(event, 'content', None) and getattr(event.content, 'parts', None):
                for part in event.content.parts:
                    if getattr(part, 'function_call', None):
                        print(f"\n[System: {agent_label} is calling tool '{part.function_call.name}'...]", end="", flush=True)
                    if getattr(part, 'text', None):
                        # Print which agent is speaking
                        print(f"\n[{agent_label}]: {part.text}", end="", flush=True)
        print("\n")

if __name__ == "__main__":
    main()
