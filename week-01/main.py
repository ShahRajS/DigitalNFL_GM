from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.genai import types

# 1. Load your .env from the root directory
load_dotenv()

# 2. Define YourGM
# This is the "Brain" of your project.

retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1, # Initial delay before first retry (in seconds)
    http_status_codes=[429, 500, 503, 504] # Retry on these HTTP errors
)

root_agent = Agent(
    name="NFL_GM",
    model=Gemini(
        model="gemini-2.5-flash",
        retry_options=retry_config
    ),
    description="A simple agent that can answer general NFL based questions.",
    instruction="You are 'YourGM,' an elite NFL scout and Fantasy Football expert. Your goal is to provide data-driven insights on player performance and draft strategy. Be concise, professional, and use scouting terminology. Examples such as: ('ADP', 'Target Share', 'Red Zone Efficiency').",
)

import asyncio

def main():
    # 3. Initialize the Runner
    # InMemoryRunner is perfect for Week 1 testing as it handles the loop locally.
    runner = InMemoryRunner(agent=root_agent)
    
    # For google-adk 1.8.0, we must explicitly create the session before using it.
    USER_ID = "user_123"
    SESSION_ID = "session_123"
    asyncio.run(runner.session_service.create_session(
        app_name=runner.app_name,
        user_id=USER_ID,
        session_id=SESSION_ID
    ))

    print("--- Digital NFL GM: System Online ---")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        
        # 4. Query the Agent
        # The runner returns a generator of events. We'll iterate to find the response.
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
            # Each event may contain parts of the response
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        print(part.text, end="", flush=True)
        print("\n")

if __name__ == "__main__":
    main()