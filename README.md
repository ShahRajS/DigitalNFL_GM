# 🏈 The Digital NFL GM: An 8-Week Agentic AI Series

Welcome to **The Digital NFL GM**! This project is a step-by-step guide to building a production-grade AI Scouting Assistant using **Google’s Agent Development Kit (ADK)** and **Gemini**.

Instead of a simple chatbot, we are building an agentic system that can reason about NFL data, use specialized tools, query databases, search the web, and utilize a Retrieval-Augmented Generation (RAG) pipeline to provide front-office level insights for your Fantasy Football and team management needs.

---

## 📅 The 8-Week Roadmap

Each week corresponds to a new agentic feature. Below is the current progress of the project:

* **Week 1: Into the Mind of YourGM** – Setting up the `LlmAgent` and basic reasoning.
* **Week 2: Giving Your GM "Hands"** – Integrating `FunctionTools` with NFL APIs.
* **Week 3: Building Your Coaching Staff** – Multi-agent orchestration and delegation.
* **Week 4: The Weekly Preview** – Using `SequentialAgent` for deterministic pipelines.
* **Week 5: Draft Room Memory** – Implementing `SessionState` for long-term context.
* **Week 6: Deep Scouting** – Connecting to local data via **Model Context Protocol (MCP)**.
* **Week 7: The Fact Checker** – Adding "Reflect & Retry" guardrails for data accuracy.
* **Week 8: Scalable RAG & Cloud Databases (Pinecone)** – Transitioning from local context to a production-grade cloud Vector DB (Pinecone) with automated PDF ingestion, rate-limiting handlers for Gemini embeddings, PostgreSQL MCP integration (Supabase), and Brave Search.

---

## 📁 Week 7 Core Components

The implementation is located in the [week-07/](file:///Users/rajshah/Desktop/DigitalNFL_GM/week-07) directory. Here are the key files and their purposes:

1. **[week-07/main.py](file:///Users/rajshah/Desktop/DigitalNFL_GM/week-07/main.py)**: The main execution script. It reads the local PDF [2025-San-Francisco-49ers-Draft-Packet.pdf](file:///Users/rajshah/Desktop/DigitalNFL_GM/week-07/2025-San-Francisco-49ers-Draft-Packet.pdf) using `PyPDF2` on startup, parses the text content, and injects it directly into the agent's system prompt context. It also launches the background MCP Postgres and MCP Brave Search servers to handle structured SQL databases and live web search.

---

## 📁 Week 8 Core Components

The active implementation is located in the [week-08/](file:///Users/rajshah/Desktop/DigitalNFL_GM/week-08) directory. Here are the key files and their purposes:

1. **[week-08/main.py](file:///Users/rajshah/Desktop/DigitalNFL_GM/week-08/main.py)**: The entry point that sets up the `Draft_Scout` ADK agent. It orchestrates:
   - A PostgreSQL tool (`execute_db_query`) that communicates with Supabase using the Postgres MCP Server.
   - A live web search tool (`execute_web_search`) using the Brave Search MCP Server.
   - Vector search tools (`query_draft_packet` and `add_note_to_draft_database`) to interact with Pinecone.
   - Session state tools (`add_player_to_draft_board`, `remove_player_from_draft_board`, `view_draft_board`) to track draft selections interactively.
2. **[week-08/rag_indexer.py](file:///Users/rajshah/Desktop/DigitalNFL_GM/week-08/rag_indexer.py)**: The pipeline for document parsing and indexing. It features:
   - Page-by-page extraction from PDFs using `PyPDF2` with special character sanitization to avoid ADK parsing issues.
   - A custom sliding-window text chunker (e.g. 1000 characters with 200 character overlap).
   - Embedding generation using `models/gemini-embedding-2` with a robust exponential backoff handler for 429 rate limit errors.
   - A smart resumption mechanism that checks existing vectors in Pinecone to avoid re-embedding chunks that have already been indexed.
3. **[week-08/rag_utils.py](file:///Users/rajshah/Desktop/DigitalNFL_GM/week-08/rag_utils.py)**: Low-level vector utilities that interface with Pinecone for searching and upserting data. On system startup, it automatically checks the health/status of the `digital-nfl-gm` Pinecone index and seeds it if it is missing or empty.

---

## 🛠️ Setup & Installation

Follow these steps to set up and run the Week 8 Agentic Scouting system:

### 1. Clone the Repository
```bash
git clone https://github.com/ShahRajS/digitalNFL_GM.git
cd digitalNFL_GM
```

### 2. Configure Environment Variables
Create a file named `.env` in the root of the project directory. Add your API credentials as described in the **Environment Variables Guide** below.

### 3. Install Dependencies
Make sure you have Python 3.12+ (e.g., using conda or venv) and install the packages defined in [requirements.txt](file:///Users/rajshah/Desktop/DigitalNFL_GM/requirements.txt):
```bash
pip install -r requirements.txt
```

### 4. Provide Source Data
To query the team's local draft packet files, ensure the following PDF files are located in their respective directories:
* **For Week 7**:
  - `week-07/2025-San-Francisco-49ers-Draft-Packet.pdf`
* **For Week 8**:
  - `week-08/2025-San-Francisco-49ers-Draft-Packet.pdf`
  - `week-08/2026-San-Francisco-49ers-Draft-Packet.pdf`
  - `week-08/San-Francisco-49ers-2025-Season-Review.pdf`

---

## 🔑 Environment Variables Guide (`.env`)

Create a [.env](file:///Users/rajshah/Desktop/DigitalNFL_GM/.env) file in the root project directory and populate the following keys:

```bash
# Google GenAI API Key (Required for model inference and Gemini embeddings)
GEMINI_API_KEY="your-gemini-api-key"
# GOOGLE_API_KEY="your-google-api-key" # (Fallback key alternative)

# Pinecone Cloud Vector DB API Key (Required for RAG)
PINECONE_API_KEY="your-pinecone-api-key"

# Supabase PostgreSQL Connection String (Required for Database MCP)
DATABASE_URL="postgresql://postgres:[password]@db.[project-ref].supabase.co:5432/postgres"

# Brave Search API Key (Required for Web Search MCP)
BRAVE_API_KEY="your-brave-search-api-key"
```

---

## 🚀 Running the System

Start the interactive console loop for the desired week:

**To run Week 8 (Pinecone Cloud RAG & Databases):**
```bash
python week-08/main.py
```

**To run Week 7 (Injected Prompt PDF Context):**
```bash
python week-07/main.py
```

### What happens behind the scenes?
1. **RAG Vector Seeding**: The system verifies if the `digital-nfl-gm` Pinecone index is online. If the index is empty or missing, the indexer parses `2025-San-Francisco-49ers-Draft-Packet.pdf` automatically and chunks/embeds it into Pinecone.
2. **MCP Connection Startup**: The system launches the PostgreSQL and Brave Search MCP servers in the background using `npx` stdio client connections.
3. **Session Initialized**: An `InMemoryRunner` session is generated for user tracking (`user_123` / `session_123`).
4. **Interactive Chat**: You can enter prompts like:
   - *"Tell me the top 3 prospects for the 2026 draft."* (searches the vector database)
   - *"Search the web for Christian McCaffrey's latest injury update."* (runs Brave Web Search)
   - *"Add Tetairoa McMillan to the draft board."* (stores to Session State)
   - *"Query the database tables to find what players are in our records."* (queries PostgreSQL tables)

---

## 🏗️ Project Architecture

```mermaid
graph TD
    User[User Console] <--> Runner[InMemoryRunner]
    Runner <--> Agent[YourGM - Draft_Scout Agent]
    
    subgraph Tools [Specialized Tool Suite]
        Agent <--> SQLTool[execute_db_query]
        Agent <--> WebTool[execute_web_search]
        Agent <--> PineconeQuery[query_draft_packet]
        Agent <--> PineconeNote[add_note_to_draft_database]
        Agent <--> Memory[Draft Board Session Memory]
    end
    
    subgraph External [External Services & Clients]
        SQLTool <--> MCP_PG[MCP Postgres Server] <--> Supabase[(Supabase PostgreSQL)]
        WebTool <--> MCP_Brave[MCP Brave Search Server] <--> BraveAPI[Brave Search API]
        PineconeQuery <--> PineconeDB[(Pinecone Vector Cloud)]
        PineconeNote <--> PineconeDB
        PineconeQuery <--> GeminiEmbed[Gemini Embedding API]
    end
```

---

## 🤝 Contributing

This is an educational series! Feel free to open issues or discussions if you find better ways to implement scouting logic using ADK or if you run into any dependency / setup issues.

