# 🏈 The Digital NFL GM: An 8-Week Agentic AI Series

Welcome to **The Digital NFL GM**! This project is a step-by-step guide to building a production-grade AI Scouting Assistant using **Google’s Agent Development Kit (ADK)** and **Gemini**.

Instead of a simple chatbot, we are building an Agentic system that can reason about NFL data, use specialized tools, and provide front-office level insights for your Fantasy Football needs.

---

## 📅 The 8-Week Roadmap

Each week will correspond to a new feature and a specific Pull Request (PR) in this repository. I will be updating this README with the progress of each week with any new Information!

* **Week 1: Into the Mind of YourGM** – Setting up the `LlmAgent` and basic reasoning.
* **Week 2: Giving Your GM "Hands"** – Integrating `FunctionTools` with NFL APIs.
* **Week 3: Building Your Coaching Staff** – Multi-agent orchestration and delegation.
* **Week 4: Draft Room Memory** – implementing `SessionState` for long-term context.
* **Week 5: The Weekly Preview** – Using `SequentialAgent` for deterministic pipelines.
* **Week 6: Deep Scouting** – Connecting to local data via **Model Context Protocol (MCP)**.
* **Week 7: The Fact Checker** – Adding "Reflect & Retry" guardrails for data accuracy.
* **Week 8: Game Day Launch** – Deploying to Google Cloud/Vertex AI Agent Engine.

---

## 🛠️ Setup & Installation

1. **Clone the repo:**
```bash
git clone https://github.com/ShahRajS/digitalNFL_GM.git
cd digitalNFL_GM
```


2. **Install dependencies:**
```bash
pip install -r requirements.txt
```


3. **Configure Environment Variables:**
Create a `.env` file in the root directory. See the **Environment Variables Guide** below for required keys.
4. **Run the Week 1 Demo:**
```bash
python week-01-foundations/main.py
```


---

## 🔑 Environment Variables Guide (`.env`)

This file will be updated weekly. Copy the template below into your root project directory and add your keys.

```bash
# === WEEK 1: Foundations ===
# Your Google Cloud Project ID where Vertex AI is enabled
GOOGLE_CLOUD_PROJECT="your-project-id"
# The region for your Vertex AI resources (e.g., us-central1)
GOOGLE_CLOUD_LOCATION="us-central1"
```

---

## 🏗️ Project Architecture

The Digital GM uses a modular architecture where the **Runner** acts as the interface, and the **Agent** acts as the decision-maker.

## 🤝 Contributing

This is an educational series! Feel free to open issues or discussions if you find better ways to implement scouting logic using ADK.
