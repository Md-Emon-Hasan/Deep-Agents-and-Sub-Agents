# Learning Deep Agents and Sub Agents

<p align="center">
  <img src="https://github.com/user-attachments/assets/aebc0b66-b62f-49ce-ae79-c10d1df4e504" width="100%" alt="Deep Agents and Sub Agents" />
</p>

A learning-by-building notebook for understanding **deep agents** and **sub-agents** from the ground up, using LangChain, LangGraph, and Groq.

This is not a polished framework or a library. It is a learning and exploration project — a single, well-commented notebook where each idea is built step by step so you can actually see how an agent thinks, plans, delegates, and finishes a task. The goal is understanding, not production.

---

## Table of Contents

- [What This Project Is](#what-this-project-is)
- [What You Will Learn](#what-you-will-learn)
- [The Four Pillars of a Deep Agent](#the-four-pillars-of-a-deep-agent)
- [Suggested Learning Path](#suggested-learning-path)
- [Notebook Walkthrough](#notebook-walkthrough)
- [Requirements](#requirements)
- [Setup](#setup)
- [Getting a Groq API Key](#getting-a-groq-api-key)
- [Try It Yourself: First Experiments](#try-it-yourself-first-experiments)
- [Exercises to Deepen Your Understanding](#exercises-to-deepen-your-understanding)
- [Tech Stack](#tech-stack)
- [Author](#author)
- [License](#license)

---

## What This Project Is

This repository is a personal study notebook on how agentic systems are put together. It is meant to be read and run, not imported. Every concept appears in two forms so you can compare them:

- **The standard way** — the conventional approach using native tool-calling and the `deepagents` package. Kept as a reference so you recognize the patterns you will see in tutorials and docs.
- **The Groq-working way** — a JSON-driven alternative built from scratch in the notebook, so you understand what is happening underneath instead of treating the framework as a black box.

Seeing both side by side is itself part of the learning: you find out what a framework abstracts away, and why.

---

## What You Will Learn

By working through the notebook you should come away understanding:

- What an agent actually is — an LLM plus tools plus a loop — and what the ReAct pattern (Reason, Act, Observe, repeat) really does.
- Why native tool-calling fails on some providers, and what a tool call looks like under the hood.
- How to build your own agent loop from nothing, with no framework magic.
- What a deep agent adds on top of a plain agent: planning, memory through files, delegation, and a strong prompt.
- How sub-agents work, using the "agent-as-a-tool" pattern, and when delegating is worth it.
- How retrieval-augmented generation (RAG) fits into an agent without needing tool-calling at all.
- When an autonomous loop is the right choice and when a fixed pipeline is more reliable.
- How an agent can test and debug its own output, using a coding agent as the example.

---

## The Four Pillars of a Deep Agent

A plain agent is an LLM, some tools, and a loop. A **deep agent** is that, plus four things you will build one at a time:

1. **Planning** — the agent writes a todo list before starting and checks items off as it goes.
2. **File system** — the agent saves notes and drafts to files and reads them back, giving it working memory.
3. **Sub-agents** — the agent hands focused work to specialist agents wrapped as tools (agent-as-a-tool).
4. **Detailed prompt** — a careful system prompt that ties the other three together into coherent behavior.

The notebook builds all four on top of `run_agent`, so you see exactly where each pillar lives.

---

## Suggested Learning Path

If you are new to agents, read and run in this order and pause to experiment between parts:

1. Run Part 1 and deliberately trigger the failure, then study `run_agent` until you can explain it to someone else.
2. Do Part 2 and trace the loop by reading the printed steps.
3. Build the pillars in Part 3 one at a time; change a prompt and watch the behavior shift.
4. Compare Part 4's standard `deepagents` reference with the manual builder right next to it.
5. Treat Parts 5 to 8 as worked examples of real use cases; tweak them rather than just reading.
6. Use Part 9 as a checklist and a source of practice tasks.

---

## Notebook Walkthrough

| Part | Topic | What you explore |
|------|-------|------------------|
| 0 | Setup | Connect to Groq |
| 1 | Foundation | See the failure happen, then build the JSON-based `run_agent` engine |
| 2 | ReAct agent | Watch a ReAct loop run, standard vs Groq-working |
| 3 | The 4 pillars | Build planning, files, sub-agents, and a detailed prompt by hand |
| 4 | The `deepagents` package | Compare the standard package with a manual Groq-working equivalent |
| 5 | Research assistant | An orchestrator agent plus a research sub-agent writing a multi-section report |
| 6 | RAG support bot | Local embeddings plus FAISS retrieval plus an escalation tool |
| 7 | Multi-step automation | A fixed, predictable pipeline using single-shot sub-agents |
| 8 | Coding agent | Plan, code, test, self-debug, review |
| 9 | Best practices and exercises | Lessons learned and tasks to try |

---

## Requirements

- Python 3.10 or newer (developed on Python 3.11)
- A Groq API key
- Jupyter Notebook, JupyterLab, or VS Code with the Jupyter extension
- Internet access (for web search and model calls)

Main packages (also installed in Part 0):

`langgraph`, `langchain`, `langchain-groq`, `langchain-community`, `duckduckgo-search`, `faiss-cpu`, `langchain-huggingface`, `sentence-transformers`

---

## Setup

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate

pip install -U langgraph langchain langchain-groq langchain-community \
    duckduckgo-search faiss-cpu langchain-huggingface sentence-transformers
```

You can also skip this and let Part 0 of the notebook install everything for you.

---

## Getting a Groq API Key

1. Create an account at the Groq Console (`https://console.groq.com`).
2. Open the API Keys section and create a key.
3. Provide it to the notebook in one of two ways.

Set an environment variable before launching Jupyter:

```bash
export GROQ_API_KEY="your_key_here"   # on Windows: set GROQ_API_KEY=your_key_here
```

Or just run the setup cell — if the variable is missing, the notebook prompts you to paste it securely with `getpass`.

---

## Try It Yourself: First Experiments

A minimal run of the engine you just learned:

```python
from langchain_core.tools import tool

@tool
def add(a: float, b: float) -> float:
    """Adds two numbers."""
    return a + b

@tool
def multiply(a: float, b: float) -> float:
    """Multiplies two numbers."""
    return a * b

ans = run_agent(
    llm,
    "You are a careful math assistant. Do calculations using the tools.",
    [add, multiply],
    "First add 23478 and 100, then multiply the result by 99211.",
)
print(ans)
```

Good first things to try changing:

- Set `verbose=True` and read every step the agent takes.
- Add a new tool of your own and see if the agent decides to use it.
- Loosen or tighten the system prompt and watch how the behavior changes.
- Give it a task that needs two tools in sequence and confirm it chains them.

---

## Exercises to Deepen Your Understanding

1. Easy — Replace the Part 6 knowledge base with 5 to 10 FAQs for a product you know.
2. Easy — Add a `tweet_writer` sub-agent to the Part 5 research assistant.
3. Medium — Make `run_agent` return a verbose trace (every step collected in a list).
4. Medium — Add an `optimize` step to the Part 8 coding agent that makes working code faster.
5. Hard — Build a small "startup team": a product manager, an engineer, and a QA sub-agent that turn a feature spec into a plan, code, and tests.
6. Hard — Validate tool arguments with Pydantic inside `run_agent` and feed fix-feedback to the model when the arguments are wrong.

---

## Tech Stack

- **LangChain** — tools, messages, and document abstractions
- **LangGraph** — the reference standard ReAct agent
- **Groq** — fast LLM inference (`llama-3.3-70b-versatile` and others)
- **FAISS** — local vector store for the RAG example
- **Sentence Transformers / Hugging Face** — local embeddings (`all-MiniLM-L6-v2`)
- **DuckDuckGo Search** — web search for the research examples

---

## Author

**Md Emon Hasan**

- Email: [emon.mlengineer@gmail.com](mailto:emon.mlengineer@gmail.com)
- LinkedIn: [md-emon-hasan](https://www.linkedin.com/in/md-emon-hasan-695483237/)
- GitHub: [Md-Emon-Hasan](https://github.com/Md-Emon-Hasan)
- Facebook: [Md-Emon-Hasan](https://www.facebook.com/mdemon.hasan2001/)
- WhatsApp: [+8801834363533](https://wa.me/8801834363533)

---

## License

Released under the MIT License. Add a `LICENSE` file to the repository to make it official.