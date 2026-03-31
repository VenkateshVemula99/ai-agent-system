import os
import json
import re
import streamlit as st
import google.generativeai as genai
from langgraph.graph import StateGraph, END

# ---------------- CONFIG ----------------
MODEL_NAME = "gemini-2.5-flash"

# ---------------- INIT MODEL ----------------
def init_model():
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        genai.configure(api_key=api_key)
        return genai.GenerativeModel(MODEL_NAME)
    except Exception as e:
        st.error(f"API Key Error: {e}")
        return None

# ---------------- LLM ----------------
def llm(prompt, model):
    try:
        response = model.generate_content(prompt)
        return response.text.strip() if response.text else "⚠️ Empty response"
    except Exception as e:
        return f"LLM ERROR: {e}"

# ---------------- TOOLS ----------------
class Tools:
    def web_search(self, query):
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=3))
                return json.dumps(results, indent=2)
        except Exception as e:
            return f"Search failed: {e}"

tools = Tools()

def execute_tool(tool_name, params):
    if hasattr(tools, tool_name):
        return getattr(tools, tool_name)(**params)
    return "Invalid tool"

# ---------------- AGENTS ----------------
def planner_agent(state):
    prompt = f"Create a step-by-step plan for:\n{state['user_query']}"
    state["plan"] = llm(prompt, state["model"])
    return state

def worker_agent(state):
    state["worker_calls"] += 1

    prompt = f"""
User Query: {state['user_query']}
Plan: {state['plan']}

If tool needed return JSON:
{{"tool": "web_search", "params": {{"query": "something"}}}}

Otherwise give final answer.
"""

    output = llm(prompt, state["model"])

    data = None
    try:
        data = json.loads(output)
    except:
        match = re.search(r"\{.*\}", output, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except:
                pass

    if data and "tool" in data:
        result = execute_tool(data["tool"], data.get("params", {}))
        state["draft_response"] = llm(f"Use this data:\n{result}", state["model"])
    else:
        state["draft_response"] = output

    return state

def reviewer_agent(state):
    state["reviewer_calls"] += 1

    review = llm(
        f"Check this answer and say approve or revise:\n{state['draft_response']}",
        state["model"]
    )

    state["review_decision"] = "approve" if "approve" in review.lower() else "revise"
    return state

# ---------------- ROUTER ----------------
def router(state):
    if state["review_decision"] == "approve" or state["reviewer_calls"] >= 2:
        return "__end__"
    return "worker"

# ---------------- GRAPH ----------------
workflow = StateGraph(dict)

workflow.add_node("planner", planner_agent)
workflow.add_node("worker", worker_agent)
workflow.add_node("reviewer", reviewer_agent)

workflow.set_entry_point("planner")

workflow.add_edge("planner", "worker")
workflow.add_edge("worker", "reviewer")

workflow.add_conditional_edges(
    "reviewer",
    router,
    {"worker": "worker", "__end__": END}
)

graph = workflow.compile()

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="AI Agent System", layout="wide")
st.title("🔥 Autonomous Multi-Agent AI System")

# Initialize model
model = init_model()

if model is None:
    st.stop()

# Chat memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Input
query = st.chat_input("Ask anything...")

if query:
    st.session_state.chat_history.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            state = {
                "user_query": query,
                "plan": "",
                "draft_response": "",
                "worker_calls": 0,
                "reviewer_calls": 0,
                "model": model
            }

            result = graph.invoke(state)

            answer = result.get("draft_response", "")
            plan = result.get("plan", "")

            st.write(answer)

            with st.expander("🧠 Plan"):
                st.write(plan)

            with st.expander("🔁 Iterations"):
                st.write(f"Worker Calls: {result.get('worker_calls')}")
                st.write(f"Reviewer Calls: {result.get('reviewer_calls')}")

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer
            })