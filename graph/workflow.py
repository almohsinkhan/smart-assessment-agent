from typing import List, TypedDict
from pathlib import Path
import logging
import os
import sys

from langgraph.graph import END, START, StateGraph

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

try:
    from groq import Groq
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    Groq = None
from retrieval.search import search_assessments

logger = logging.getLogger(__name__)
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
_groq_client = None
_groq_unavailable = False


class AgentState(TypedDict):
    query: str
    intent: str
    steps: List[str]
    results: List[str]
    response: str
    history: List[str]


def _append_step(state: AgentState, step: str) -> None:
    state["steps"] = [*(state.get("steps") or []), step]


def analyze_query(state: AgentState) -> AgentState:
    query = state.get("query", "").lower().strip()
    _append_step(state, "analyze_query")
    state["history"] = [*(state.get("history") or []), state.get("query", "")]

    if "compare" in query:
        state["intent"] = "compare"
    elif len(query.split()) < 3:
        state["intent"] = "clarify"
    else:
        state["intent"] = "retrieve"

    return state


def clarification_node(state: AgentState) -> AgentState:
    _append_step(state, "clarification_node")
    state["results"] = state.get("results", [])
    state["response"] = "Could you specify the role, skills, or seniority level?"
    return state


def retrieval_node(state: AgentState) -> AgentState:
    _append_step(state, "retrieval_node")
    results = search_assessments(state.get("query", ""), top_k=3)
    state["results"] = results["name"].tolist()
    return state


def compare_node(state: AgentState) -> AgentState:
    _append_step(state, "compare_node")
    results = search_assessments(state.get("query", ""), top_k=5)
    state["results"] = results["name"].tolist()[:3]
    return state


def _fallback_response(intent: str, results: List[str]) -> str:
    if not results:
        return "I couldn't find any relevant assessments based on your query."
    if intent == "compare":
        if len(results) >= 2:
            return f"You can compare these assessments: {results[0]} and {results[1]}."
        return f"I found one assessment to compare: {results[0]}."
    return (
        "Based on your query, I found the following assessments: "
        f"{', '.join(results)}."
    )


def _get_groq_client():
    global _groq_client, _groq_unavailable

    if _groq_unavailable:
        return None
    if _groq_client is not None:
        return _groq_client
    if Groq is None:
        _groq_unavailable = True
        logger.warning("groq package not installed. Using deterministic fallback.")
        return None

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        _groq_unavailable = True
        logger.warning("GROQ_API_KEY not set. Using deterministic fallback response.")
        return None

    try:
        _groq_client = Groq(api_key=api_key)
    except Exception as exc:  # noqa: BLE001 - guard LLM path
        _groq_unavailable = True
        logger.warning("Could not initialize Groq client (%s).", exc)
        return None

    return _groq_client


def _generate_llm_response(state: AgentState) -> str:
    intent = state.get("intent", "retrieve")
    query = state.get("query", "")
    results = state.get("results", [])
    fallback = _fallback_response(intent=intent, results=results)
    client = _get_groq_client()
    if client is None:
        return fallback

    model = os.environ.get("GROQ_MODEL", DEFAULT_GROQ_MODEL)
    results_block = "\n".join(f"- {name}" for name in results) or "- No results found"

    system_prompt = (
        "You are an SHL assessment recommendation assistant. "
        "Generate a concise, helpful response using only the provided candidates. "
        "Do not invent assessment names."
    )
    user_prompt = (
        f"User query: {query}\n"
        f"Intent: {intent}\n"
        f"Candidate assessments:\n{results_block}\n\n"
        "If intent is compare, compare top options briefly. "
        "If no candidates, ask for clearer role, skills, and seniority."
    )

    try:
        completion = client.chat.completions.create(
            model=model,
            temperature=0.2,
            max_tokens=220,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = completion.choices[0].message.content if completion.choices else ""
        return content.strip() or fallback
    except Exception as exc:  # noqa: BLE001 - fallback on API failures
        logger.warning("Groq generation failed (%s). Using deterministic fallback.", exc)
        return fallback


def response_generation_node(state: AgentState) -> AgentState:
    _append_step(state, "response_generation_node")
    state["response"] = _generate_llm_response(state)
    return state


def route(state: AgentState) -> str:
    return state.get("intent", "clarify")


def make_workflow():
    graph = StateGraph(AgentState)

    graph.add_node("analyze_query", analyze_query)
    graph.add_node("clarification_node", clarification_node)
    graph.add_node("retrieval_node", retrieval_node)
    graph.add_node("compare_node", compare_node)
    graph.add_node("response_generation_node", response_generation_node)

    graph.add_edge(START, "analyze_query")
    graph.add_conditional_edges(
        "analyze_query",
        route,
        {
            "clarify": "clarification_node",
            "retrieve": "retrieval_node",
            "compare": "compare_node",
        },
    )
    graph.add_edge("clarification_node", END)
    graph.add_edge("retrieval_node", "response_generation_node")
    graph.add_edge("compare_node", "response_generation_node")
    graph.add_edge("response_generation_node", END)

    return graph.compile()


if __name__ == "__main__":
    app = make_workflow()
    result = app.invoke(
        {
            "query": "I want to find assessments for a backend developer role with communication skills.",
            "intent": "",
            "steps": [],
            "results": [],
            "response": "",
            "history": [],
        }
    )
    print(result["response"])
