from graph.workflow import make_workflow
from retrieval.search import search_assessments


def run_smoke_test() -> None:
    results = search_assessments("backend developer communication", top_k=3)
    assert not results.empty, "Search returned no results."

    app = make_workflow()
    state = {
        "query": "compare backend and javascript assessments",
        "intent": "",
        "steps": [],
        "results": [],
        "response": "",
        "history": [],
    }
    output = app.invoke(state)
    assert output.get("response"), "Workflow produced an empty response."
    print("Workflow response:", output["response"])
    print("Smoke test passed.")
    


if __name__ == "__main__":
    run_smoke_test()
