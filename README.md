# SHL Chat API

Stateless FastAPI service for hiring-assessment recommendations.

## Endpoints

- `GET /health` -> `{"status":"ok"}`
- `POST /chat` -> strict schema:
  - `reply: string`
  - `recommendations: [] | [1..10 items]`
  - `end_of_conversation: boolean`

`POST /chat` expects full message history in each request:

```json
{
  "messages": [
    {"role": "user", "content": "Hiring a Java developer who works with stakeholders"},
    {"role": "assistant", "content": "Sure. What seniority level should I target?"},
    {"role": "user", "content": "Mid-level, around 4 years"}
  ]
}
```

## Local Run

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

Quick test:

```bash
curl -s http://127.0.0.1:8000/health
curl -s -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hiring Java developer"}]}'
```

## Render Deployment (Free)

This repo includes `render.yaml` and `runtime.txt`.

1. Push repo to GitHub.
2. In Render: New + -> Blueprint -> select repo.
3. Render reads `render.yaml` and deploys automatically.
4. Health check path is `/health`.

## Notes

- Service is stateless by design.
- Off-topic requests are safely refused.
- Retrieval includes hybrid re-ranking, intent-tag boosting, diversification, and comparison workflow.
# smart-assessment-agent
