from __future__ import annotations

import re
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator

from retrieval.search import search_assessments


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=4000)

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("content must not be blank.")
        return cleaned


class ChatRequest(BaseModel):
    messages: list[Message] = Field(..., min_length=1, max_length=8)


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: Literal["K", "P", "A", "S"]


class ChatResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation]
    end_of_conversation: bool


class HealthResponse(BaseModel):
    status: Literal["ok"]


app = FastAPI(title="SHL Chat API", version="1.0.0")

ROLE_PATTERN = re.compile(
    r"\b("
    r"developer|engineer|analyst|manager|architect|administrator|admin|"
    r"consultant|scientist|specialist|designer|tester|qa|devops|sre|programmer"
    r")\b",
    re.IGNORECASE,
)
SENIORITY_PATTERN = re.compile(
    r"\b("
    r"junior|jr\.?|mid|mid-level|mid level|senior|sr\.?|lead|principal|staff|"
    r"entry[- ]level|intern"
    r")\b|\b\d+\s*\+?\s*(years?|yrs?)\b",
    re.IGNORECASE,
)
CLOSING_PATTERN = re.compile(
    r"\b(thanks|thank you|thx|that'?s all|done|resolved)\b",
    re.IGNORECASE,
)
COMMUNICATION_TERMS = (
    "communication",
    "interpersonal",
    "stakeholder",
    "teamwork",
    "collaboration",
    "influence",
    "listening",
    "verbal",
    "presentation",
    "customer service",
)
TECHNICAL_TERMS = (
    "developer",
    "engineering",
    "programming",
    "coding",
    "software",
    "data",
    "cloud",
    "java",
    "python",
    "sql",
    "javascript",
    "react",
    "node",
    "linux",
    "devops",
)
COGNITIVE_TERMS = (
    "reasoning",
    "ability",
    "aptitude",
    "numerical",
    "verbal",
    "deductive",
    "inductive",
)
PERSONALITY_TERMS = (
    "opq",
    "personality",
    "motivation",
    "behavior",
    "leadership style",
    "managerial style",
)
COMPARISON_PATTERN = re.compile(
    r"\b(compare|comparison|vs|versus|difference|differentiate|between)\b",
    re.IGNORECASE,
)
HIRING_INTENT_TERMS = (
    "hire",
    "hiring",
    "candidate",
    "role",
    "assessment",
    "test",
    "screen",
    "evaluate",
    "developer",
    "engineer",
    "skills",
    "seniority",
    "compare",
)
OFFTOPIC_TERMS = (
    "ipl",
    "cricket",
    "football",
    "weather",
    "bitcoin",
    "stock",
    "movie",
    "song",
    "recipe",
    "celebrity",
    "election",
    "match score",
)
QUERY_TAG_MAP: dict[str, tuple[str, ...]] = {
    "communication": ("interpersonal",),
    "stakeholder": ("business communication",),
    "backend": ("software engineering",),
    "leadership": ("managerial",),
    "coding": ("technical",),
}
TAG_MATCH_TERMS: dict[str, tuple[str, ...]] = {
    "interpersonal": ("interpersonal", "communication", "collaboration", "teamwork"),
    "business communication": (
        "business communication",
        "stakeholder",
        "presentation",
        "verbal",
        "listening",
    ),
    "software engineering": (
        "software",
        "engineering",
        "developer",
        "backend",
        "programming",
    ),
    "managerial": ("managerial", "leadership", "management", "supervisor"),
    "technical": ("technical", "coding", "programming", "developer"),
}


def _user_messages(messages: list[Message]) -> list[str]:
    return [m.content for m in messages if m.role == "user"]


def _infer_test_type(name: str, description: str) -> Literal["K", "P", "A", "S"]:
    text = f"{name} {description}".lower()

    if any(token in text for token in ("opq", "personality", "motivation", "mq ")):
        return "P"
    if any(token in text for token in ("reasoning", "ability", "verify")):
        return "A"
    if any(token in text for token in ("simulation", "scenario", "interview")):
        return "S"
    return "K"


def _as_text(value: object) -> str:
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value) if value is not None else ""


def _row_text(row: object) -> str:
    return " ".join(
        [
            _as_text(row.get("name", "")),
            _as_text(row.get("description", "")),
            _as_text(row.get("keys", [])),
            _as_text(row.get("job_levels", [])),
            _as_text(row.get("languages", [])),
        ]
    ).lower()


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def _extract_intent_tags(query_text: str) -> set[str]:
    text = query_text.lower()
    tags: set[str] = set()
    for phrase, mapped_tags in QUERY_TAG_MAP.items():
        if phrase in text:
            tags.update(mapped_tags)
    return tags


def _is_off_topic(query_text: str) -> bool:
    text = query_text.lower()
    has_off_topic = _contains_any(text, OFFTOPIC_TERMS)
    has_hiring_intent = _contains_any(text, HIRING_INTENT_TERMS)
    return has_off_topic and not has_hiring_intent


def _is_soft_skill_assessment(row: object) -> bool:
    text = _row_text(row)
    inferred_type = _infer_test_type(
        name=str(row.get("name", "")),
        description=str(row.get("description", "")),
    )
    return _contains_any(text, COMMUNICATION_TERMS) or inferred_type in ("P", "S")


def _is_technical_assessment(row: object) -> bool:
    text = _row_text(row)
    return _contains_any(text, TECHNICAL_TERMS)


def _is_cognitive_assessment(row: object) -> bool:
    text = _row_text(row)
    inferred_type = _infer_test_type(
        name=str(row.get("name", "")),
        description=str(row.get("description", "")),
    )
    return inferred_type == "A" or _contains_any(text, COGNITIVE_TERMS)


def _is_personality_assessment(row: object) -> bool:
    text = _row_text(row)
    inferred_type = _infer_test_type(
        name=str(row.get("name", "")),
        description=str(row.get("description", "")),
    )
    return inferred_type == "P" or _contains_any(text, PERSONALITY_TERMS)


def _communication_boost(row: object) -> float:
    text = _row_text(row)
    matches = sum(term in text for term in COMMUNICATION_TERMS)
    if matches == 0:
        return 0.0
    return 0.18 + (0.05 * min(matches, 3))


def _metadata_tag_boost(row: object, intent_tags: set[str]) -> float:
    if not intent_tags:
        return 0.0
    text = _row_text(row)
    boost = 0.0
    for tag in intent_tags:
        terms = TAG_MATCH_TERMS.get(tag, ())
        if terms and _contains_any(text, terms):
            boost += 0.08
    return min(boost, 0.32)


def _pick_from_bucket(
    bucket: list[dict],
    selected: list[dict],
    selected_names: set[str],
    count: int,
) -> None:
    for candidate in bucket:
        if count <= 0:
            break
        name = str(candidate.get("name", ""))
        if not name or name in selected_names:
            continue
        selected.append(candidate)
        selected_names.add(name)
        count -= 1


def _hybrid_rank_results(results, query_text: str, target_k: int = 5):
    if results.empty:
        return results

    intent_tags = _extract_intent_tags(query_text)
    rows = list(results.to_dict(orient="records"))
    semantic_mode = str(rows[0].get("search_mode", "")).lower() == "semantic"
    total = max(len(rows), 1)

    scored_rows = []
    for index, row in enumerate(rows):
        # Keep strong influence from retrieval order, then apply hybrid boosts.
        score = 1.0 - (index / total)
        if semantic_mode:
            score += _communication_boost(row)
        score += _metadata_tag_boost(row, intent_tags)
        if _is_soft_skill_assessment(row):
            score += 0.03
        if _is_technical_assessment(row):
            score += 0.03

        row["_hybrid_score"] = score
        scored_rows.append(row)

    scored_rows.sort(key=lambda item: item["_hybrid_score"], reverse=True)

    technical = [row for row in scored_rows if _is_technical_assessment(row)]
    soft_pool = [row for row in scored_rows if _is_soft_skill_assessment(row)]
    communication = [row for row in scored_rows if _contains_any(_row_text(row), COMMUNICATION_TERMS)]
    personality = [row for row in scored_rows if _is_personality_assessment(row)]
    cognitive = [row for row in scored_rows if _is_cognitive_assessment(row)]

    selected: list[dict] = []
    selected_names: set[str] = set()

    # Diversification target for 5-item shortlist: 3 technical + 2 soft-skill.
    _pick_from_bucket(technical, selected, selected_names, count=3)
    _pick_from_bucket(soft_pool, selected, selected_names, count=2)

    # If soft pool underfills, try comm/personality explicitly before cognitive.
    if len(selected) < target_k:
        _pick_from_bucket(communication, selected, selected_names, count=target_k - len(selected))
    if len(selected) < target_k:
        _pick_from_bucket(personality, selected, selected_names, count=target_k - len(selected))

    if len(selected) < target_k:
        _pick_from_bucket(cognitive, selected, selected_names, count=target_k - len(selected))

    if len(selected) < target_k:
        for candidate in scored_rows:
            name = str(candidate.get("name", ""))
            if name in selected_names:
                continue
            selected.append(candidate)
            selected_names.add(name)
            if len(selected) >= target_k:
                break

    for row in selected:
        row.pop("_hybrid_score", None)

    return results.__class__(selected)


def _merge_unique_by_name(primary_results, secondary_results):
    rows: list[dict] = []
    seen_names: set[str] = set()

    for frame in (primary_results, secondary_results):
        if frame is None or frame.empty:
            continue
        for row in frame.to_dict(orient="records"):
            name = str(row.get("name", ""))
            if not name or name in seen_names:
                continue
            rows.append(row)
            seen_names.add(name)

    return primary_results.__class__(rows)


def _parse_comparison_targets(query_text: str) -> tuple[str | None, str | None]:
    text = query_text.strip()
    parts = re.split(r"\bvs\b|\bversus\b", text, flags=re.IGNORECASE)
    if len(parts) == 2:
        left = parts[0].strip(" ,.-")
        right = parts[1].strip(" ,.-")
        return (left or None, right or None)

    between_match = re.search(
        r"between\s+(.*?)\s+(?:and|&)\s+(.*)",
        text,
        flags=re.IGNORECASE,
    )
    if between_match:
        left = between_match.group(1).strip(" ,.-")
        right = between_match.group(2).strip(" ,.-")
        return (left or None, right or None)

    return (None, None)


def _get_single_best_result(query_text: str):
    results = search_assessments(query_text, top_k=1)
    if results.empty:
        return None
    return results.iloc[0]


def _comparison_summary(row_a: object, row_b: object) -> str:
    name_a = str(row_a.get("name", "Assessment A"))
    name_b = str(row_b.get("name", "Assessment B"))
    duration_a = str(row_a.get("duration", "NA"))
    duration_b = str(row_b.get("duration", "NA"))
    keys_a = _as_text(row_a.get("keys", []))
    keys_b = _as_text(row_b.get("keys", []))
    type_a = _infer_test_type(name_a, str(row_a.get("description", "")))
    type_b = _infer_test_type(name_b, str(row_b.get("description", "")))

    if type_a == type_b:
        recommendation_note = (
            "Both are the same assessment type; choose based on which topic focus best matches your role."
        )
    else:
        recommendation_note = (
            "These are complementary types; combine them for broader technical and behavioral coverage."
        )

    return (
        f"Comparison: {name_a} vs {name_b}. "
        f"{name_a} (type {type_a}, duration {duration_a}) focuses on {keys_a or 'technical knowledge'}. "
        f"{name_b} (type {type_b}, duration {duration_b}) focuses on {keys_b or 'competency assessment'}. "
        f"{recommendation_note}"
    )


def _clarification_reply(combined_user_text: str) -> str | None:
    has_role = bool(ROLE_PATTERN.search(combined_user_text))
    has_seniority = bool(SENIORITY_PATTERN.search(combined_user_text))

    if not has_role:
        return (
            "Sure. What role are you hiring for, and what are the top required skills?"
        )
    if not has_seniority:
        return "Sure. What seniority level should I target (e.g., junior, mid, senior)?"
    return None


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    users = _user_messages(request.messages)
    if not users:
        return ChatResponse(
            reply="Please share the hiring requirement so I can recommend assessments.",
            recommendations=[],
            end_of_conversation=False,
        )

    latest_user = users[-1]
    combined_user_text = " ".join(users)

    if _is_off_topic(latest_user):
        return ChatResponse(
            reply=(
                "I can only help with hiring assessment recommendations. "
                "Please share the job role, seniority, and required skills."
            ),
            recommendations=[],
            end_of_conversation=False,
        )

    if CLOSING_PATTERN.search(latest_user):
        return ChatResponse(
            reply="Happy to help. If you need a refined shortlist later, I can do that.",
            recommendations=[],
            end_of_conversation=True,
        )

    if COMPARISON_PATTERN.search(combined_user_text):
        left_target, right_target = _parse_comparison_targets(combined_user_text)
        if not left_target or not right_target:
            return ChatResponse(
                reply=(
                    "Sure. Please provide the two assessments to compare, "
                    "for example: 'Compare Java 8 (New) vs Java Frameworks (New)'."
                ),
                recommendations=[],
                end_of_conversation=False,
            )

        left_row = _get_single_best_result(left_target)
        right_row = _get_single_best_result(right_target)
        if left_row is None or right_row is None:
            return ChatResponse(
                reply=(
                    "I couldn't confidently find both assessments. "
                    "Please provide exact assessment names for comparison."
                ),
                recommendations=[],
                end_of_conversation=False,
            )

        comparison_recommendations = [
            Recommendation(
                name=str(left_row.get("name", "")),
                url=str(left_row.get("link", "")),
                test_type=_infer_test_type(
                    name=str(left_row.get("name", "")),
                    description=str(left_row.get("description", "")),
                ),
            ),
            Recommendation(
                name=str(right_row.get("name", "")),
                url=str(right_row.get("link", "")),
                test_type=_infer_test_type(
                    name=str(right_row.get("name", "")),
                    description=str(right_row.get("description", "")),
                ),
            ),
        ]

        return ChatResponse(
            reply=_comparison_summary(left_row, right_row),
            recommendations=comparison_recommendations,
            end_of_conversation=True,
        )

    clarification = _clarification_reply(combined_user_text)
    if clarification:
        return ChatResponse(
            reply=clarification,
            recommendations=[],
            end_of_conversation=False,
        )

    raw_results = search_assessments(combined_user_text, top_k=20)
    search_mode = (
        str(raw_results.iloc[0].get("search_mode", "")).lower()
        if not raw_results.empty
        else ""
    )
    if search_mode == "semantic":
        soft_query = "communication interpersonal stakeholder collaboration teamwork"
        soft_results = search_assessments(soft_query, top_k=10)
        raw_results = _merge_unique_by_name(raw_results, soft_results)

    results = _hybrid_rank_results(raw_results, query_text=combined_user_text, target_k=5)
    recommendations: list[Recommendation] = []

    for _, row in results.iterrows():
        recommendations.append(
            Recommendation(
                name=str(row.get("name", "")),
                url=str(row.get("link", "")),
                test_type=_infer_test_type(
                    name=str(row.get("name", "")),
                    description=str(row.get("description", "")),
                ),
            )
        )

    if not recommendations:
        return ChatResponse(
            reply=(
                "I could not find a confident shortlist yet. "
                "Please share role, seniority, and key skills."
            ),
            recommendations=[],
            end_of_conversation=False,
        )

    return ChatResponse(
        reply=(
            "Got it. Here are 5 assessments that fit your requirement. "
            "If you want, I can refine this list by seniority emphasis or domain focus."
        ),
        recommendations=recommendations[:10],
        end_of_conversation=False,
    )
