from pathlib import Path
import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    import faiss
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    SentenceTransformer = None

MODEL_NAME = "all-MiniLM-L6-v2"
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "catalog.json"
INDEX_PATH = BASE_DIR / "data" / "catalog_index.faiss"
PICKLE_PATH = BASE_DIR / "data" / "catalog_df.pkl"

logger = logging.getLogger(__name__)
_model: Any | None = None
_index: Any | None = None
_df: pd.DataFrame | None = None
_tfidf_vectorizer: TfidfVectorizer | None = None
_tfidf_matrix = None


def _safe_join(value: object) -> str:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value) if value is not None else ""


def _build_search_text(row: pd.Series) -> str:
    return " ".join(
        [
            str(row.get("name", "")),
            str(row.get("description", "")),
            _safe_join(row.get("keys", [])),
            _safe_join(row.get("job_levels", [])),
            _safe_join(row.get("languages", [])),
        ]
    )


def _load_dataframe() -> pd.DataFrame:
    global _df
    if _df is not None:
        return _df

    if PICKLE_PATH.exists():
        try:
            _df = pd.read_pickle(PICKLE_PATH)
        except Exception as exc:  # noqa: BLE001 - compatibility fallback
            logger.warning(
                "Could not load %s (%s). Falling back to %s.",
                PICKLE_PATH,
                type(exc).__name__,
                DATA_PATH,
            )
            _df = pd.read_json(DATA_PATH, lines=True)
    else:
        _df = pd.read_json(DATA_PATH, lines=True)

    if "retrieval_text" not in _df.columns:
        _df["retrieval_text"] = _df.apply(_build_search_text, axis=1)
    return _df


def _load_index():
    global _index
    if _index is None:
        if faiss is None:
            raise RuntimeError(
                "faiss is not installed in the active Python environment."
            )
        if not INDEX_PATH.exists():
            raise FileNotFoundError(f"Index file not found: {INDEX_PATH}")
        _index = faiss.read_index(str(INDEX_PATH))
    return _index


def _load_model():
    global _model
    if _model is None:
        if SentenceTransformer is None:
            raise RuntimeError(
                "sentence-transformers is not installed in the active "
                "Python environment."
            )
        _model = SentenceTransformer(MODEL_NAME, local_files_only=True)
    return _model


def _semantic_search(query: str, top_k: int) -> pd.DataFrame:
    df = _load_dataframe()
    index = _load_index()
    model = _load_model()

    query_embedding = model.encode([query], show_progress_bar=False)
    query_embedding = np.asarray(query_embedding, dtype="float32")
    distances, indices = index.search(query_embedding, top_k)

    results = df.iloc[indices[0]].copy()
    results["score"] = distances[0]
    results["search_mode"] = "semantic"
    return results.sort_values("score", ascending=True).reset_index(drop=True)


def _keyword_search(query: str, top_k: int) -> pd.DataFrame:
    global _tfidf_matrix, _tfidf_vectorizer

    df = _load_dataframe()
    corpus = df["retrieval_text"].fillna("").tolist()

    if _tfidf_vectorizer is None or _tfidf_matrix is None:
        _tfidf_vectorizer = TfidfVectorizer(stop_words="english")
        _tfidf_matrix = _tfidf_vectorizer.fit_transform(corpus)

    query_vec = _tfidf_vectorizer.transform([query])
    scores = (_tfidf_matrix @ query_vec.T).toarray().ravel()

    top_indices = np.argsort(scores)[::-1][:top_k]
    results = df.iloc[top_indices].copy()
    results["score"] = scores[top_indices]
    results["search_mode"] = "keyword_fallback"
    return results.sort_values("score", ascending=False).reset_index(drop=True)


def search_assessments(query: str, top_k: int = 5) -> pd.DataFrame:
    if not query or not query.strip():
        raise ValueError("Query must be a non-empty string.")

    df = _load_dataframe()
    limit = max(1, min(top_k, len(df)))

    try:
        return _semantic_search(query=query, top_k=limit)
    except Exception as exc:  # noqa: BLE001 - fallback behavior is intentional
        logger.warning(
            "Semantic search unavailable (%s). Falling back to keyword search.",
            exc,
        )
        return _keyword_search(query=query, top_k=limit)
