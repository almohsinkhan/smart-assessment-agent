from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "catalog.json"
INDEX_PATH = BASE_DIR / "data" / "catalog_index.faiss"
PICKLE_PATH = BASE_DIR / "data" / "catalog_df.pkl"


def _safe_join(value: object) -> str:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value) if value is not None else ""


def create_retrieval_text(row: pd.Series) -> str:
    description = str(row.get("description") or "")
    short_description = description[:300]

    return (
        f"Assessment: {row.get('name', '')}\n"
        f"Categories: {_safe_join(row.get('keys', []))}\n"
        f"Job Levels: {_safe_join(row.get('job_levels', []))}\n"
        f"Languages: {_safe_join(row.get('languages', []))}\n"
        f"Duration: {row.get('duration', '')}\n"
        f"Remote: {row.get('remote', '')}\n"
        f"Adaptive: {row.get('adaptive', '')}\n"
        f"Description: {short_description}"
    )


def create_embeddings(
    data_path: Path = DATA_PATH,
    index_path: Path = INDEX_PATH,
    pickle_path: Path = PICKLE_PATH,
    model_name: str = MODEL_NAME,
    allow_download: bool = False,
) -> None:
    df = pd.read_json(data_path, lines=True)
    df["retrieval_text"] = df.apply(create_retrieval_text, axis=1)

    try:
        if allow_download:
            model = SentenceTransformer(model_name)
        else:
            model = SentenceTransformer(model_name, local_files_only=True)
    except Exception as exc:
        raise RuntimeError(
            "Embedding model is not available locally. Connect to the internet "
            "once to cache it, or call create_embeddings(..., allow_download=True)."
        ) from exc

    embeddings = model.encode(df["retrieval_text"].tolist(), show_progress_bar=True)
    embedding_matrix = np.asarray(embeddings, dtype="float32")

    dimension = embedding_matrix.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embedding_matrix)

    faiss.write_index(index, str(index_path))
    df.to_pickle(pickle_path)

    print(f"FAISS index saved to: {index_path}")
    print(f"Processed dataframe saved to: {pickle_path}")


if __name__ == "__main__":
    create_embeddings()
