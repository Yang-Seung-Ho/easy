import csv
import os
import re
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai._common import GoogleGenerativeAIError

from app.config import settings

REQUIRED_COLUMNS = ["유형", "이름", "지원대상", "지원내용", "신청절차", "필요서류", "문의처"]


def _validate_csv_headers(fieldnames: list[str] | None) -> None:
    if not fieldnames:
        raise ValueError("CSV header is missing.")

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
    if missing_columns:
        raise ValueError(f"CSV is missing required columns: {missing_columns}")


def _load_institution_documents(csv_path: str | Path) -> list[Document]:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    documents: list[Document] = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        _validate_csv_headers(reader.fieldnames)

        for row in reader:
            metadata = {column: row.get(column, "").strip() for column in REQUIRED_COLUMNS}
            page_content = (
                f"유형: {metadata['유형']}\n"
                f"이름: {metadata['이름']}\n"
                f"지원대상: {metadata['지원대상']}\n"
                f"지원내용: {metadata['지원내용']}\n"
                f"신청절차: {metadata['신청절차']}\n"
                f"필요서류: {metadata['필요서류']}\n"
                f"문의처: {metadata['문의처']}"
            )
            documents.append(Document(page_content=page_content, metadata=metadata))

    if not documents:
        raise ValueError("No records found in CSV.")

    return documents


def _build_vectorstore(documents: list[Document]) -> FAISS:
    api_key = settings.gemini_api_key or os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is not set. Add it to .env before running RAG search."
        )
    model_candidates = [
        settings.gemini_embedding_model,
        "models/gemini-embedding-001",
        "models/gemini-embedding-2-preview",
    ]
    tried_models: list[str] = []
    last_error: Exception | None = None

    for model_name in model_candidates:
        if model_name in tried_models:
            continue
        tried_models.append(model_name)
        try:
            embeddings = GoogleGenerativeAIEmbeddings(
                model=model_name,
                google_api_key=api_key,
            )
            return FAISS.from_documents(documents, embeddings)
        except GoogleGenerativeAIError as exc:
            last_error = exc
            continue

    raise RuntimeError(
        f"Failed to build embeddings with models: {tried_models}"
    ) from last_error


def _score_text_match(query_terms: list[str], text: str) -> float:
    text_lower = text.lower()
    unique_terms = {term for term in query_terms if term}
    if not unique_terms:
        return 0.0
    matches = sum(1 for term in unique_terms if term in text_lower)
    return matches / len(unique_terms)


def _fallback_similarity_search(
    documents: list[Document],
    query: str,
    top_k: int,
) -> list[dict[str, Any]]:
    query_terms = re.findall(r"[가-힣A-Za-z0-9]{2,}", query.lower())
    scored_docs: list[tuple[Document, float]] = []

    for doc in documents:
        score = _score_text_match(query_terms, doc.page_content)
        scored_docs.append((doc, score))

    scored_docs.sort(key=lambda item: item[1], reverse=True)
    top_docs = scored_docs[:top_k]

    formatted: list[dict[str, Any]] = []
    for doc, score in top_docs:
        distance = round(1.0 - score, 6)
        formatted.append(
            {
                "distance": distance,
                "relevance_score": round(score if score > 0 else 0.2, 6),
                **doc.metadata,
            }
        )
    return formatted


def search_relevant_institutions(
    query: str,
    top_k: int = 3,
    csv_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """
    Search relevant institution/policy rows from CSV using
    Gemini embeddings + LangChain FAISS (in-memory).
    """
    if not query.strip():
        raise ValueError("Query text is empty.")

    target_csv_path = csv_path or settings.institutions_csv_path
    documents = _load_institution_documents(target_csv_path)
    try:
        vectorstore = _build_vectorstore(documents)
        results = vectorstore.similarity_search_with_score(query, k=top_k)
        formatted: list[dict[str, Any]] = []
        for doc, distance in results:
            # FAISS distance is lower-is-better. Convert to easy-to-read score as well.
            relevance_score = 1 / (1 + float(distance))
            formatted.append(
                {
                    "distance": float(distance),
                    "relevance_score": round(relevance_score, 6),
                    **doc.metadata,
                }
            )
        return formatted
    except Exception:
        # When free-tier quota is exhausted, provide deterministic keyword fallback.
        return _fallback_similarity_search(documents, query=query, top_k=top_k)
