import csv
import os
import re
from pathlib import Path
from typing import Any

from app.config import settings

REQUIRED_COLUMNS = ["유형", "이름", "지원대상", "지원내용", "신청절차", "필요서류", "문의처"]
OPTIONAL_COLUMNS = ["링크"]
FIELD_WEIGHTS: dict[str, float] = {
    "이름": 2.4,
    "지원대상": 2.0,
    "지원내용": 2.2,
    "신청절차": 1.0,
    "필요서류": 0.8,
    "유형": 0.6,
}
KOREAN_STOPWORDS = {
    "학생",
    "지원",
    "연계",
    "필요",
    "요청",
    "학교",
    "기관",
    "제도",
    "대한",
    "및",
    "또는",
    "에서",
    "으로",
}
DOMAIN_KEYWORDS: dict[str, set[str]] = {
    "academic": {
        "학업",
        "학습",
        "기초학력",
        "기초",
        "수학",
        "국어",
        "학습부진",
        "숙제",
        "집중",
        "수업",
        "무기력",
        "자신감",
        "코칭",
    },
    "counseling": {
        "정서",
        "심리",
        "상담",
        "위축",
        "방어적",
        "불안",
        "자존감",
        "스트레스",
        "개인상담",
        "집단상담",
        "위클래스",
        "wee",
    },
    "social": {
        "또래",
        "갈등",
        "사회성",
        "충동",
        "짜증",
        "분노",
        "관계",
        "협동",
        "학교폭력",
        "친구",
    },
    "care": {
        "돌봄",
        "방과후",
        "방과",
        "혼자",
        "맞벌이",
        "귀가",
        "공백",
        "보호",
        "학원",
        "지역아동센터",
    },
    "digital": {
        "스마트폰",
        "인터넷",
        "동영상",
        "과의존",
        "중독",
        "게임",
    },
}
LOW_INCOME_KEYWORDS = {
    "저소득",
    "기초생활수급자",
    "수급자",
    "차상위",
    "한부모",
    "취약계층",
    "의료급여",
    "복지",
    "급여",
    "바우처",
}
HIGH_RISK_ONLY_KEYWORDS = {
    "자해",
    "우울증",
    "정신건강",
    "성폭력",
    "가정폭력",
    "아동학대",
    "가출",
    "비행",
    "도박",
    "학업 중단",
    "장기 결석",
    "위기학생",
}
DISABILITY_KEYWORDS = {
    "특수교육",
    "장애",
    "자폐",
    "지적장애",
    "시청각장애",
    "경계선지능",
}
SECONDARY_ONLY_KEYWORDS = {
    "중고생",
    "고등학생",
    "고등학교",
    "고3",
    "직업계고",
    "마이스터고",
    "대학생",
}
ELEMENTARY_HINT_KEYWORDS = {
    "초등",
    "초중",
    "아동",
    "재학생",
}
DIRECT_NAME_BOOST_PATTERNS = {
    "학교 wee클래스": {"wee", "위클래스", "상담", "개인상담", "집단상담"},
    "학교 기초학력 디딤돌 교실": {"기초학력", "디딤돌", "학습", "국어", "수학"},
    "학습종합클리닉센터": {"학습", "기초학력", "집중", "학습부진", "코칭"},
    "스마트쉼센터": {"스마트폰", "인터넷", "동영상", "과의존"},
}
MIN_RECOMMENDATION_SCORE = 0.34


def _validate_csv_headers(fieldnames: list[str] | None) -> None:
    if not fieldnames:
        raise ValueError("CSV header is missing.")

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
    if missing_columns:
        raise ValueError(f"CSV is missing required columns: {missing_columns}")


def _load_institution_documents(csv_path: str | Path) -> list[Any]:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    documents: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        _validate_csv_headers(reader.fieldnames)

        for row in reader:
            metadata = {column: row.get(column, "").strip() for column in REQUIRED_COLUMNS}
            for column in OPTIONAL_COLUMNS:
                metadata[column] = row.get(column, "").strip()
            page_content = (
                f"유형: {metadata['유형']}\n"
                f"이름: {metadata['이름']}\n"
                f"지원대상: {metadata['지원대상']}\n"
                f"지원내용: {metadata['지원내용']}\n"
                f"신청절차: {metadata['신청절차']}\n"
                f"필요서류: {metadata['필요서류']}\n"
                f"문의처: {metadata['문의처']}\n"
                f"링크: {metadata['링크']}"
            )
            documents.append({"page_content": page_content, "metadata": metadata})

    if not documents:
        raise ValueError("No records found in CSV.")

    return documents


def _build_vectorstore(documents: list[Any]) -> Any:
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from langchain_google_genai._common import GoogleGenerativeAIError

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
            lc_documents = [
                Document(
                    page_content=doc["page_content"],
                    metadata=doc["metadata"],
                )
                for doc in documents
            ]
            return FAISS.from_documents(lc_documents, embeddings)
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


def _tokenize_query(query: str) -> list[str]:
    raw_terms = re.findall(r"[가-힣A-Za-z0-9]{2,}", query.lower())
    deduped: list[str] = []
    for term in raw_terms:
        if term in KOREAN_STOPWORDS:
            continue
        if term not in deduped:
            deduped.append(term)
    return deduped


def _contains_any(text: str, keywords: set[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def _keyword_hits(text: str, keywords: set[str]) -> int:
    return sum(1 for keyword in keywords if keyword in text)


def _domain_profile_scores(text: str) -> dict[str, float]:
    profile_scores: dict[str, float] = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        hits = _keyword_hits(text, keywords)
        profile_scores[domain] = min(1.0, hits / 3)
    return profile_scores


def _score_context_alignment(
    metadata: dict[str, str],
    context: dict[str, Any] | None,
) -> float:
    if not context:
        return 0.0

    doc_text = " ".join(
        metadata.get(field, "")
        for field in REQUIRED_COLUMNS + OPTIONAL_COLUMNS
    ).lower()
    student_text = str(context.get("student_text", "")).lower()
    support_request = str(context.get("support_request", "")).lower()
    observation_text = str(context.get("observation_text", "")).lower()
    application_reason = str(context.get("application_reason", "")).lower()
    economy_text = " ".join(
        [
            str(context.get("basic_living_security_status", "")),
            str(context.get("student_basic_info", "")),
            str(context.get("economy_life", "")),
            str(context.get("family_status", "")),
        ]
    ).lower()

    score = 0.0
    student_domains = _domain_profile_scores(student_text)
    doc_domains = _domain_profile_scores(doc_text)
    for domain, student_strength in student_domains.items():
        if student_strength <= 0:
            continue
        doc_strength = doc_domains.get(domain, 0.0)
        score += student_strength * doc_strength * 0.12

    support_terms = _tokenize_query(support_request)
    if support_terms:
        overlap = sum(1 for term in support_terms if term in doc_text) / len(support_terms)
        score += overlap * 0.22

    observation_terms = _tokenize_query(observation_text)
    if observation_terms:
        overlap = sum(1 for term in observation_terms if term in doc_text) / len(observation_terms)
        score += overlap * 0.14

    reason_terms = _tokenize_query(application_reason)
    if reason_terms:
        overlap = sum(1 for term in reason_terms if term in doc_text) / len(reason_terms)
        score += overlap * 0.10

    doc_name = metadata.get("이름", "").strip().lower()
    for target_name, trigger_keywords in DIRECT_NAME_BOOST_PATTERNS.items():
        if doc_name != target_name:
            continue
        if _contains_any(student_text, trigger_keywords) or _contains_any(support_request, trigger_keywords):
            score += 0.18

    # School-requested supports should outrank outside/community supports when they match.
    if _contains_any(support_request, {"wee", "위클래스", "기초학력", "디딤돌"}):
        if _contains_any(doc_text, {"wee", "위클래스", "기초학력", "디딤돌"}):
            score += 0.12

    grade = int(context.get("student_grade", 0) or 0)
    if grade and grade <= 6:
        if _contains_any(doc_text, SECONDARY_ONLY_KEYWORDS):
            score -= 0.30
        if _contains_any(doc_text, ELEMENTARY_HINT_KEYWORDS):
            score += 0.05

    if (
        ("해당사항없음" in economy_text or "일반" in economy_text)
        and not _contains_any(student_text, {"저소득", "생계", "경제적 어려움", "수급", "차상위"})
        and _contains_any(doc_text, LOW_INCOME_KEYWORDS)
    ):
        score -= 0.28

    if not _contains_any(student_text, {"자해", "우울", "학대", "가출", "비행", "도박", "장기 결석", "학업 중단"}):
        if _contains_any(doc_text, HIGH_RISK_ONLY_KEYWORDS):
            score -= 0.24

    if not _contains_any(student_text, DISABILITY_KEYWORDS) and _contains_any(doc_text, DISABILITY_KEYWORDS):
        score -= 0.26

    if _contains_any(student_text, {"스마트폰", "동영상", "인터넷"}) and _contains_any(doc_text, {"스마트폰", "인터넷", "과의존"}):
        score += 0.10

    if _contains_any(student_text, {"혼자", "맞벌이", "방과 후", "돌봄"}) and _contains_any(
        doc_text,
        {"돌봄", "방과 후", "방과후", "지역아동센터"},
    ):
        score += 0.06

    return max(-0.6, min(0.6, score))


def _rank_documents(
    documents: list[Any],
    query: str,
    top_k: int,
    context: dict[str, Any] | None = None,
    vector_scores: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    query_terms = _tokenize_query(query)
    vector_scores = vector_scores or {}
    scored_docs: list[dict[str, Any]] = []

    for doc in documents:
        metadata = doc["metadata"]
        weighted_score = _weighted_row_match_score(query_terms, metadata)
        text_score = _score_text_match(query_terms, doc["page_content"])
        lexical_score = (weighted_score * 0.55) + (text_score * 0.15)
        context_score = _score_context_alignment(metadata, context)
        vector_score = vector_scores.get(metadata.get("이름", ""), 0.0) * 0.20
        score = max(0.0, min(1.0, lexical_score + context_score + vector_score))
        scored_docs.append(
            {
                "distance": round(1.0 - score, 6),
                "relevance_score": round(score, 6),
                **metadata,
            }
        )

    scored_docs.sort(key=lambda item: item["relevance_score"], reverse=True)
    filtered_docs = [
        item for item in scored_docs if item["relevance_score"] >= MIN_RECOMMENDATION_SCORE
    ]
    return (filtered_docs or scored_docs)[:top_k]


def _weighted_row_match_score(query_terms: list[str], metadata: dict[str, str]) -> float:
    if not query_terms:
        return 0.0

    weighted_score = 0.0
    total_weight = 0.0

    for field, weight in FIELD_WEIGHTS.items():
        field_text = metadata.get(field, "").lower()
        if not field_text:
            total_weight += weight
            continue
        hit_count = sum(1 for term in query_terms if term in field_text)
        field_score = hit_count / len(query_terms)
        weighted_score += field_score * weight
        total_weight += weight

    base = (weighted_score / total_weight) if total_weight else 0.0

    # Add small boosts for highly indicative terms in target/content.
    target_plus_content = (
        f"{metadata.get('지원대상', '')} {metadata.get('지원내용', '')}".lower()
    )
    bonus_terms = {
        "정서",
        "심리",
        "상담",
        "학습",
        "기초학력",
        "돌봄",
        "안전",
        "폭력",
        "중독",
        "인터넷",
        "스마트폰",
        "가정",
    }
    bonus_hits = sum(1 for term in query_terms if term in bonus_terms and term in target_plus_content)
    boosted = base + min(0.2, bonus_hits * 0.03)
    return min(1.0, boosted)


def _fallback_similarity_search(
    documents: list[Any],
    query: str,
    top_k: int,
    context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    return _rank_documents(
        documents=documents,
        query=query,
        top_k=top_k,
        context=context,
    )


def search_relevant_institutions(
    query: str,
    top_k: int = 3,
    csv_path: str | Path | None = None,
    context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Search relevant institution/policy rows from CSV using
    Gemini embeddings + LangChain FAISS (in-memory).
    """
    if not query.strip():
        raise ValueError("Query text is empty.")

    target_csv_path = csv_path or settings.institutions_csv_path
    documents = _load_institution_documents(target_csv_path)
    if not settings.use_gemini_embeddings:
        # Free-tier friendly mode: skip embedding API calls and use keyword matching.
        return _fallback_similarity_search(
            documents,
            query=query,
            top_k=top_k,
            context=context,
        )

    try:
        vectorstore = _build_vectorstore(documents)
        results = vectorstore.similarity_search_with_score(query, k=min(len(documents), max(top_k * 4, 8)))
        vector_scores: dict[str, float] = {}
        for doc, distance in results:
            # FAISS distance is lower-is-better. Convert to easy-to-read score as well.
            relevance_score = 1 / (1 + float(distance))
            name = str(doc.metadata.get("이름", "")).strip()
            if not name:
                continue
            vector_scores[name] = max(vector_scores.get(name, 0.0), relevance_score)
        return _rank_documents(
            documents=documents,
            query=query,
            top_k=top_k,
            context=context,
            vector_scores=vector_scores,
        )
    except Exception:
        # When free-tier quota is exhausted, provide deterministic keyword fallback.
        return _fallback_similarity_search(
            documents,
            query=query,
            top_k=top_k,
            context=context,
        )
