import csv
import os
import re
from pathlib import Path
from typing import Any

from app.config import settings

REQUIRED_COLUMNS = [
    "category",
    "welfareType",
    "servId",
    "servNm",
    "agency",
    "department",
    "intrsThemaArray",
    "lifeArray",
    "srvPvsnNm",
    "sprtCycNm",
    "servDgst",
    "servDtlLink",
    "inqNum",
    "contact",
]
OPTIONAL_COLUMNS = ["source_csv"]
FIELD_WEIGHTS: dict[str, float] = {
    "servNm": 2.5,
    "lifeArray": 2.0,
    "servDgst": 2.4,
    "agency": 1.8,
    "welfareType": 1.1,
    "srvPvsnNm": 1.0,
    "intrsThemaArray": 0.8,
    "category": 0.6,
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
    "관련",
    "지역",
    "대상",
    "및",
    "또는",
    "에서",
    "으로",
}
REGION_ALIASES: dict[str, tuple[str, ...]] = {
    "서울": ("서울", "서울시", "서울특별시"),
    "부산": ("부산", "부산시", "부산광역시"),
    "대구": ("대구", "대구시", "대구광역시"),
    "인천": ("인천", "인천시", "인천광역시"),
    "광주": ("광주", "광주시", "광주광역시"),
    "대전": ("대전", "대전시", "대전광역시"),
    "울산": ("울산", "울산시", "울산광역시"),
    "세종": ("세종", "세종시", "세종특별자치시"),
    "경기": ("경기", "경기도"),
    "강원": ("강원", "강원도", "강원특별자치도"),
    "충북": ("충북", "충청북도"),
    "충남": ("충남", "충청남도"),
    "전북": ("전북", "전라북도", "전북특별자치도"),
    "전남": ("전남", "전라남도"),
    "경북": ("경북", "경상북도"),
    "경남": ("경남", "경상남도"),
    "제주": ("제주", "제주도", "제주특별자치도"),
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
        "교육비",
        "학비",
        "장학금",
    },
    "counseling": {
        "정서",
        "심리",
        "상담",
        "위축",
        "불안",
        "자존감",
        "스트레스",
        "개인상담",
        "집단상담",
        "위클래스",
        "wee",
        "정신건강",
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
        "지역아동센터",
        "아이돌봄",
    },
    "economic": {
        "교육비",
        "학비",
        "생활비",
        "경제",
        "생계",
        "급여",
        "바우처",
        "차상위",
        "저소득",
        "수급",
        "장학금",
    },
    "safety": {
        "안전",
        "위기",
        "긴급",
        "폭력",
        "학대",
        "가출",
        "보호",
        "건강",
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
    "교육비",
    "학비",
    "장학금",
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
    "발달장애",
}
SECONDARY_ONLY_KEYWORDS = {
    "중고생",
    "고등학생",
    "고등학교",
    "고1",
    "고2",
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
DOMAIN_TO_DOC_KEYWORDS = {
    "학업": {"교육", "학습", "기초학력", "학비", "교육비", "학교", "장학금"},
    "정서_심리": {"상담", "정서", "심리", "심리상담", "위클래스", "청소년특별지원"},
    "사회성": {"또래", "관계", "사회성", "집단상담", "청소년"},
    "돌봄": {"돌봄", "방과후", "아이돌봄", "지역아동센터", "보호"},
    "경제": {"교육비", "바우처", "지원금", "급여", "저소득", "장학금"},
    "위기": {"위기", "긴급복지", "폭력", "가출", "보호시설"},
    "장애_특수": {"장애", "특수교육", "발달장애", "치료지원"},
}
INSTITUTION_NAME_KEYWORDS = {
    "counseling": {"청소년상담복지센터", "상담복지센터", "상담센터", "wee", "위"},
    "care": {"지역아동센터", "돌봄센터", "다함께돌봄", "가족센터"},
    "academic": {"학습", "교육지원청", "진로", "진학", "클리닉"},
    "youth": {"청소년지원센터", "꿈드림", "청소년수련관", "청소년문화의집"},
}
LOCALIZED_WELFARE_TYPES = {
    "지자체(출자출연기관)",
    "지자체",
    "지역센터",
    "센터",
}
DEFAULT_MULTI_CSV_PATHS = [
    "integrated_institution_data.csv",
    "transformed_scholarships_detailed_dgst.csv",
    "welfare_integrated_data.csv",
]
MIN_RECOMMENDATION_SCORE = 0.12


def _validate_csv_headers(fieldnames: list[str] | None, csv_path: Path) -> None:
    if not fieldnames:
        raise ValueError(f"CSV header is missing: {csv_path}")

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
    if missing_columns:
        raise ValueError(f"CSV is missing required columns: {missing_columns} ({csv_path})")


def _resolve_csv_paths(csv_path: str | Path | list[str] | list[Path] | None) -> list[Path]:
    if csv_path is None:
        configured = settings.institutions_csv_path
        raw_candidates = [part.strip() for part in str(configured).split(",") if part.strip()]
        raw_paths = raw_candidates + DEFAULT_MULTI_CSV_PATHS
    elif isinstance(csv_path, (str, Path)):
        raw_paths = csv_path
    else:
        raw_paths = list(csv_path)

    if isinstance(raw_paths, Path):
        return [raw_paths]

    if isinstance(raw_paths, list):
        path_candidates = [str(part).strip() for part in raw_paths if str(part).strip()]
    else:
        path_candidates = [part.strip() for part in str(raw_paths).split(",") if part.strip()]

    paths = [Path(part) for part in dict.fromkeys(path_candidates)]
    if not paths:
        raise ValueError("No CSV paths configured for institution search.")
    return paths


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _build_page_content(metadata: dict[str, str]) -> str:
    return _normalize_whitespace(
        "\n".join(
            [
                f"category: {metadata.get('category', '')}",
                f"welfareType: {metadata.get('welfareType', '')}",
                f"servNm: {metadata.get('servNm', '')}",
                f"agency: {metadata.get('agency', '')}",
                f"department: {metadata.get('department', '')}",
                f"lifeArray: {metadata.get('lifeArray', '')}",
                f"servDgst: {metadata.get('servDgst', '')}",
                f"intrsThemaArray: {metadata.get('intrsThemaArray', '')}",
                f"srvPvsnNm: {metadata.get('srvPvsnNm', '')}",
                f"contact: {metadata.get('contact', '')}",
                f"servDtlLink: {metadata.get('servDtlLink', '')}",
                f"source_csv: {metadata.get('source_csv', '')}",
            ]
        )
    )


def _load_institution_documents(csv_paths: str | Path | list[str] | list[Path] | None) -> list[Any]:
    documents: list[dict[str, Any]] = []

    for path in _resolve_csv_paths(csv_paths):
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")

        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            _validate_csv_headers(reader.fieldnames, path)

            for row in reader:
                metadata = {column: row.get(column, "").strip() for column in REQUIRED_COLUMNS}
                metadata["source_csv"] = path.name
                page_content = _build_page_content(metadata)
                documents.append({"page_content": page_content, "metadata": metadata})

    if not documents:
        raise ValueError("No records found in CSV.")

    deduped_documents: dict[str, dict[str, Any]] = {}
    for doc in documents:
        metadata = doc["metadata"]
        dedupe_key = metadata.get("servId") or metadata.get("servNm") or doc["page_content"]
        existing = deduped_documents.get(dedupe_key)
        if not existing or len(doc["page_content"]) > len(existing["page_content"]):
            deduped_documents[dedupe_key] = doc

    return list(deduped_documents.values())


def _build_vectorstore(documents: list[Any]) -> Any:
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from langchain_google_genai._common import GoogleGenerativeAIError

    api_key = settings.gemini_api_key or os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set. Add it to .env before running RAG search.")

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
                Document(page_content=doc["page_content"], metadata=doc["metadata"])
                for doc in documents
            ]
            return FAISS.from_documents(lc_documents, embeddings)
        except GoogleGenerativeAIError as exc:
            last_error = exc
            continue

    raise RuntimeError(f"Failed to build embeddings with models: {tried_models}") from last_error


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


def _extract_region_tokens(region: str) -> tuple[list[str], list[str]]:
    cleaned = _normalize_whitespace(region)
    if not cleaned:
        return [], []

    province_tokens: list[str] = []
    district_tokens: list[str] = []

    for canonical, aliases in REGION_ALIASES.items():
        if any(alias in cleaned for alias in aliases):
            province_tokens.extend(list(aliases))
            province_tokens.append(canonical)

    district_tokens.extend(re.findall(r"[가-힣]+(?:시|군|구)", cleaned))
    district_tokens.extend(re.findall(r"[가-힣]+(?:동|읍|면)", cleaned))

    if not province_tokens and not district_tokens:
        district_tokens.extend(re.findall(r"[가-힣A-Za-z0-9]{2,}", cleaned))

    province_tokens = list(dict.fromkeys(token for token in province_tokens if token))
    district_tokens = list(dict.fromkeys(token for token in district_tokens if token))
    return province_tokens, district_tokens


def _document_region_text(metadata: dict[str, str]) -> str:
    return " ".join(
        [
            metadata.get("agency", ""),
            metadata.get("servNm", ""),
            metadata.get("servDgst", ""),
            metadata.get("lifeArray", ""),
            metadata.get("servDtlLink", ""),
        ]
    ).lower()


def _has_explicit_region(text: str) -> bool:
    aliases = {alias.lower() for values in REGION_ALIASES.values() for alias in values}
    return any(alias in text for alias in aliases) or bool(re.search(r"[가-힣]+(?:시|군|구)", text))


def _score_region_alignment(metadata: dict[str, str], context: dict[str, Any] | None) -> float:
    if not context:
        return 0.0

    region = str(context.get("student_region", "")).strip()
    if not region:
        return 0.0

    doc_region_text = _document_region_text(metadata)
    province_tokens, district_tokens = _extract_region_tokens(region)
    welfare_type = metadata.get("welfareType", "")
    is_localized_doc = (
        metadata.get("category", "") == "기관"
        or welfare_type in LOCALIZED_WELFARE_TYPES
        or "지자체" in welfare_type
        or _has_explicit_region(doc_region_text)
    )

    score = 0.0
    province_hits = sum(1 for token in province_tokens if token.lower() in doc_region_text)
    district_hits = sum(1 for token in district_tokens if token.lower() in doc_region_text)
    doc_has_specific_district = bool(re.search(r"[가-힣]+(?:시|군|구)", doc_region_text))

    if province_hits:
        score += min(0.16, province_hits * 0.06)
    if district_hits:
        score += min(0.24, district_hits * 0.12)
    if region.lower() in doc_region_text:
        score += 0.10
    if district_tokens and province_hits and not district_hits and doc_has_specific_district:
        score -= 0.24

    if is_localized_doc and (province_tokens or district_tokens) and not (province_hits or district_hits):
        score -= 0.28

    return max(-0.24, min(0.20, score))


def _score_institution_fit(
    metadata: dict[str, str],
    context: dict[str, Any] | None,
    domain_scores: dict[str, float] | None = None,
) -> float:
    if metadata.get("category") != "기관":
        return 0.0

    name_text = " ".join(
        [
            metadata.get("servNm", ""),
            metadata.get("agency", ""),
            metadata.get("department", ""),
        ]
    ).lower()
    student_text = str((context or {}).get("student_text", "")).lower()
    support_request = str((context or {}).get("support_request", "")).lower()

    score = 0.0

    if _contains_any(name_text, INSTITUTION_NAME_KEYWORDS["counseling"]) and _contains_any(
        f"{student_text} {support_request}",
        {"정서", "심리", "불안", "상담", "갈등", "충동"},
    ):
        score += 0.10

    if _contains_any(name_text, INSTITUTION_NAME_KEYWORDS["care"]) and _contains_any(
        f"{student_text} {support_request}",
        {"돌봄", "방과 후", "방과후", "보호자 부재", "공백"},
    ):
        score += 0.08

    if _contains_any(name_text, INSTITUTION_NAME_KEYWORDS["academic"]) and _contains_any(
        f"{student_text} {support_request}",
        {"학업", "학습", "기초학력", "수업", "집중"},
    ):
        score += 0.06

    if _contains_any(name_text, {"청소년상담복지센터", "상담복지센터"}) and domain_scores:
        score += min(0.10, float(domain_scores.get("정서_심리", 0.0)) * 0.10)
        score += min(0.05, float(domain_scores.get("사회성", 0.0)) * 0.08)

    if _contains_any(name_text, {"지역아동센터", "다함께돌봄", "돌봄센터", "가족센터"}) and domain_scores:
        score += min(0.08, float(domain_scores.get("돌봄", 0.0)) * 0.10)

    if "학교밖" in name_text and not _contains_any(
        f"{student_text} {support_request}",
        {"학교밖", "학업 중단", "장기 결석", "자퇴", "검정고시"},
    ):
        score -= 0.16

    score += max(0.0, _score_region_alignment(metadata, context)) * 0.5
    return max(-0.12, min(0.24, score))


def _score_context_alignment(
    metadata: dict[str, str],
    context: dict[str, Any] | None,
    domain_scores: dict[str, float] | None = None,
) -> float:
    if not context:
        return 0.0

    doc_text = _build_page_content(metadata).lower()
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
        score += student_strength * doc_strength * 0.14

    support_terms = _tokenize_query(support_request)
    if support_terms:
        overlap = sum(1 for term in support_terms if term in doc_text) / len(support_terms)
        score += overlap * 0.24

    observation_terms = _tokenize_query(observation_text)
    if observation_terms:
        overlap = sum(1 for term in observation_terms if term in doc_text) / len(observation_terms)
        score += overlap * 0.18

    reason_terms = _tokenize_query(application_reason)
    if reason_terms:
        overlap = sum(1 for term in reason_terms if term in doc_text) / len(reason_terms)
        score += overlap * 0.10

    doc_name = metadata.get("servNm", "").strip().lower()
    for target_name, trigger_keywords in DIRECT_NAME_BOOST_PATTERNS.items():
        if doc_name != target_name:
            continue
        if _contains_any(student_text, trigger_keywords) or _contains_any(support_request, trigger_keywords):
            score += 0.18

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

    if _contains_any(student_text, {"스마트폰", "동영상", "인터넷"}) and _contains_any(
        doc_text, {"스마트폰", "인터넷", "과의존"}
    ):
        score += 0.10

    if _contains_any(student_text, {"혼자", "맞벌이", "방과 후", "돌봄"}) and _contains_any(
        doc_text, {"돌봄", "방과 후", "방과후", "지역아동센터"}
    ):
        score += 0.08

    if domain_scores:
        dynamic_score = 0.0
        for domain, urgency in domain_scores.items():
            if domain == "분석근거" or not isinstance(urgency, (float, int)):
                continue
            if urgency < 0.3:
                continue
            keywords = DOMAIN_TO_DOC_KEYWORDS.get(domain, set())
            if _contains_any(doc_text, keywords):
                dynamic_score += float(urgency) * 0.15
        score += min(0.45, dynamic_score)

    return max(-0.55, min(0.24, score))


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
    target_plus_content = f"{metadata.get('lifeArray', '')} {metadata.get('servDgst', '')}".lower()
    bonus_terms = {
        "정서",
        "심리",
        "상담",
        "학습",
        "기초학력",
        "돌봄",
        "안전",
        "폭력",
        "교육비",
        "학비",
        "장학금",
        "가정",
    }
    bonus_hits = sum(1 for term in query_terms if term in bonus_terms and term in target_plus_content)
    boosted = base + min(0.22, bonus_hits * 0.03)
    return min(1.0, boosted)


def _rank_documents(
    documents: list[Any],
    query: str,
    top_k: int,
    context: dict[str, Any] | None = None,
    vector_scores: dict[str, float] | None = None,
    domain_scores: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    query_terms = _tokenize_query(query)
    vector_scores = vector_scores or {}
    scored_docs: list[dict[str, Any]] = []

    for doc in documents:
        metadata = doc["metadata"]
        weighted_score = _weighted_row_match_score(query_terms, metadata)
        text_score = _score_text_match(query_terms, doc["page_content"])
        lexical_score = (weighted_score * 0.48) + (text_score * 0.12)
        context_score = _score_context_alignment(metadata, context, domain_scores=domain_scores)
        region_score = _score_region_alignment(metadata, context)
        institution_score = _score_institution_fit(metadata, context, domain_scores=domain_scores)

        vector_key = metadata.get("servId") or metadata.get("servNm", "")
        vector_score = vector_scores.get(vector_key, 0.0) * 0.18
        score = max(
            0.0,
            min(1.0, lexical_score + context_score + region_score + institution_score + vector_score),
        )

        scored_docs.append(
            {
                "distance": round(1.0 - score, 6),
                "relevance_score": round(score, 6),
                **metadata,
            }
        )

    scored_docs.sort(
        key=lambda item: (
            item["relevance_score"],
            item.get("category") == "기관",
            item.get("inqNum", 0),
        ),
        reverse=True,
    )
    filtered_docs = [item for item in scored_docs if item["relevance_score"] >= MIN_RECOMMENDATION_SCORE]
    return (filtered_docs or scored_docs)[:top_k]


def _fallback_similarity_search(
    documents: list[Any],
    query: str,
    top_k: int,
    context: dict[str, Any] | None = None,
    domain_scores: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    return _rank_documents(
        documents=documents,
        query=query,
        top_k=top_k,
        context=context,
        domain_scores=domain_scores,
    )


def search_relevant_institutions(
    query: str,
    top_k: int = 100,
    csv_path: str | Path | list[str] | list[Path] | None = None,
    context: dict[str, Any] | None = None,
    domain_scores: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """
    Search relevant institution/policy rows from CSVs using
    Gemini embeddings + LangChain FAISS (in-memory) or a deterministic fallback.
    """
    if not query.strip():
        raise ValueError("Query text is empty.")

    # ---------------- Vercel 환경 경로 수정 시작 ----------------
    target_csv_path = str(csv_path or settings.institutions_csv_path)
    
    # 경로가 절대 경로가 아니라면, 프로젝트 최상위 폴더(EASY) 기준으로 찾도록 설정
    if not os.path.isabs(target_csv_path):
        # app/rag.py 기준 2단계 위(EASY 폴더)를 찾습니다.
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        target_csv_path = os.path.join(base_dir, target_csv_path)
    # ---------------- Vercel 환경 경로 수정 끝 ----------------

    # 수정된 경로를 기반으로 문서 로드
    documents = _load_institution_documents(target_csv_path)
    
    # 임베딩 방식에 따른 분기 처리 (새로 추가된 부분)
    if not settings.use_gemini_embeddings:
        return _fallback_similarity_search(
            documents,
            query=query,
            top_k=top_k,
            context=context,
            domain_scores=domain_scores,   # ← 전달
        )

    try:
        vectorstore = _build_vectorstore(documents)
        results = vectorstore.similarity_search_with_score(
            query,
            k=min(len(documents), max(top_k * 4, 32)),
        )
        vector_scores: dict[str, float] = {}
        for doc, distance in results:
            relevance_score = 1 / (1 + float(distance))
            metadata = doc.metadata
            key = str(metadata.get("servId") or metadata.get("servNm", "")).strip()
            if not key:
                continue
            vector_scores[key] = max(vector_scores.get(key, 0.0), relevance_score)
        return _rank_documents(
            documents=documents,
            query=query,
            top_k=top_k,
            context=context,
            vector_scores=vector_scores,
        )
    except Exception:
        return _fallback_similarity_search(
            documents,
            query=query,
            top_k=top_k,
            context=context,
        )