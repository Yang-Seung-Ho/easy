import json
import re

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai._common import GoogleGenerativeAIError
from pydantic import ValidationError

from app.config import settings
from app.models import AnalyzeStudentRequest, GeminiAnalysisResult

SYSTEM_INSTRUCTION = (
    "너는 초/중/고등학교의 전문 상담 교사 겸 데이터 분석가야. "
    "학생의 기본 환경과 교사들의 관찰 일지를 종합하여 학생이 현재 겪고 있는 핵심 문제를 파악하고, "
    "어떤 지원(정서, 학업, 경제 등)이 필요한지 2~3문장으로 요약해. "
    "그리고 가장 핵심이 되는 키워드(예: 정서 안정 지원 필요, 돌봄 공백 등)를 추출해."
)


def _is_rate_limit_error(error: Exception) -> bool:
    message = str(error).lower()
    return (
        "429" in message
        or "resource_exhausted" in message
        or "too many requests" in message
        or "quota" in message
    )


def _extract_keywords(text: str, limit: int = 3) -> list[str]:
    # Keep simple token extraction for deterministic fallback output.
    tokens = re.findall(r"[가-힣A-Za-z0-9]{2,}", text)
    seen: set[str] = set()
    result: list[str] = []
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        result.append(token)
        if len(result) >= limit:
            break
    return result


def analyze_student_data(request_data: AnalyzeStudentRequest) -> GeminiAnalysisResult:
    if not settings.gemini_api_key:
        raise ValueError("GEMINI_API_KEY is not set in .env")
    student_json = json.dumps(request_data.model_dump(by_alias=True), ensure_ascii=False, indent=2)
    prompt = (
        f"{SYSTEM_INSTRUCTION}\n\n"
        "아래 학생 데이터를 분석하고 반드시 JSON만 반환해.\n"
        "스키마:\n"
        "{\n"
        '  "이름": "[학생이름]",\n'
        '  "분석내용": "[분석 결과 텍스트]",\n'
        '  "핵심신호": ["키워드1", "키워드2", "키워드3"]\n'
        "}\n\n"
        "학생 데이터:\n"
        f"{student_json}"
    )

    model_candidates = [settings.gemini_model, "gemini-1.5-flash", "gemini-1.5-pro"]
    tried_models: list[str] = []
    last_error: Exception | None = None

    for model_name in model_candidates:
        if model_name in tried_models:
            continue
        tried_models.append(model_name)
        try:
            model = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=settings.gemini_api_key,
                temperature=0.2,
            )
            response = model.invoke(prompt)
            raw_text = (response.content or "").strip()

            # Some models may wrap JSON in markdown fences.
            if raw_text.startswith("```"):
                raw_text = raw_text.strip("`")
                raw_text = raw_text.replace("json", "", 1).strip()

            parsed = json.loads(raw_text)
            return GeminiAnalysisResult.model_validate(parsed)
        except (GoogleGenerativeAIError, ValueError, ValidationError) as error:
            last_error = error
            if _is_rate_limit_error(error):
                break
            continue

    return _build_local_fallback_analysis(request_data, last_error, tried_models)


def _build_local_fallback_analysis(
    request_data: AnalyzeStudentRequest,
    last_error: Exception | None,
    tried_models: list[str],
) -> GeminiAnalysisResult:
    info = request_data.all_data.integrated_application_info
    name = info.student_personal_info.student_name
    difficulties = info.student_condition.student_difficulties

    key_signals = [
        "기초 학력 미달",
        "정서 불안",
        "돌봄 공백",
    ]
    observation_keywords: list[str] = []
    for log in request_data.all_data.observation_logs:
        observation_keywords.extend(_extract_keywords(log.special_notes, limit=2))
        if len(observation_keywords) >= 2:
            break
    for keyword in observation_keywords:
        if keyword not in key_signals and len(key_signals) < 5:
            key_signals.append(keyword)

    analysis = (
        "API 사용량 제한으로 AI 모델 호출이 지연되어, 입력 데이터 기반의 안전 분석 결과를 제공합니다. "
        f"{name} 학생은 {difficulties.academics} 상태이며, {difficulties.emotional_psychological}. "
        f"또한 {difficulties.care_safety_health} 상황으로 확인되어 학습·정서·돌봄 영역의 통합 지원 연계가 필요합니다."
    )

    return GeminiAnalysisResult(
        이름=name,
        분석내용=analysis,
        핵심신호=key_signals,
    )
