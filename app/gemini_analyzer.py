import json
import re
import ssl
import time
from typing import Any
from urllib.parse import quote
from urllib import error as urlerror
from urllib import request as urlrequest

import certifi
from pydantic import ValidationError

from app.config import settings
from app.models import AnalyzeStudentRequest, GeminiAnalysisResult

SYSTEM_INSTRUCTION = (
    "너는 초/중/고등학교의 전문 상담 교사 겸 데이터 분석가야. "
    "학생의 기본 환경과 교사들의 관찰 일지를 종합하여 학생이 현재 겪고 있는 핵심 문제를 파악하고, "
    "어떤 지원(정서, 학업, 경제 등)이 필요한지 2~3문장으로 요약해. "
    "그리고 가장 핵심이 되는 키워드(예: 정서 안정 지원 필요, 돌봄 공백 등)를 추출해. "
    "반드시 JSON만 반환하고, 설명 문장은 쓰지 마."
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


def _extract_json_object(raw_text: str) -> dict:
    cleaned = (raw_text or "").strip()
    if not cleaned:
        raise ValueError("Empty response from Gemini.")

    # Prefer fenced JSON body if present.
    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if fenced_match:
        return json.loads(fenced_match.group(1))

    # Fallback: find first JSON object-like block.
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        return json.loads(cleaned[start : end + 1])

    raise ValueError("Could not find JSON object in Gemini response.")


def _response_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts).strip()
    return str(content).strip()


def _normalize_model_name(model_name: str) -> str:
    normalized = (model_name or "").strip()
    if not normalized:
        return "gemini-2.5-flash"

    aliases = {
        "gemini 3.1 flash lite": "gemini-2.5-flash-lite",
        "gemini 2.5 flash": "gemini-2.5-flash",
        "gemini 2.0 flash": "gemini-2.0-flash",
        "gemini 2.0 flash lite": "gemini-2.0-flash-lite",
    }
    key = normalized.lower()
    if key in aliases:
        return aliases[key]
    return normalized


def _gemini_generate_content(prompt: str, model_name: str, api_key: str) -> dict[str, Any]:
    safe_model_name = _normalize_model_name(model_name)
    encoded_model_name = quote(safe_model_name, safe="")
    endpoint = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{encoded_model_name}:generateContent?key={api_key}"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "responseMimeType": "application/json",
        },
    }
    data = json.dumps(payload).encode("utf-8")
    req = urlrequest.Request(
        endpoint,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    try:
        with urlrequest.urlopen(req, timeout=30, context=ssl_context) as response:
            body = response.read().decode("utf-8")
            return json.loads(body)
    except urlerror.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Gemini HTTP {exc.code}: {body}") from exc
    except urlerror.URLError as exc:
        raise RuntimeError(f"Gemini network error: {exc.reason}") from exc


def _extract_text_from_gemini_response(response_body: dict[str, Any]) -> str:
    candidates = response_body.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("Gemini returned no candidates.")
    content = candidates[0].get("content", {})
    parts = content.get("parts", [])
    return _response_to_text(parts)


def _normalize_gemini_payload(raw_payload: dict, fallback_name: str) -> dict:
    name = (
        raw_payload.get("이름")
        or raw_payload.get("name")
        or raw_payload.get("학생이름")
        or fallback_name
    )
    analysis = (
        raw_payload.get("분석내용")
        or raw_payload.get("요약분석")
        or raw_payload.get("analysis")
        or raw_payload.get("summary")
        or ""
    )
    key_signals = (
        raw_payload.get("핵심신호")
        or raw_payload.get("key_signals")
        or raw_payload.get("키워드")
        or []
    )

    if isinstance(key_signals, str):
        key_signals = [part.strip() for part in re.split(r"[,/|]", key_signals) if part.strip()]
    if not isinstance(key_signals, list):
        key_signals = []

    normalized_signals: list[str] = []
    for signal in key_signals:
        if not isinstance(signal, str):
            continue
        token = signal.strip()
        if token and token not in normalized_signals:
            normalized_signals.append(token)
        if len(normalized_signals) >= 5:
            break

    if len(normalized_signals) < 3:
        normalized_signals.extend(
            keyword
            for keyword in ["기초 학력 미달", "정서 불안", "돌봄 공백"]
            if keyword not in normalized_signals
        )

    return {
        "이름": str(name).strip() or fallback_name,
        "분석내용": str(analysis).strip(),
        "핵심신호": normalized_signals[:5],
    }


def _build_compact_student_context(request_data: AnalyzeStudentRequest) -> str:
    info = request_data.all_data.integrated_application_info
    personal = info.student_personal_info
    difficulties = info.student_condition.student_difficulties

    observation_lines: list[str] = []
    for log in request_data.all_data.observation_logs[:3]:
        observation_lines.append(
            f"- {log.date} {log.time} {log.place}: {log.content} / 특이사항: {log.special_notes}"
        )

    return (
        f"학생이름: {personal.student_name}\n"
        f"학년/반: {personal.grade}학년 {personal.class_number}반\n"
        f"가정환경: {info.home_environment_and_eligibility.student_basic_info}, "
        f"{info.home_environment_and_eligibility.family_status}\n"
        f"지원요청: {info.support_request}\n"
        f"학생현황: {info.student_condition.student_status}\n"
        f"학생어려움(학업): {difficulties.academics}\n"
        f"학생어려움(심리정서): {difficulties.emotional_psychological}\n"
        f"학생어려움(돌봄안전건강): {difficulties.care_safety_health}\n"
        f"관찰일지:\n" + "\n".join(observation_lines)
    )


def analyze_student_data(request_data: AnalyzeStudentRequest) -> GeminiAnalysisResult:
    if not settings.gemini_api_key:
        raise ValueError("GEMINI_API_KEY is not set in .env")
    compact_context = _build_compact_student_context(request_data)
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
        f"{compact_context}"
    )

    model_candidates = [
        settings.gemini_model,
        "gemini-2.0-flash-lite",
        "gemini-2.0-flash",
    ]
    tried_models: list[str] = []
    last_error: Exception | None = None
    had_rate_limit = False

    for model_name in model_candidates:
        if model_name in tried_models:
            continue
        tried_models.append(model_name)
        try:
            max_attempts = max(1, settings.gemini_max_retries + 1)
            for attempt in range(max_attempts):
                try:
                    response_body = _gemini_generate_content(
                        prompt=prompt,
                        model_name=model_name,
                        api_key=settings.gemini_api_key,
                    )
                    raw_text = _extract_text_from_gemini_response(response_body)
                    parsed = _extract_json_object(raw_text)
                    normalized = _normalize_gemini_payload(
                        parsed,
                        fallback_name=request_data.all_data.integrated_application_info.student_personal_info.student_name,
                    )
                    return GeminiAnalysisResult.model_validate(normalized)
                except (ValueError, ValidationError) as error:
                    last_error = error
                    if attempt == max_attempts - 1:
                        raise
                except RuntimeError as error:
                    last_error = error
                    had_rate_limit = had_rate_limit or _is_rate_limit_error(error)
                    if _is_rate_limit_error(error) and attempt < max_attempts - 1:
                        time.sleep(1 + attempt)
                        continue
                    raise
        except (RuntimeError, ValueError, ValidationError) as error:
            last_error = error
            had_rate_limit = had_rate_limit or _is_rate_limit_error(error)
            continue

    if not settings.allow_local_fallback:
        if had_rate_limit:
            raise RuntimeError(
                "Gemini quota exceeded. API-generated analysis is unavailable until quota resets "
                "or billing is enabled."
            ) from last_error
        raise RuntimeError(
            "Gemini analysis failed (model unavailable or invalid response). "
            f"Last error: {last_error}. "
            "Set ALLOW_LOCAL_FALLBACK=true only if you want deterministic local fallback."
        ) from last_error

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
        f"{name} 학생은 {difficulties.academics} 상태이며, {difficulties.emotional_psychological}. "
        f"또한 {difficulties.care_safety_health} 상황으로 확인되어 학습·정서·돌봄 영역의 통합 지원 연계가 필요합니다."
    )

    return GeminiAnalysisResult(
        이름=name,
        분석내용=analysis,
        핵심신호=key_signals,
    )
