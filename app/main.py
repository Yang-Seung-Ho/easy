from fastapi import FastAPI, HTTPException
import re
import ast

from app.config import settings
from app.gemini_analyzer import analyze_student_data, analyze_observation_domains
from app.models import AnalysisSummary, AnalyzeStudentRequest, AnalyzeStudentResponse
from app.models import RecommendationItem
from app.rag import search_relevant_institutions

app = FastAPI(title=settings.app_name)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok", "env": settings.app_env}


def _parse_array_field(value: str) -> list[str]:
    """'항목1, 항목2' 또는 '["항목1", "항목2"]' 형태 모두 처리"""
    if not value:
        return []
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return [str(i).strip() for i in parsed]
    except Exception:
        pass
    return [v.strip() for v in value.split(",") if v.strip()]


def _build_rag_context(
    request: AnalyzeStudentRequest,
    analysis_summary: str,
    key_signals: list[str],
) -> dict[str, object]:
    info = request.all_data.integrated_application_info
    personal = info.student_personal_info
    home = info.home_environment_and_eligibility
    difficulties = info.student_condition.student_difficulties

    observation_texts = [
        f"{log.content} {log.special_notes}".strip()
        for log in request.all_data.observation_logs
    ]

    full_text_parts = [
        personal.region,
        analysis_summary,
        " ".join(key_signals),
        info.support_request,
        info.application_reason,
        info.student_condition.student_status,
        difficulties.academics,
        difficulties.emotional_psychological,
        difficulties.care_safety_health,
        difficulties.economy_life,
        difficulties.etc,
        home.student_basic_info,
        home.basic_living_security_status,
        home.family_status,
        " ".join(observation_texts),
    ]

    return {
        "student_grade": personal.grade,
        "student_region": personal.region,
        "student_text": " ".join(part for part in full_text_parts if part).strip(),
        "support_request": info.support_request,
        "application_reason": info.application_reason,
        "student_status": info.student_condition.student_status,
        "observation_text": " ".join(observation_texts).strip(),
        "basic_living_security_status": home.basic_living_security_status,
        "student_basic_info": home.student_basic_info,
        "family_status": home.family_status,
        "economy_life": difficulties.economy_life,
        "key_signals": key_signals,
    }


@app.post("/api/analyze-student", response_model=AnalyzeStudentResponse)
def analyze_student(request: AnalyzeStudentRequest) -> AnalyzeStudentResponse:
    try:
        analysis = analyze_student_data(request)
        student_context = (
            f"학년: {request.all_data.integrated_application_info.student_personal_info.grade}\n"
            f"지역: {request.all_data.integrated_application_info.student_personal_info.region}\n"
            f"지원요청: {request.all_data.integrated_application_info.support_request}\n"
            f"핵심신호: {', '.join(analysis.key_signals)}"
        )
        domain_scores = analyze_observation_domains(
            observation_logs=request.all_data.observation_logs,
            student_context=student_context,
        )

    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail=f"Gemini analysis failed: {error}",
        ) from error

    key_signals = analysis.key_signals
    if not key_signals:
        raise HTTPException(
            status_code=422,
            detail="Gemini returned empty 핵심신호. Cannot run RAG search.",
        )

    info = request.all_data.integrated_application_info
    rag_query = " | ".join(
        [
            ", ".join(key_signals),
            info.student_personal_info.region,
            analysis.analysis,
            info.support_request,
            info.application_reason,
            info.student_condition.student_difficulties.academics,
            info.student_condition.student_difficulties.emotional_psychological,
            info.student_condition.student_difficulties.care_safety_health,
            info.student_condition.student_difficulties.economy_life,
            info.student_condition.student_difficulties.etc,
            " | ".join(
                f"{log.content} {log.special_notes}".strip()
                for log in request.all_data.observation_logs
            ),
        ]
    )
    rag_context = _build_rag_context(
        request=request,
        analysis_summary=analysis.analysis,
        key_signals=key_signals,
    )

    try:
        rag_results = search_relevant_institutions(
            query=rag_query,
            top_k=100,
            context=rag_context,
            domain_scores=domain_scores,   # ← 추가

        )
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"RAG retrieval failed: {error}",
        ) from error

    recommendations: list[RecommendationItem] = []
    for item in rag_results:
        suitability = round(max(0.0, min(1.0, float(item.get("relevance_score", 0)))) * 100)
        recommendations.append(
            RecommendationItem(
                category=item.get("category", ""),
                suitability=suitability,
                welfareType=item.get("welfareType", ""),
                servId=item.get("servId", ""),
                servNm=item.get("servNm", ""),
                agency=item.get("agency", ""),
                department=item.get("department", ""),
                intrsThemaArray=_parse_array_field(item.get("intrsThemaArray", "")),
                lifeArray=_parse_array_field(item.get("lifeArray", "")),
                srvPvsnNm=item.get("srvPvsnNm", ""),
                sprtCycNm=item.get("sprtCycNm", ""),
                servDgst=item.get("servDgst", ""),
                servDtlLink=item.get("servDtlLink", ""),
                inqNum=int(item.get("inqNum", 0) or 0),
                contact=item.get("contact") or None,
            )
        )

    return AnalyzeStudentResponse(
        ai_analysis_summary=AnalysisSummary(
            이름=analysis.name,
            요약분석=analysis.analysis,
            핵심신호=analysis.key_signals,
        ),
        ai_recommended_supports=recommendations,
    )
