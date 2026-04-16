from fastapi import FastAPI, HTTPException

from app.config import settings
from app.gemini_analyzer import analyze_student_data
from app.models import AnalysisSummary, AnalyzeStudentRequest, AnalyzeStudentResponse
from app.models import RecommendationItem
from app.rag import search_relevant_institutions

app = FastAPI(title=settings.app_name)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok", "env": settings.app_env}


def _split_to_list(text: str) -> list[str]:
    normalized = (
        text.replace(" 및 ", "|")
        .replace(" 후 ", "|")
        .replace("+", "|")
        .replace(",", "|")
        .replace("/", "|")
    )
    items = [part.strip() for part in normalized.split("|") if part.strip()]
    return items if items else [text.strip()]


@app.post("/api/analyze-student", response_model=AnalyzeStudentResponse)
def analyze_student(request: AnalyzeStudentRequest) -> AnalyzeStudentResponse:
    try:
        analysis = analyze_student_data(request)
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

    rag_query = ", ".join(key_signals)

    try:
        rag_results = search_relevant_institutions(query=rag_query, top_k=3)
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"RAG retrieval failed: {error}",
        ) from error

    recommendations: list[RecommendationItem] = []
    for item in rag_results:
        score_percent = round(max(0.0, min(1.0, item["relevance_score"])) * 100)
        support_list = _split_to_list(item["지원내용"])
        process_list = _split_to_list(item["신청절차"])
        docs_list = _split_to_list(item["필요서류"])
        institution_description = (
            "지역 기관 안내 데이터 기반으로 학생 특성과 연관성이 높은 "
            f"{item['유형']} 연계 정보"
        )
        recommendations.append(
            RecommendationItem(
                구분=item["유형"],
                기관명=item["이름"],
                적합도=f"{score_percent}%",
                기관설명=institution_description,
                대상=item["지원대상"],
                지원내용=support_list,
                신청절차=process_list,
                필요서류=docs_list,
                문의=item["문의처"],
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
