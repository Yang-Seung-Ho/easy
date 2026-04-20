from pydantic import BaseModel, ConfigDict, Field


class StudentPersonalInfo(BaseModel):
    student_name: str = Field(..., alias="학생이름")
    grade: int = Field(..., alias="학년")
    class_number: int = Field(..., alias="반")
    birth_date: str = Field(..., alias="생년월일")
    gender: str = Field(..., alias="성별")

    model_config = ConfigDict(extra="forbid")


class HomeEnvironmentAndEligibility(BaseModel):
    student_basic_info: str = Field(..., alias="학생기본사항")
    basic_living_security_status: str = Field(..., alias="기초수급보장현황")
    family_status: str = Field(..., alias="가족현황")

    model_config = ConfigDict(extra="forbid")


class StudentDifficulties(BaseModel):
    academics: str = Field(..., alias="학업")
    emotional_psychological: str = Field(..., alias="심리_정서")
    care_safety_health: str = Field(..., alias="돌봄_안전_건강")
    economy_life: str = Field(..., alias="경제_생활")
    etc: str = Field(..., alias="기타")

    model_config = ConfigDict(extra="forbid")


class StudentCondition(BaseModel):
    student_status: str = Field(..., alias="학생현황")
    student_difficulties: StudentDifficulties = Field(..., alias="학생어려움")

    model_config = ConfigDict(extra="forbid")


class IntegratedApplicationInfo(BaseModel):
    student_personal_info: StudentPersonalInfo = Field(..., alias="학생인적사항")
    home_environment_and_eligibility: HomeEnvironmentAndEligibility = Field(
        ...,
        alias="가정환경및자격",
    )
    student_condition: StudentCondition = Field(..., alias="학생상태")
    application_reason: str = Field(..., alias="신청사유")
    support_request: str = Field(..., alias="지원요청사항")

    model_config = ConfigDict(extra="forbid")


class ObservationLog(BaseModel):
    teacher_name: str = Field(..., alias="교사이름")
    position: str = Field(..., alias="직위")
    date: str = Field(..., alias="날짜")
    time: str = Field(..., alias="시간")
    place: str = Field(..., alias="장소")
    content: str = Field(..., alias="내용")
    special_notes: str = Field(..., alias="특이사항")

    model_config = ConfigDict(extra="forbid")


class AllData(BaseModel):
    integrated_application_info: IntegratedApplicationInfo = Field(
        ...,
        alias="통합신청서정보",
    )
    observation_logs: list[ObservationLog] = Field(..., alias="관찰일지목록")

    model_config = ConfigDict(extra="forbid")


class AnalyzeStudentRequest(BaseModel):
    all_data: AllData = Field(..., alias="전체데이터")

    model_config = ConfigDict(extra="forbid")


class AnalysisSummary(BaseModel):
    name: str = Field(..., alias="이름")
    summarized_analysis: str = Field(..., alias="요약분석")
    key_signals: list[str] = Field(..., alias="핵심신호")

    model_config = ConfigDict(extra="forbid", populate_by_name=True, serialize_by_alias=True)


class RecommendationItem(BaseModel):
    category: str
    suitability: int          # relevance_score * 100 반올림값 (0~100)
    welfareType: str
    servId: str
    servNm: str
    agency: str
    department: str
    intrsThemaArray: list[str]
    lifeArray: list[str]
    srvPvsnNm: str
    sprtCycNm: str
    servDgst: str
    servDtlLink: str
    inqNum: int
    contact: str | None

    model_config = ConfigDict(extra="forbid")


class AnalyzeStudentResponse(BaseModel):
    ai_analysis_summary: AnalysisSummary = Field(..., alias="ai_분석정리_요약")
    ai_recommended_supports: list[RecommendationItem] = Field(
        ...,
        alias="ai_추천기관_제도",
    )

    model_config = ConfigDict(extra="forbid", populate_by_name=True, serialize_by_alias=True)


class GeminiAnalysisResult(BaseModel):
    name: str = Field(..., alias="이름")
    analysis: str = Field(..., alias="분석내용")
    key_signals: list[str] = Field(..., alias="핵심신호")

    model_config = ConfigDict(extra="forbid", populate_by_name=True, serialize_by_alias=True)
