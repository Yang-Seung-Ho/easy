import csv
from pathlib import Path


CSV_HEADERS = ["유형", "이름", "지원대상", "지원내용", "신청절차", "필요서류", "문의처"]
CSV_ROWS = [
    {
        "유형": "기관",
        "이름": "마음성장 아동청소년 상담센터",
        "지원대상": "정서 불안 및 행동 조절 어려움이 있는 초중학생",
        "지원내용": "주 1회 전문 심리상담 및 부모 코칭 8회기 제공",
        "신청절차": "학교 추천서 접수 후 초기면접 진행",
        "필요서류": "통합신청서+보호자 동의서+학교의견서",
        "문의처": "02-1111-2222",
    },
    {
        "유형": "제도",
        "이름": "기초학력 디딤학습 바우처",
        "지원대상": "기초 학력 미달 또는 학습 결손이 확인된 초등학생",
        "지원내용": "방과 후 학습코칭 비용 월 20만원 한도 지원",
        "신청절차": "온라인 신청 후 학교 확인 절차",
        "필요서류": "학생생활기록 요약+진단평가 결과지+신분증 사본",
        "문의처": "교육청 콜센터 1588-3000",
    },
    {
        "유형": "기관",
        "이름": "우리동네 긴급돌봄센터",
        "지원대상": "방과 후 보호 공백이 발생하는 맞벌이 및 취약가정 학생",
        "지원내용": "평일 13시-20시 긴급 돌봄 및 간식 지원",
        "신청절차": "주민센터 또는 학교 복지담당 연계 신청",
        "필요서류": "돌봄 공백 확인서+가족관계증명서",
        "문의처": "동주민센터 120",
    },
    {
        "유형": "제도",
        "이름": "교육비 안심지원 특별사업",
        "지원대상": "저소득 및 차상위계층 학생 가정",
        "지원내용": "체험학습비와 방과후활동비 일부를 학기별 정액 지원",
        "신청절차": "학교 행정실 신청서 제출 후 심사",
        "필요서류": "수급자/차상위 확인서+통장 사본",
        "문의처": "학교 행정실 또는 02-3333-4444",
    },
]


def create_sample_csv(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_HEADERS)
        writer.writeheader()
        writer.writerows(CSV_ROWS)


if __name__ == "__main__":
    target_path = Path(__file__).resolve().parents[1] / "sample_institutions.csv"
    create_sample_csv(target_path)
    print(f"Sample CSV created: {target_path}")
