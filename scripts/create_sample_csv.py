import argparse
import csv
import math
import os
from html import unescape
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen
import xml.etree.ElementTree as ET

DEFAULT_SERVICE_KEY = (
    "L+to0WzuEIGDdZMhaRFd1tSsQPCsPEcjCHzUKhZMdCO4dyLJxAEvBLEyvjUub34lQDQ+Kt+eAOLLzjMocDD4fg=="
)
DEFAULT_OUTPUT_FILENAME = "sample_institutions.csv"
DEFAULT_NUM_OF_ROWS = 100
DEFAULT_LIFE_ARRAY = "003"
DEFAULT_SRCH_KEY_CODE = "003"
CSV_HEADERS = [
    "유형",
    "이름",
    "지원대상",
    "지원내용",
    "신청절차",
    "필요서류",
    "문의처",
    "링크",
    "출처구분",
    "소관기관",
    "지역",
    "지원주기",
    "지원형태",
    "온라인신청가능여부",
    "생애주기",
    "관심주제",
    "원본서비스ID",
    "원본조회번호",
]
CENTRAL_API_URL = (
    "https://apis.data.go.kr/B554287/NationalWelfareInformationsV001/NationalWelfarelistV001"
)
LOCAL_API_URL = (
    "https://apis.data.go.kr/B554287/LocalGovernmentWelfareInformations/LcgvWelfarelist"
)


def _clean_text(value: str | None) -> str:
    if not value:
        return ""
    return " ".join(unescape(value).replace("\r", " ").replace("\n", " ").split())


def _get_text(element: ET.Element, tag: str) -> str:
    child = element.find(tag)
    return _clean_text(child.text if child is not None else "")


def _join_non_empty(*parts: str) -> str:
    return " / ".join(part for part in parts if part)


def _build_query_url(base_url: str, params: dict[str, Any]) -> str:
    return f"{base_url}?{urlencode(params)}"


def _fetch_xml_root(base_url: str, params: dict[str, Any]) -> ET.Element:
    url = _build_query_url(base_url, params)
    with urlopen(url) as response:
        payload = response.read()
    return ET.fromstring(payload)


def _fetch_all_service_nodes(base_url: str, params: dict[str, Any]) -> list[ET.Element]:
    first_root = _fetch_xml_root(base_url, params)
    result_code = _get_text(first_root, "resultCode")
    result_message = _get_text(first_root, "resultMessage")
    if result_code and result_code != "0":
        raise RuntimeError(f"API request failed: {result_code} {result_message}".strip())

    total_count_text = _get_text(first_root, "totalCount")
    total_count = int(total_count_text or "0")
    page_size = int(str(params["numOfRows"]))

    nodes = list(first_root.findall("servList"))
    if total_count <= page_size:
        return nodes

    total_pages = math.ceil(total_count / page_size)
    for page_no in range(2, total_pages + 1):
        page_params = dict(params)
        page_params["pageNo"] = page_no
        root = _fetch_xml_root(base_url, page_params)
        page_result_code = _get_text(root, "resultCode")
        if page_result_code and page_result_code != "0":
            page_result_message = _get_text(root, "resultMessage")
            raise RuntimeError(
                f"API request failed on page {page_no}: "
                f"{page_result_code} {page_result_message}".strip()
            )
        nodes.extend(root.findall("servList"))

    return nodes


def _normalize_central_service(node: ET.Element) -> dict[str, str]:
    ministry = _get_text(node, "jurMnofNm")
    department = _get_text(node, "jurOrgNm")
    life_stage = _get_text(node, "lifeArray")
    target_traits = _get_text(node, "trgterIndvdlArray")
    online_apply = "온라인 신청 가능" if _get_text(node, "onapPsbltYn") == "Y" else "온라인 신청 정보 없음"
    contact = _get_text(node, "rprsCtadr")
    detail_link = _get_text(node, "servDtlLink")

    return {
        "유형": "제도",
        "이름": _get_text(node, "servNm"),
        "지원대상": _join_non_empty(
            f"생애주기: {life_stage}" if life_stage else "",
            f"대상특성: {target_traits}" if target_traits else "",
        ),
        "지원내용": _get_text(node, "servDgst"),
        "신청절차": _join_non_empty(
            online_apply,
            "복지로 상세 페이지 참고",
        ),
        "필요서류": "복지로 상세 페이지 참고",
        "문의처": contact or _join_non_empty(ministry, department),
        "링크": detail_link,
        "출처구분": "중앙부처복지",
        "소관기관": _join_non_empty(ministry, department),
        "지역": "전국",
        "지원주기": _get_text(node, "sprtCycNm"),
        "지원형태": _get_text(node, "srvPvsnNm"),
        "온라인신청가능여부": _get_text(node, "onapPsbltYn"),
        "생애주기": life_stage,
        "관심주제": _get_text(node, "intrsThemaArray"),
        "원본서비스ID": _get_text(node, "servId"),
        "원본조회번호": _get_text(node, "inqNum"),
    }


def _normalize_local_service(node: ET.Element) -> dict[str, str]:
    province = _get_text(node, "ctpvNm")
    city = _get_text(node, "sggNm")
    department = _get_text(node, "bizChrDeptNm")
    life_stage = _get_text(node, "lifeNmArray")
    target_traits = _get_text(node, "trgterIndvdlNmArray")

    return {
        "유형": "제도",
        "이름": _get_text(node, "servNm"),
        "지원대상": _join_non_empty(
            f"생애주기: {life_stage}" if life_stage else "",
            f"대상특성: {target_traits}" if target_traits else "",
            f"대상지역: {_join_non_empty(province, city)}" if province or city else "",
        ),
        "지원내용": _get_text(node, "servDgst"),
        "신청절차": _join_non_empty(
            _get_text(node, "aplyMtdNm"),
            "복지로 상세 페이지 참고",
        ),
        "필요서류": "복지로 상세 페이지 참고",
        "문의처": department or _join_non_empty(province, city),
        "링크": _get_text(node, "servDtlLink"),
        "출처구분": "지자체복지",
        "소관기관": department,
        "지역": _join_non_empty(province, city),
        "지원주기": _get_text(node, "sprtCycNm"),
        "지원형태": _get_text(node, "srvPvsnNm"),
        "온라인신청가능여부": "",
        "생애주기": life_stage,
        "관심주제": _get_text(node, "intrsThemaNmArray"),
        "원본서비스ID": _get_text(node, "servId"),
        "원본조회번호": _get_text(node, "inqNum"),
    }


def _deduplicate_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str]] = set()
    deduped: list[dict[str, str]] = []

    for row in rows:
        key = (row["출처구분"], row["원본서비스ID"] or row["이름"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)

    return deduped


def fetch_central_welfare_rows(
    service_key: str,
    life_array: str = DEFAULT_LIFE_ARRAY,
    srch_key_code: str = DEFAULT_SRCH_KEY_CODE,
    num_of_rows: int = DEFAULT_NUM_OF_ROWS,
) -> list[dict[str, str]]:
    params = {
        "serviceKey": service_key,
        "callTp": "L",
        "pageNo": 1,
        "numOfRows": num_of_rows,
        "srchKeyCode": srch_key_code,
        "lifeArray": life_array,
    }
    nodes = _fetch_all_service_nodes(CENTRAL_API_URL, params)
    return [_normalize_central_service(node) for node in nodes]


def fetch_local_welfare_rows(
    service_key: str,
    life_array: str = DEFAULT_LIFE_ARRAY,
    num_of_rows: int = DEFAULT_NUM_OF_ROWS,
) -> list[dict[str, str]]:
    params = {
        "serviceKey": service_key,
        "pageNo": 1,
        "numOfRows": num_of_rows,
        "lifeArray": life_array,
        "arrgOrd": "001",
    }
    nodes = _fetch_all_service_nodes(LOCAL_API_URL, params)
    return [_normalize_local_service(node) for node in nodes]


def build_integrated_welfare_rows(
    service_key: str,
    life_array: str = DEFAULT_LIFE_ARRAY,
    srch_key_code: str = DEFAULT_SRCH_KEY_CODE,
    num_of_rows: int = DEFAULT_NUM_OF_ROWS,
) -> list[dict[str, str]]:
    central_rows = fetch_central_welfare_rows(
        service_key=service_key,
        life_array=life_array,
        srch_key_code=srch_key_code,
        num_of_rows=num_of_rows,
    )
    local_rows = fetch_local_welfare_rows(
        service_key=service_key,
        life_array=life_array,
        num_of_rows=num_of_rows,
    )

    integrated_rows = _deduplicate_rows(central_rows + local_rows)
    integrated_rows.sort(key=lambda row: (row["출처구분"], row["지역"], row["이름"]))
    return integrated_rows


def create_sample_csv(output_path: Path, service_key: str) -> list[dict[str, str]]:
    rows = build_integrated_welfare_rows(service_key=service_key)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_HEADERS)
        writer.writeheader()
        writer.writerows(rows)

    return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="중앙부처복지 + 지자체복지 공공데이터를 통합 CSV로 저장합니다."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / DEFAULT_OUTPUT_FILENAME,
        help="생성할 CSV 경로",
    )
    parser.add_argument(
        "--service-key",
        default=os.getenv("SOCIAL_WELFARE_API_KEY", DEFAULT_SERVICE_KEY),
        help="공공데이터포털 서비스 키",
    )
    parser.add_argument(
        "--life-array",
        default=DEFAULT_LIFE_ARRAY,
        help="생애주기 코드 (기본값: 003, 청소년)",
    )
    parser.add_argument(
        "--srch-key-code",
        default=DEFAULT_SRCH_KEY_CODE,
        help="중앙부처복지 검색 키 코드 (기본값: 003, 청소년 카테고리)",
    )
    parser.add_argument(
        "--num-of-rows",
        type=int,
        default=DEFAULT_NUM_OF_ROWS,
        help="API 페이지당 요청 건수",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rows = build_integrated_welfare_rows(
        service_key=args.service_key,
        life_array=args.life_array,
        srch_key_code=args.srch_key_code,
        num_of_rows=args.num_of_rows,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_HEADERS)
        writer.writeheader()
        writer.writerows(rows)

    central_count = sum(1 for row in rows if row["출처구분"] == "중앙부처복지")
    local_count = sum(1 for row in rows if row["출처구분"] == "지자체복지")
    print(
        f"통합 CSV 생성 완료: {args.output} "
        f"(총 {len(rows)}건, 중앙부처 {central_count}건, 지자체 {local_count}건)"
    )


if __name__ == "__main__":
    main()
