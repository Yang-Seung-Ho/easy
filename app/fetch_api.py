import os
import csv
import requests
import xml.etree.ElementTree as ET
from dotenv import load_dotenv

# 헬퍼 함수: 콤마 구분자 텍스트를 깔끔한 문자열로 변환 (CSV 저장용)
def _parse_array_to_string(text):
    if not text:
        return ""
    items = [item.strip() for item in text.split(',')]
    return ", ".join(items)

# 헬퍼 함수: 정수 변환
def _parse_int(text):
    try:
        return int(text) if text else 0
    except ValueError:
        return 0

# 헬퍼 함수: 줄바꿈 등 공백 문자 제거 (CSV 구조 깨짐 방지)
def _clean_text(text):
    if not text:
        return ""
    return " ".join(text.replace("\r", " ").replace("\n", " ").split())

def fetch_national_welfare(api_key, timeout=10):
    """중앙부처 복지 API 호출"""
    url = f"https://apis.data.go.kr/B554287/NationalWelfareInformationsV001/NationalWelfarelistV001?serviceKey={api_key}&callTp=L&pageNo=1&numOfRows=20&srchKeyCode=003&lifeArray=003"
    
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status() 
        
        results = []
        root = ET.fromstring(response.content)

        if not root.findall('.//servList'):
            print("⚠️ 중앙부처 API: 검색된 데이터가 없습니다.")
            return []

        for item in root.findall('.//servList'):
            results.append({
                "category": "제도",
                "welfareType": "중앙부처",
                "servId": _clean_text(item.findtext('servId', default='')),
                "servNm": _clean_text(item.findtext('servNm', default='')),
                "agency": _clean_text(item.findtext('jurMnofNm', default='')),
                "department": _clean_text(item.findtext('jurOrgNm', default='')),
                "intrsThemaArray": _parse_array_to_string(item.findtext('intrsThemaArray', default='')),
                "lifeArray": _parse_array_to_string(item.findtext('lifeArray', default='')),
                "srvPvsnNm": _clean_text(item.findtext('srvPvsnNm', default='')),
                "sprtCycNm": _clean_text(item.findtext('sprtCycNm', default='')),
                "servDgst": _clean_text(item.findtext('servDgst', default='')),
                "servDtlLink": _clean_text(item.findtext('servDtlLink', default='')),
                "inqNum": _parse_int(item.findtext('inqNum', default='0')),
                "contact": _clean_text(item.findtext('rprsCtadr', default=''))
            })
        print(f"✅ 중앙부처 API 호출 성공: {len(results)}건 수집")
        return results

    except requests.exceptions.Timeout:
        print("❌ 중앙부처 API: 응답 시간 초과 (Timeout)")
    except requests.exceptions.HTTPError as errh:
        print(f"❌ 중앙부처 API HTTP 에러: {errh}")
    except Exception as e:
        print(f"❌ 중앙부처 API 오류: {e}")
    
    return []

def fetch_local_welfare(api_key, timeout=10):
    """지자체 복지 API 호출"""
    url = f"https://apis.data.go.kr/B554287/LocalGovernmentWelfareInformations/LcgvWelfarelist?serviceKey={api_key}&pageNo=1&numOfRows=20&lifeArray=003&arrgOrd=001"
    
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        
        results = []
        root = ET.fromstring(response.content)

        if not root.findall('.//servList'):
            print("⚠️ 지자체 API: 검색된 데이터가 없습니다.")
            return []

        for item in root.findall('.//servList'):
            results.append({
                "category": "제도",
                "welfareType": "지자체",
                "servId": _clean_text(item.findtext('servId', default='')),
                "servNm": _clean_text(item.findtext('servNm', default='')),
                "agency": _clean_text(item.findtext('ctpvNm', default='')),
                "department": _clean_text(item.findtext('bizChrDeptNm', default='')),
                "intrsThemaArray": _parse_array_to_string(item.findtext('intrsThemaNmArray', default='')),
                "lifeArray": _parse_array_to_string(item.findtext('lifeNmArray', default='')),
                "srvPvsnNm": _clean_text(item.findtext('srvPvsnNm', default='')),
                "sprtCycNm": _clean_text(item.findtext('sprtCycNm', default='')),
                "servDgst": _clean_text(item.findtext('servDgst', default='')),
                "servDtlLink": _clean_text(item.findtext('servDtlLink', default='')),
                "inqNum": _parse_int(item.findtext('inqNum', default='0')),
                "contact": ""  # 지자체는 연락처가 없으므로 빈 문자열 처리
            })
        print(f"✅ 지자체 API 호출 성공: {len(results)}건 수집")
        return results

    except requests.exceptions.Timeout:
        print("❌ 지자체 API: 응답 시간 초과 (Timeout)")
    except requests.exceptions.HTTPError as errh:
        print(f"❌ 지자체 API HTTP 에러: {errh}")
    except Exception as e:
        print(f"❌ 지자체 API 오류: {e}")
        
    return []

def main():
    load_dotenv()
    
    national_key = os.getenv("NATIONAL_API_KEY")
    local_key = os.getenv("LOCAL_API_KEY")

    if not national_key or not local_key:
        print("오류: .env 파일에 API 키가 설정되지 않았습니다.")
        return

    print("데이터 수집 중...")
    national_data = fetch_national_welfare(national_key, timeout=10)
    local_data = fetch_local_welfare(local_key, timeout=10)
    
    integrated_data = national_data + local_data
    
    if not integrated_data:
        print("수집된 데이터가 없어 파일을 생성하지 않습니다.")
        return

    # CSV 저장을 위한 헤더(필드명) 정의
    fieldnames = [
        "category", "welfareType", "servId", "servNm", "agency", 
        "department", "intrsThemaArray", "lifeArray", "srvPvsnNm", 
        "sprtCycNm", "servDgst", "servDtlLink", "inqNum", "contact"
    ]

    output_filename = "welfare_integrated_data.csv"
    
    # CSV 파일 작성 (utf-8-sig로 저장하여 엑셀에서 한글 깨짐 방지)
    with open(output_filename, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        # 첫 줄에 헤더 쓰기
        writer.writeheader()
        
        # 통합된 데이터 쓰기
        writer.writerows(integrated_data)
        
    print(f"\n🎉 총 {len(integrated_data)}개의 데이터가 성공적으로 통합되어 '{output_filename}'에 저장되었습니다.")

if __name__ == "__main__":
    main()