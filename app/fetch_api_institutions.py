import pandas as pd
import xml.etree.ElementTree as ET

# 1. API(XML) 데이터 파싱
xml_data = """<response>
... (제공해주신 XML 데이터) ...
</response>"""

root = ET.fromstring(xml_data)
items = root.findall('.//item')

xml_rows = []
for idx, item in enumerate(items):
    name = item.find('teenGdctCntrNm').text if item.find('teenGdctCntrNm') is not None else ""
    region = item.find('areaDvsnNm').text if item.find('areaDvsnNm') is not None else ""
    address = item.find('addr').text if item.find('addr') is not None else ""
    phone = item.find('telno').text if item.find('telno') is not None else ""
    home = item.find('hmpgAddr').text if item.find('hmpgAddr') is not None else ""
    
    xml_rows.append({
        'category': '기관',
        'welfareType': '지역센터',
        'servId': f'ORG{idx+1:04d}',
        'servNm': name,
        'agency': region,
        'department': '',
        'intrsThemaArray': '청소년',
        'lifeArray': '청소년',
        'srvPvsnNm': '상담/지원',
        'sprtCycNm': '수시',
        'servDgst': address,
        'servDtlLink': home,
        'inqNum': '',
        'contact': phone
    })

df_xml = pd.DataFrame(xml_rows)

# 2. CSV 파일 데이터 로드 및 중복 제거 병합
csv_files = [
    "한국청소년상담복지개발원_전국_꿈드림_센터_20250903 (3).csv", 
    "한국청소년상담복지개발원_전국_꿈드림_센터_20250903 (2).csv"
]
df_csvs = []
for f in csv_files:
    try:
        df = pd.read_csv(f, encoding='utf-8')
    except:
        df = pd.read_csv(f, encoding='cp949')
    df_csvs.append(df)

df_csv_all = pd.concat(df_csvs).drop_duplicates()

# 3. CSV 데이터 포맷 매핑
csv_rows = []
start_idx = len(xml_rows) + 1
for _, row in df_csv_all.iterrows():
    name = row.get('센터명', '')
    region = row.get('시도', '')
    addr1 = str(row.get('주소1', ''))
    addr2 = str(row.get('주소2', ''))
    
    # 주소2가 "비어있음" 등의 값일 경우를 대비해 처리
    if addr2 == '비어있음' or addr2 == 'nan':
        address = addr1.strip()
    else:
        address = f"{addr1} {addr2}".strip()
        
    phone = row.get('대표전화번호', '')
    home = row.get('홈페이지', '')
    if home == '비어있음':
        home = ''
    
    csv_rows.append({
        'category': '기관',
        'welfareType': '지역센터',
        'servId': f'ORG{start_idx:04d}',
        'servNm': name,
        'agency': region,
        'department': '',
        'intrsThemaArray': '청소년',
        'lifeArray': '청소년',
        'srvPvsnNm': '상담/지원',
        'sprtCycNm': '수시',
        'servDgst': address,
        'servDtlLink': home,
        'inqNum': '',
        'contact': phone
    })
    start_idx += 1

# 4. 최종 병합 및 파일 추출
df_final = pd.concat([df_xml, pd.DataFrame(csv_rows)])
df_final.to_csv("integrated_centers.csv", index=False, encoding='utf-8-sig')