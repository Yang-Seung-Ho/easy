# 파이썬 3.10 환경 사용
FROM python:3.10-slim

WORKDIR /app

# 라이브러리 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 코드 전체 복사
COPY . .

# 구글 Cloud Run은 기본적으로 8080 포트를 사용합니다.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]