import json

from app.rag import search_relevant_institutions


def run_test() -> None:
    query = "정서 불안, 돌봄 공백, 기초 학력 미달"
    results = search_relevant_institutions(query=query, top_k=3)
    print(f"Query: {query}")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run_test()
