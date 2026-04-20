from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Student Support Recommendation API"
    app_env: str = "local"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash-lite"
    gemini_embedding_model: str = "models/gemini-embedding-001"
    gemini_max_retries: int = 0
    allow_local_fallback: bool = True
    use_gemini_embeddings: bool = False
    institutions_csv_path: str = "welfare_integrated_data.csv"


    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
