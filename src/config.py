from pathlib import Path

from pydantic_settings import BaseSettings


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"


class Settings(BaseSettings):
    database_url: str = "sqlite:///./data/energy.db"
    secret_key: str = "change_me_to_a_secure_random_key"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60
    mqtt_broker_host: str = "localhost"
    mqtt_broker_port: int = 1883

    class Config:
        env_file = BASE_DIR / ".env"
        env_file_encoding = "utf-8"


settings = Settings()

for d in (DATA_DIR, MODELS_DIR, LOGS_DIR):
    d.mkdir(parents=True, exist_ok=True)

