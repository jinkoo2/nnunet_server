from pydantic import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "nnUNet Server"
    DATABASE_URL: str = "sqlite:///./nnunet.db"
    SECRET_KEY: str
    LOG_LEVEL: str = "INFO"
    SLURM_USER: str = "youruser"

    class Config:
        env_file = ".env"

settings = Settings()
