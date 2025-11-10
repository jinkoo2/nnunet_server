from pydantic_settings  import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "nnUNet Server"
    DATABASE_URL: str = "sqlite:///./nnunet.db"
    SECRET_KEY: str
    LOG_LEVEL: str = "INFO"
    SLURM_USER: str = "jinkokim"
    NNUNET_DATA_DIR: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
