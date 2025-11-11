from pydantic_settings  import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "nnUNet Server"
    DATABASE_URL: str = "sqlite:///./nnunet.db"
    SECRET_KEY: str
    LOG_LEVEL: str = "INFO"
    SLURM_USER: str = "jinkokim"
    NNUNET_DATA_DIR: str
    JOB_PROCESSOR: str = "slurm"

    venv_dir: str = "/home/jk/nnunet/_venv"
    scripts_dir: str = "/home/jk/projects/nnunet_server/scripts"
    nnunet_dir: str = "/home/jk/nnunet"
    data_dir: str = "/home/jk/data/nnunet_data"
    script_output_files_dir: str = "/home/jk/data/nnunet_data/script_output_files"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
