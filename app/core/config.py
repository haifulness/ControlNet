import os
import time
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
	api_host: str
	device: str = 'cpu'
	s3_root_dir: str = 'cct-raw-prod'
	s3_input_dir: str = 'zil/controlnet/test/input'
	s3_output_dir: str = 'zil/controlnet/test/output'
	s3_access_key: str
	s3_secret_key: str
	model_config = SettingsConfigDict(env_file=".env")


settings = Settings()
