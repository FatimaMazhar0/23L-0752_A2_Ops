import os
from dotenv import load_dotenv

load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH", "models/champion.keras")
THRESHOLD = float(os.getenv("THRESHOLD", "0.46"))     # validation-chosen in 2.5
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///instance/visionguard.db")
MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "5"))
MODEL_NAME = os.getenv("MODEL_NAME", "23L-0752_A2_Ops")
MODEL_VERSION = os.getenv("MODEL_VERSION", "2")
IMG_SIZE = 160


MODEL_BACKEND = os.getenv("MODEL_BACKEND", "local")     # "local" or "remote"
MODEL_API_URL = os.getenv("MODEL_API_URL", "").rstrip("/")
MODEL_API_KEY = os.getenv("MODEL_API_KEY", "")
MODEL_API_TIMEOUT = float(os.getenv("MODEL_API_TIMEOUT", "60"))