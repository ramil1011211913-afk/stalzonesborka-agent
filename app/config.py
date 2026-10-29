"""Configuration loader from environment variables."""
import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # Stalzone API credentials — must be set via environment variables
    CLIENT_ID = os.getenv("STALCRAFT_CLIENT_ID", "")
    CLIENT_SECRET = os.getenv("STALCRAFT_CLIENT_SECRET", "")

    # All available API keys for parallel/rotated fetching
    API_KEYS = [(CLIENT_ID, CLIENT_SECRET)]
    for _i in range(2, 10):
        _cid = os.getenv(f"STALCRAFT_CLIENT_ID_{_i}", "")
        _csec = os.getenv(f"STALCRAFT_CLIENT_SECRET_{_i}", "")
        if _cid and _csec:
            API_KEYS.append((_cid, _csec))

    # Region: RU, EU, NA, SEA, NEA
    REGION = os.getenv("STALCRAFT_REGION", "EU")

    # API base URLs
    API_BASE = os.getenv("STALCRAFT_API_BASE", "https://eapi.stalzone.com")
    OAUTH_BASE = os.getenv("STALCRAFT_OAUTH_BASE", "https://exbo.net/oauth")

    # GitHub database repo
    DB_REPO_OWNER = "EXBO-Studio"
    DB_REPO_NAME = "stalzone-database"
    DB_REPO_BRANCH = "main"
    DB_REALM = "ru"  # items database realm (ru or global)

    # Request settings
    REQUEST_TIMEOUT = 30
    GITHUB_API_BASE = "https://api.github.com"
    RAW_GITHUB_BASE = "https://raw.githubusercontent.com"

    # Data directory
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    DB_PATH = os.path.join(DATA_DIR, "stalzone.db")

    @classmethod
    def auth_headers(cls) -> dict:
        """Client-Id/Client-Secret header auth — bypasses exbo.net OAuth entirely."""
        return {
            "Client-Id": cls.CLIENT_ID,
            "Client-Secret": cls.CLIENT_SECRET,
        }
