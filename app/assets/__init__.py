"""
VeritasLM Visual Assets and Brand Identity Specifications.
"""
from pathlib import Path
import base64

ASSETS_DIR = Path(__file__).parent
LOGO_PNG_PATH = ASSETS_DIR / "veritaslm_logo.png"
ICON_PNG_PATH = ASSETS_DIR / "veritaslm_icon.png"
ICON_64_PNG_PATH = ASSETS_DIR / "veritaslm_icon_64.png"

# Read 64x64 icon for inline CSS and HTML data URIs
try:
    with open(ICON_64_PNG_PATH, "rb") as _f:
        VERITASLM_LOGO_B64 = base64.b64encode(_f.read()).decode("utf-8")
except Exception:
    VERITASLM_LOGO_B64 = ""

VERITASLM_LOGO_DATA_URI = f"data:image/png;base64,{VERITASLM_LOGO_B64}"
