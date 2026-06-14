from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import load_app_config
from app.voice.service import ResidentVoiceService


def main():
    config = load_app_config("configs/voice.yaml")
    service = ResidentVoiceService(config)
    print(service.build_health_report())
    service.shutdown()


if __name__ == "__main__":
    main()
