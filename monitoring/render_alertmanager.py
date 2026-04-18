from pathlib import Path
import os

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = ROOT / "monitoring" / "alertmanager.yml.template"
OUTPUT_PATH = ROOT / "monitoring" / "alertmanager.yml"


def main() -> None:
    load_dotenv(ROOT / ".env", override=True)

    required = {
        "ALERT_EMAIL_FROM": os.getenv("ALERT_EMAIL_FROM", ""),
        "ALERT_EMAIL_USER": os.getenv("ALERT_EMAIL_USER", ""),
        "ALERT_EMAIL_PASS": os.getenv("ALERT_EMAIL_PASS", ""),
    }
    missing = [key for key, value in required.items() if not value]
    if missing:
        raise SystemExit(f"Missing required env vars for Alertmanager: {', '.join(missing)}")

    rendered = TEMPLATE_PATH.read_text(encoding="utf-8")
    for key, value in required.items():
        rendered = rendered.replace(f"{{{{{key}}}}}", value)

    OUTPUT_PATH.write_text(rendered, encoding="utf-8")
    print(f"Rendered {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
