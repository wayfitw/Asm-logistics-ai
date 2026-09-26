"""Настройки из .env и config/fleet.yaml, график водителей."""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PLANS = DATA / "plans"


def _load_env() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


_load_env()


@dataclass(frozen=True)
class Settings:
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_base_url: str | None = os.getenv("OPENAI_BASE_URL") or None   # прокси/шлюз, если API недоступен напрямую
    model: str = os.getenv("OPENAI_MODEL", "gpt-5.5")                   # агент правок и разбор заявок
    model_fast: str = os.getenv("OPENAI_MODEL_FAST", "gpt-5.4-mini")    # дешёвые задачи
    yandex_geocoder_key: str = os.getenv("YANDEX_GEOCODER_KEY", "")
    osrm_url: str = os.getenv("OSRM_URL", "https://router.project-osrm.org")
    google_sa_json: str = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "")
    google_sheet_id: str = os.getenv("GOOGLE_SHEET_ID", "")
    bitrix_webhook: str = os.getenv("BITRIX_WEBHOOK", "")            # https://portal.bitrix24.ru/rest/1/xxxx/
    public_url: str = os.getenv("PUBLIC_URL", "http://localhost:8000")
    import_token: str = os.getenv("IMPORT_TOKEN", "")                # токен, с которым 1С шлёт заявки

    @property
    def llm_enabled(self) -> bool:
        return bool(self.openai_api_key)


settings = Settings()


def load_fleet() -> dict:
    return yaml.safe_load((ROOT / "config" / "fleet.yaml").read_text(encoding="utf-8"))


def hm(s: str) -> int:
    """'08:30' -> 510 минут от полуночи."""
    h, m = s.split(":")
    return int(h) * 60 + int(m)


def parse_hm(value) -> str | None:
    """'9', '9.00', '09:00', '9-30', '9ч' → 'ЧЧ:ММ'; мусор → None."""
    import re
    m = re.fullmatch(r"\s*(\d{1,2})(?:\s*[:.\-ч]\s*(\d{2}))?\s*(?:ч|час\w*)?\s*", str(value or ""))
    if not m:
        return None
    h, mnt = int(m.group(1)), int(m.group(2) or 0)
    return f"{h:02d}:{mnt:02d}" if h < 24 and mnt < 60 else None


def in_region(lat, lon) -> bool:
    """Москва и область с запасом — всё, что вне, считаем ошибкой геокодера или модели."""
    return lat is not None and lon is not None and 54.0 <= lat <= 57.5 and 35.0 <= lon <= 40.5


def fmt_hm(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def driver_works(driver: dict, day: date, overrides: dict) -> bool:
    ov = (overrides or {}).get(day.isoformat(), {})
    if driver["id"] in ov:
        return bool(ov[driver["id"]])
    sch = driver.get("schedule", {})
    if sch.get("type") == "2/2":
        start = date.fromisoformat(sch["cycle_start"])
        return (day - start).days % 4 in (0, 1)
    return day.weekday() < 5   # 5/2
