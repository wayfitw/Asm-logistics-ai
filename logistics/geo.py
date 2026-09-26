"""Геокодирование и матрица времени/расстояний.

Геокодер: координаты из текста → Яндекс Геокодер (если есть ключ) → Nominatim.
Матрица: OSRM (без пробок, поэтому умножаем на TRAFFIC_FACTOR) → запасной вариант по прямой.
Для боевого режима по Москве лучше Яндекс Матрица расстояний с пробками — интерфейс тот же.
"""
from __future__ import annotations

import json
import math
import os
import re
import time

import httpx

from .config import DATA, settings

TRAFFIC_FACTOR = float(os.getenv("TRAFFIC_FACTOR", "1.5"))
_CACHE_FILE = DATA / "geocache.json"
_COORDS_RE = re.compile(r"(5[45]\.\d{3,})\s*[,;\s]\s*(3[5-9]\.\d{3,})")   # Москва и область


def _cache() -> dict:
    if _CACHE_FILE.exists():
        return json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
    return {}


def _save_cache(c: dict) -> None:
    _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _CACHE_FILE.write_text(json.dumps(c, ensure_ascii=False, indent=1), encoding="utf-8")


def coords_from_text(text: str) -> tuple[float, float] | None:
    m = _COORDS_RE.search(text or "")
    return (float(m.group(1)), float(m.group(2))) if m else None


def geocode(address: str) -> tuple[float, float] | None:
    if c := coords_from_text(address):
        return c
    cache = _cache()
    if address in cache:
        return tuple(cache[address]) if cache[address] else None
    res = None
    try:
        if settings.yandex_geocoder_key:
            r = httpx.get("https://geocode-maps.yandex.ru/1.x/", timeout=10, params={
                "apikey": settings.yandex_geocoder_key, "geocode": address, "format": "json", "results": 1,
                "bbox": "35.1,54.2~40.2,56.97", "rspn": 0})
            members = r.json()["response"]["GeoObjectCollection"]["featureMember"]
            if members:
                lon, lat = map(float, members[0]["GeoObject"]["Point"]["pos"].split())
                res = (lat, lon)
        else:
            for q in _variants(address):
                time.sleep(1)   # правила Nominatim: не чаще 1 запроса в секунду
                r = httpx.get("https://nominatim.openstreetmap.org/search", timeout=10,
                              headers={"User-Agent": "asm-logistics-prototype"},
                              params={"q": q, "format": "json", "limit": 1, "countrycodes": "ru",
                                      "viewbox": "35.1,56.97,40.2,54.2"})
                if r.json():
                    res = (float(r.json()[0]["lat"]), float(r.json()[0]["lon"]))
                    break
    except Exception:
        res = None
    if res:   # промахи не кэшируем — адрес могут поправить
        cache[address] = res
        _save_cache(cache)
    return res


# (?<![\w-]) — чтобы «г.» не срабатывало внутри слов, а «МО» не съедало «Мо» в «Москва»
_W = r"(?<![\w-])"
_ABBR = [(_W + r"ш\.", "шоссе"), (_W + r"пр-т", "проспект"), (_W + r"пр-д", "проезд"), (_W + r"б-р", "бульвар"),
         (_W + r"ул\.", "улица"), (_W + r"пер\.", "переулок"), (_W + r"(д\.|дом)\s*", ""),
         (_W + r"(г\.|город)\s*", ""), (_W + r"(МО|Московская обл(\.|асть)?)(?!\w)", ""),
         (_W + r"\d{6}(?!\d)", ""), (_W + r"(вл\.|владение|домовладение)\s*", ""),
         (r",?\s*(кв(\.|артира)|оф(\.|ис))\s*\d+", "")]


def _variants(address: str) -> list[str]:
    """Nominatim плохо понимает сокращения 1С — пробуем «очищенные» варианты адреса."""
    clean = address
    for pat, rep in _ABBR:
        clean = re.sub(pat, rep, clean)
    clean = re.sub(r"\s*,\s*(,\s*)+", ", ", clean).strip(" ,")
    no_building = re.sub(r",?\s*(стр(оение|\.)?|к(орп(ус|\.)?)?)\s*\d+\w*\s*$", "", clean, flags=re.I)
    no_building = re.sub(r"(\d+)[А-Яа-я]$", r"\1", no_building)   # «1Б» → «1»
    return list(dict.fromkeys([address, clean, no_building]))


def _haversine_km(a, b) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (*a, *b))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(h))


_matrix_cache: dict[tuple, tuple] = {}


def matrix(points: list[tuple[float, float]]) -> tuple[list[list[int]], list[list[float]]]:
    """Возвращает (минуты, километры) между всеми парами точек."""
    key = tuple(points)
    if key not in _matrix_cache:
        if len(_matrix_cache) > 500:
            _matrix_cache.clear()
        _matrix_cache[key] = _matrix(points)
    return _matrix_cache[key]


def _matrix(points: list[tuple[float, float]]) -> tuple[list[list[int]], list[list[float]]]:
    try:
        coords = ";".join(f"{lon},{lat}" for lat, lon in points)
        r = httpx.get(f"{settings.osrm_url}/table/v1/driving/{coords}",
                      params={"annotations": "duration,distance"}, timeout=20)
        d = r.json()
        if d.get("code") != "Ok":
            raise ValueError(d.get("message"))
        mins = [[round(s / 60 * TRAFFIC_FACTOR) for s in row] for row in d["durations"]]
        kms = [[round(m / 1000, 1) for m in row] for row in d["distances"]]
        return mins, kms
    except Exception:
        kms = [[round(_haversine_km(a, b) * 1.35, 1) for b in points] for a in points]
        mins = [[round(k / 25 * 60) for k in row] for row in kms]   # ~25 км/ч по Москве
        return mins, kms


def yandex_route_url(points: list[tuple[float, float]]) -> str:
    return "https://yandex.ru/maps/?rtt=auto&rtext=" + "~".join(f"{lat},{lon}" for lat, lon in points)
