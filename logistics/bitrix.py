"""Передача маршрута водителю в Bitrix24 (этап после MVP — здесь готовая заготовка).

Нужно: входящий вебхук портала с правом `im` (BITRIX_WEBHOOK) и bitrix_user_id водителей в fleet.yaml.
Без вебхука функция ничего не отправляет и возвращает, что отправила бы (для проверки текста).
"""
from __future__ import annotations

import httpx

from .config import load_fleet, settings
from .models import Plan


def message_for(plan: Plan, route) -> str:
    lines = [f"[B]Маршрут на {plan.date}[/B]"]
    for n, p in enumerate(route.stops, 1):
        s = plan.stops[p.stop_id]
        mark = " [B]СРОЧНО[/B]" if s.urgent else ""
        lines.append(f"{n}. {p.eta} — {s.title}{mark}\n   {s.address}\n   {s.task} {s.contact}".rstrip())
    lines.append(f"Возврат ≈ {route.return_eta}, {route.distance_km} км")
    lines.append(f"[URL={route.yandex_url}]Маршрут в Яндекс.Картах[/URL]")
    lines.append(f"[URL={settings.public_url}/driver/{plan.date}/{route.driver_id}]Лист маршрута[/URL]")
    return "\n".join(lines)


def send(plan: Plan) -> list[dict]:
    fleet = load_fleet()
    out = []
    for r in plan.routes:
        d = next(x for x in fleet["drivers"] if x["id"] == r.driver_id)
        msg = message_for(plan, r)
        if not settings.bitrix_webhook or not d.get("bitrix_user_id"):
            out.append({"driver": d["name"], "sent": False, "reason": "нет BITRIX_WEBHOOK или bitrix_user_id",
                        "message": msg})
            continue
        resp = httpx.post(settings.bitrix_webhook.rstrip("/") + "/im.message.add.json", timeout=15,
                          data={"DIALOG_ID": d["bitrix_user_id"], "MESSAGE": msg})
        out.append({"driver": d["name"], "sent": resp.is_success, "response": resp.json()})
    return out
