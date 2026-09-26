"""Всё, что делает ИИ. Провайдер — OpenAI Responses API.

1. normalize()  — разбирает «кривые» заявки (1С + ручные) в чистые точки маршрута (Structured Outputs).
2. PlanAgent    — агент с function calling: логист пишет «Балашиху отдай Ушакову», агент правит план
                  через инструменты, солвер пересчитывает время, агент объясняет, что получилось.
3. explain()    — короткое пояснение к плану для логиста.
"""
from __future__ import annotations

import json
import re

from openai import OpenAI

from .config import in_region, parse_hm, settings
from .models import Plan, RawTask, Stop

_client: OpenAI | None = None


def client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
    return _client


def _fleet_brief(fleet: dict) -> str:
    v = "\n".join(f"- {x['id']}: {x['name']}, {x['capacity_kg']} кг, {x['volume_m3']} м³, "
                  f"{'закрытый' if x['closed_body'] else 'ОТКРЫТЫЙ борт'}"
                  f"{', европаллета НЕ входит' if not x['fits_europallet'] else ''}"
                  f"{', РЕЗЕРВ' if x.get('reserve') else ''}" for x in fleet["vehicles"])
    d = "\n".join(f"- {x['id']}: {x['name']}, машины: {', '.join(x['vehicles'])}" for x in fleet["drivers"])
    return f"База: {fleet['base']['address']}. Смена {fleet['shift']['start']}–{fleet['shift']['end']}.\nМашины:\n{v}\nВодители:\n{d}"


# ---------------------------------------------------------------- 1. Нормализация заявок

NORMALIZE_PROMPT = """Ты — помощник логиста-снабженца компании АСМ (Москва). Тебе дают оплаченные заказы поставщикам
из 1С ERP (с карточками контрагентов) и ручные задачи логиста, записанные как попало.
Преврати их в точки маршрута для водителей.

Правила:
- Заказ поставщику = водитель ЗАБИРАЕТ товар у поставщика (kind=pickup) и везёт на базу.
  Адрес бери складской/фактический, а не юридический. Если в карточке только юридический адрес
  или складов несколько и непонятно какой — выбери наиболее вероятный и напиши warning.
- Ручная задача может содержать несколько действий: «забрать масло на Коломенской → выгрузить в Балашихе» —
  это две точки: pickup и delivery; у pickup в pair_with_local_id укажи local_id доставки
  (local_id уникален в пределах одной задачи task_id: «1», «2»…).
- «Выгрузить/отвезти» без места погрузки = delivery (груз берётся на базе).
- Сервис, фото, подписать документы без груза = service.
- Времена: «к 9.00» → tw_end 09:00; «до 11 часов» → tw_end 11:00; «после 14» → tw_start 14:00.
  Если окна нет — {window_start}–{window_end}. Формат строго ЧЧ:ММ.
- «по согласованию» → requires_approval=true. «срочно», приоритет «Высокий» → urgent=true.
- «грузим 2 машины» → vehicles_needed=2 (точка будет на двух машинах), иначе 1.
- Вес/объём: если в строках заказа есть — суммируй; если нет — оцени консервативно по названию товара
  и напиши warning «вес оценён». Бумага, электроника, мебель, химия в таре → needs_closed_body=true.
  Европаллета → needs_europallet=true.
- Если в тексте есть координаты (55.56, 38.04) — перенеси в lat/lon. Сам координаты НЕ выдумывай, ставь null.
- vehicle_ids / driver_ids заполняй, только если задача явно привязана к машине/водителю, иначе пустой список.
- task — одна-две короткие фразы в повелительном наклонении для водителя. contact — «Имя, +7…».
- title — короткое имя точки: «КОМУС (Химки)».
- Ничего не выдумывай: неясное → warning, вопрос логисту → questions.

{fleet}
"""


def _normalize_schema(fleet: dict) -> dict:
    vids = [v["id"] for v in fleet["vehicles"]]
    dids = [d["id"] for d in fleet["drivers"]]
    s_str, s_num = {"type": "string"}, {"type": "number"}
    stop = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "local_id": s_str, "task_id": s_str,
            "kind": {"type": "string", "enum": ["pickup", "delivery", "service"]},
            "title": s_str, "address": s_str,
            "lat": {"type": ["number", "null"]}, "lon": {"type": ["number", "null"]},
            "tw_start": s_str, "tw_end": s_str, "service_min": {"type": "integer"},
            "weight_kg": s_num, "volume_m3": s_num,
            "needs_closed_body": {"type": "boolean"}, "needs_europallet": {"type": "boolean"},
            "urgent": {"type": "boolean"}, "requires_approval": {"type": "boolean"},
            "vehicles_needed": {"type": "integer"},
            "pair_with_local_id": {"type": ["string", "null"]},
            "vehicle_ids": {"type": "array", "items": {"type": "string", "enum": vids}},
            "driver_ids": {"type": "array", "items": {"type": "string", "enum": dids}},
            "task": s_str, "contact": s_str, "invoice_no": s_str, "order_no": s_str,
            "warnings": {"type": "array", "items": s_str},
        },
    }
    stop["required"] = list(stop["properties"])
    return {"type": "object", "additionalProperties": False, "required": ["stops", "questions"],
            "properties": {"stops": {"type": "array", "items": stop},
                           "questions": {"type": "array", "items": s_str}}}


def normalize(tasks: list[RawTask], fleet: dict) -> tuple[list[Stop], list[str]]:
    if not settings.llm_enabled:
        return _normalize_fallback(tasks, fleet)
    sh = fleet["shift"]
    resp = client().responses.create(
        model=settings.model,
        instructions=NORMALIZE_PROMPT.format(window_start=sh["default_window"][0], window_end=sh["default_window"][1],
                                             fleet=_fleet_brief(fleet)),
        input=json.dumps([t.model_dump(exclude_defaults=True) for t in tasks], ensure_ascii=False),
        text={"format": {"type": "json_schema", "name": "route_stops", "strict": True,
                         "schema": _normalize_schema(fleet)}},
        reasoning={"effort": "low"},
    )
    data = json.loads(resp.output_text)
    known = {t.id for t in tasks}
    items = [it for it in data["stops"] if it["task_id"] in known]   # выдуманные заявки отбрасываем
    stops, questions = _materialize(items, sh), data["questions"]
    # заявки, которые модель пропустила, не теряем: разбираем правилами и помечаем
    missed = [t for t in tasks if t.id not in {s.task_id for s in stops}]
    if missed:
        extra, _ = _normalize_fallback(missed, fleet)
        for s in extra:
            s.warnings.append("ИИ пропустил эту заявку — разобрана правилами, проверьте")
        stops += extra
    return stops, questions


def _sanitize(it: dict, shift: dict) -> dict:
    """Модель может ошибиться в формате — чиним или откатываем к умолчанию с предупреждением,
    чтобы одна кривая строка не роняла расчёт всего дня."""
    it = dict(it, warnings=list(it.get("warnings") or []))
    d0, d1 = shift["default_window"]
    a, b = parse_hm(it.get("tw_start")), parse_hm(it.get("tw_end"))
    if a is None or b is None or a >= b:
        it["warnings"].append(f"окно «{it.get('tw_start')}–{it.get('tw_end')}» не распознано — поставлено {d0}–{d1}")
        a, b = d0, d1
    it["tw_start"], it["tw_end"] = a, b
    if (it.get("lat") is not None or it.get("lon") is not None) and not in_region(it.get("lat"), it.get("lon")):
        it["warnings"].append("координаты вне Москвы и области — адрес будет найден заново")
        it["lat"] = it["lon"] = None
    it["service_min"] = min(max(int(it.get("service_min") or shift["default_service_min"]), 5), 240)
    it["weight_kg"] = max(float(it.get("weight_kg") or 0), 0.0)
    it["volume_m3"] = max(float(it.get("volume_m3") or 0), 0.0)
    it["vehicles_needed"] = min(max(int(it.get("vehicles_needed") or 1), 1), 4)
    return it


def _materialize(items: list[dict], shift: dict) -> list[Stop]:
    """local_id → стабильные id, «нужно N машин» → N копий с distinct_group."""
    items = [_sanitize(it, shift) for it in items]
    ids = {(it["task_id"], it["local_id"]): f"{it['task_id']}:{it['local_id']}" for it in items}
    out: list[Stop] = []
    for it in items:
        n = max(1, it.get("vehicles_needed", 1))
        for k in range(n):
            key = (it["task_id"], it["local_id"])
            sid = ids[key] + (f"#{k + 1}" if n > 1 else "")
            out.append(Stop(
                id=sid, task_id=it["task_id"], kind=it["kind"], title=it["title"], address=it["address"],
                lat=it["lat"], lon=it["lon"], tw_start=it["tw_start"], tw_end=it["tw_end"],
                service_min=it["service_min"] or shift["default_service_min"],
                weight_kg=it["weight_kg"], volume_m3=it["volume_m3"],
                needs_closed_body=it["needs_closed_body"], needs_europallet=it["needs_europallet"],
                urgent=it["urgent"], requires_approval=it["requires_approval"],
                pair_with=ids.get((it["task_id"], it["pair_with_local_id"])) if it["pair_with_local_id"] else None,
                distinct_group=ids[key] if n > 1 else None,
                vehicle_only=it["vehicle_ids"], driver_only=it["driver_ids"],
                task=it["task"], contact=it["contact"], invoice_no=it["invoice_no"], order_no=it["order_no"],
                warnings=it["warnings"]))
    return out


def _normalize_fallback(tasks: list[RawTask], fleet: dict) -> tuple[list[Stop], list[str]]:
    """Без ключа OpenAI: простые правила, чтобы прототип работал end-to-end."""
    sh, items = fleet["shift"], []
    for t in tasks:
        text = t.text
        warn = ["разобрано без ИИ (нет OPENAI_API_KEY)"]
        if t.counterparty:
            addrs = t.counterparty.addresses
            pref = [a for a in addrs if re.search(r"склад|фактич", a.kind, re.I)] or addrs
            address = pref[0].text if pref else ""
            if len(pref) > 1:
                warn.append(f"у КА {len(pref)} адреса — выбран первый")
            contact = ", ".join(t.counterparty.contacts[:1] + t.counterparty.phones[:1])
            title = t.counterparty.name
            weight = sum((l.weight_kg or 0) * 1 for l in t.lines)
            volume = sum((l.volume_m3 or 0) for l in t.lines)
        else:
            address, contact, title, weight, volume = text, "", text[:30], 0, 0
        m_end = re.search(r"(?:к|до)\s*(\d{1,2})[.:]?(\d{2})?\s*(?:час|утра)?", text)
        items.append({
            "local_id": "1", "task_id": t.id, "kind": "delivery" if re.search(r"выгруз|отвез", text, re.I) else "pickup",
            "title": title, "address": address, "lat": None, "lon": None,
            "tw_start": sh["default_window"][0],
            "tw_end": f"{int(m_end.group(1)):02d}:{m_end.group(2) or '00'}" if m_end else sh["default_window"][1],
            "service_min": sh["default_service_min"], "weight_kg": weight, "volume_m3": volume,
            "needs_closed_body": False, "needs_europallet": False,
            "urgent": t.priority.lower() in ("высокий", "срочно"),
            "requires_approval": "согласован" in text.lower(),
            "vehicles_needed": int(m.group(1)) if (m := re.search(r"(\d)\s*машин", text)) else 1,
            "pair_with_local_id": None, "vehicle_ids": [], "driver_ids": [],
            "task": text or f"Забрать товар по заказу {t.order_no}", "contact": contact,
            "invoice_no": t.invoice_no, "order_no": t.order_no, "warnings": warn})
    return _materialize(items, sh), []


# ---------------------------------------------------------------- 2. Агент правок плана

AGENT_PROMPT = """Ты — ассистент логиста. Логист пишет правки к плану маршрутов на день обычным языком.
Выполняй их через инструменты, а не на словах. Порядок работы:
1) get_plan, если не уверен в id точек/водителей;
2) правки: move_stop / reorder_route / update_stop / optimize_order / set_driver / set_odometer / replan;
3) каждый инструмент возвращает нарушения (опоздания, перегруз, конец смены) — если они есть,
   скажи о них логисту и предложи вариант, но решение за ним.
Отвечай по-русски, коротко: что сделал и что стоит проверить. Не придумывай точки и водителей.

{fleet}
"""

AGENT_TOOLS = [
    {"type": "function", "name": "get_plan", "strict": True,
     "description": "Текущий план: маршруты по водителям с id точек, ETA, нераспределённые точки.",
     "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}},
    {"type": "function", "name": "move_stop", "strict": True,
     "description": "Перенести точку другому водителю (или снять с маршрута, driver_id=null). "
                    "Точка закрепляется за водителем (locked) и не будет перенесена при replan.",
     "parameters": {"type": "object", "additionalProperties": False, "required": ["stop_id", "driver_id", "position"],
                    "properties": {"stop_id": {"type": "string"},
                                   "driver_id": {"type": ["string", "null"]},
                                   "position": {"type": ["integer", "null"],
                                                "description": "1-based позиция в маршруте; null — оптимально"}}}},
    {"type": "function", "name": "reorder_route", "strict": True,
     "description": "Задать порядок точек в маршруте водителя. Нужно перечислить все его точки.",
     "parameters": {"type": "object", "additionalProperties": False, "required": ["driver_id", "stop_ids"],
                    "properties": {"driver_id": {"type": "string"},
                                   "stop_ids": {"type": "array", "items": {"type": "string"}}}}},
    {"type": "function", "name": "update_stop", "strict": True,
     "description": "Изменить поля точки. null — не менять.",
     "parameters": {"type": "object", "additionalProperties": False,
                    "required": ["stop_id", "address", "tw_start", "tw_end", "urgent", "task", "contact", "service_min"],
                    "properties": {"stop_id": {"type": "string"},
                                   "address": {"type": ["string", "null"], "description": "новый адрес — будет геокодирован"},
                                   "tw_start": {"type": ["string", "null"]}, "tw_end": {"type": ["string", "null"]},
                                   "urgent": {"type": ["boolean", "null"]}, "task": {"type": ["string", "null"]},
                                   "contact": {"type": ["string", "null"]},
                                   "service_min": {"type": ["integer", "null"]}}}},
    {"type": "function", "name": "optimize_order", "strict": True,
     "description": "Переставить точки маршрута водителя в оптимальный порядок (точное решение задачи "
                    "коммивояжёра с окнами времени). Состав точек не меняется.",
     "parameters": {"type": "object", "additionalProperties": False, "required": ["driver_id"],
                    "properties": {"driver_id": {"type": "string"}}}},
    {"type": "function", "name": "set_driver", "strict": True,
     "description": "Отметить, что водитель выходит на эту дату вне графика (works=true) или не выходит "
                    "(works=false), и пересчитать план.",
     "parameters": {"type": "object", "additionalProperties": False, "required": ["driver_id", "works"],
                    "properties": {"driver_id": {"type": "string"}, "works": {"type": "boolean"}}}},
    {"type": "function", "name": "set_odometer", "strict": True,
     "description": "Записать пробег машины водителя на начало дня, км.",
     "parameters": {"type": "object", "additionalProperties": False, "required": ["driver_id", "km"],
                    "properties": {"driver_id": {"type": "string"}, "km": {"type": ["integer", "null"]}}}},
    {"type": "function", "name": "replan", "strict": True,
     "description": "Пересобрать план солвером с учётом закреплённых точек и недоступных машин.",
     "parameters": {"type": "object", "additionalProperties": False, "required": ["unavailable_vehicles"],
                    "properties": {"unavailable_vehicles": {"type": "array", "items": {"type": "string"}}}}},
]


class PlanAgent:
    def __init__(self, editor):
        self.editor = editor   # pipeline.PlanEditor: реализует инструменты

    def run(self, message: str, fleet: dict, max_steps: int = 8) -> str:
        if not settings.llm_enabled:
            return "ИИ-помощник недоступен: не задан OPENAI_API_KEY. Правки можно делать вручную — перетаскиванием и кнопками на карточках."
        resp = client().responses.create(
            model=settings.model, instructions=AGENT_PROMPT.format(fleet=_fleet_brief(fleet)),
            input=message, tools=AGENT_TOOLS, reasoning={"effort": "medium"})
        for _ in range(max_steps):
            calls = [o for o in resp.output if o.type == "function_call"]
            if not calls:
                return resp.output_text
            outputs = []
            for c in calls:
                try:
                    result = self.editor.call(c.name, json.loads(c.arguments or "{}"))
                except Exception as e:   # ошибку отдаём модели — пусть исправится
                    result = {"error": str(e)}
                outputs.append({"type": "function_call_output", "call_id": c.call_id,
                                "output": json.dumps(result, ensure_ascii=False)})
            resp = client().responses.create(
                model=settings.model, instructions=AGENT_PROMPT.format(fleet=_fleet_brief(fleet)),
                previous_response_id=resp.id, input=outputs, tools=AGENT_TOOLS, reasoning={"effort": "medium"})
        return resp.output_text or "Остановился: слишком много шагов. Проверьте план."


# ---------------------------------------------------------------- 3. Пояснение к плану

def explain(plan: Plan, fleet: dict) -> str:
    if not settings.llm_enabled:
        return ""
    resp = client().responses.create(
        model=settings.model_fast,
        instructions="Ты логист. По плану маршрутов в JSON напиши логисту 3–6 коротких пунктов по-русски: "
                     "кто куда едет и почему так, что не влезло и почему, что требует согласования, "
                     "риски по времени. Без воды и без повтора всего плана.\n" + _fleet_brief(fleet),
        input=plan.model_dump_json(exclude={"history"}),
    )
    return resp.output_text
