"""Рабочее место логиста: FastAPI + одностраничный интерфейс (static/index.html).

Запуск:  python -m uvicorn app:app --reload   →  http://localhost:8000
"""
from __future__ import annotations

import re
import threading
from collections import defaultdict
from datetime import date

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

from logistics import bitrix, db, importer, llm, pipeline, render
from logistics.config import DATA, ROOT, load_fleet, settings
from logistics.models import RawTask

app = FastAPI(title="АСМ · Планирование маршрутов")

# Изменения плана одного дня — строго по очереди: иначе двойной клик или две вкладки
# читают один и тот же план и последний записавший затирает правки другого.
_locks: defaultdict[str, threading.Lock] = defaultdict(threading.Lock)
_DAY_IN_PATH = re.compile(r"^/(?:api/state|api/plan|api/import/1c|driver)/([^/]+)")


@app.middleware("http")
async def check_day(request: Request, call_next):
    """Дата в адресе — только ГГГГ-ММ-ДД и реальная, иначе понятный ответ вместо 500."""
    if m := _DAY_IN_PATH.match(request.url.path):
        try:
            date.fromisoformat(m.group(1))
        except ValueError:
            return JSONResponse({"detail": f"Неверная дата «{m.group(1)}» — нужен формат ГГГГ-ММ-ДД"}, 400)
    return await call_next(request)


@app.exception_handler(ValueError)
async def value_error(_: Request, e: ValueError):
    # сюда попадают ошибки разбора файла, битый JSON и понятные ошибки правок из PlanEditor
    return JSONResponse({"detail": str(e)}, 400)


def _ensure_not_approved(day: str) -> None:
    plan = pipeline.load_plan(day)
    if plan and plan.status != "draft":
        raise HTTPException(409, "План на эту дату уже утверждён. Чтобы пересобрать, сначала верните его в черновик.")


def _parse_drivers(value: str) -> dict[str, bool]:
    """'rybkin:1,ushakov:0' → {'rybkin': True, 'ushakov': False}"""
    out = {}
    for part in filter(None, (x.strip() for x in value.split(","))):
        did, _, flag = part.partition(":")
        out[did] = flag not in ("0", "false", "")
    return out


def _plan_or_404(day: str):
    plan = pipeline.load_plan(day)
    if not plan:
        raise HTTPException(404, "План на эту дату ещё не сформирован")
    return plan


def _fleet_view(day: str, plan=None) -> dict:
    fleet = load_fleet()
    overrides = plan.driver_overrides if plan else None
    working = {d["id"] for d in pipeline.working_drivers(fleet, date.fromisoformat(day), overrides)}
    for d in fleet["drivers"]:
        d["works_today"] = d["id"] in working
    return fleet


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/state/{day}")
def state(day: str):
    plan = pipeline.load_plan(day)
    return {"plan": plan.model_dump() if plan else None, "fleet": _fleet_view(day, plan),
            "llm": settings.llm_enabled, "model": settings.model}


@app.post("/api/plan/{day}/build")
async def build(day: str, file: UploadFile | None = File(None), sample: bool = Form(False),
                unavailable: str = Form(""), drivers: str = Form("")):
    _ensure_not_approved(day)
    if file is not None:
        tasks = importer.parse(file.filename, await file.read())
    elif sample:
        tasks = importer.parse("sample.json", (DATA / "sample_tasks.json").read_bytes())
    else:
        raise HTTPException(400, "Нужен файл с заявками")
    if not tasks:
        raise HTTPException(400, "В файле не найдено заявок")
    unav = [x for x in unavailable.split(",") if x]
    return await run_in_threadpool(_build_locked, day, tasks, unav, _parse_drivers(drivers))


def _build_locked(day: str, tasks: list[RawTask], unav: list[str], drivers: dict[str, bool]) -> dict:
    with _locks[day]:   # солвер и ИИ — синхронные и долгие, поэтому не в event loop
        _ensure_not_approved(day)
        return pipeline.build_plan(day, tasks, unav, drivers).model_dump()


class TaskIn(BaseModel):
    text: str
    priority: str = "Высокий"


@app.post("/api/plan/{day}/tasks")
def add_task(day: str, body: TaskIn):
    if len(body.text.strip()) < 5:
        raise HTTPException(400, "Опишите заявку: адрес, что сделать, к какому времени, контакт")
    with _locks[day]:
        plan = _plan_or_404(day)
        n = sum(1 for s in plan.stops.values() if s.task_id.startswith("U-")) + 1
        ed = pipeline.PlanEditor(plan)
        res = ed.add_tasks([RawTask(id=f"U-{n}", source="manual", text=body.text, priority=body.priority)])
        return {"result": res, "plan": ed.plan.model_dump()}


class EditIn(BaseModel):
    tool: str
    args: dict


@app.post("/api/plan/{day}/edit")
def edit(day: str, body: EditIn):
    with _locks[day]:
        if body.tool not in {"move_stop", "reorder_route", "update_stop", "replan", "optimize_order",
                             "set_odometer", "set_driver"}:
            raise HTTPException(400, "Неизвестная операция")
        ed = pipeline.PlanEditor(_plan_or_404(day))
        try:
            res = ed.call(body.tool, body.args)
        except ValueError as e:
            raise HTTPException(400, str(e))
        except TypeError:
            raise HTTPException(400, f"Неверные параметры для операции {body.tool}")
        return {"result": res, "plan": ed.plan.model_dump()}


class AgentIn(BaseModel):
    message: str


@app.post("/api/plan/{day}/agent")
def agent(day: str, body: AgentIn):
    with _locks[day]:
        ed = pipeline.PlanEditor(_plan_or_404(day), actor="ai")
        try:
            answer = llm.PlanAgent(ed).run(body.message, ed.fleet)
        except Exception as e:
            raise HTTPException(502, f"Ошибка ИИ: {e}")
        db.log(day, "agent", {"message": body.message, "answer": answer})
        return {"answer": answer, "plan": ed.plan.model_dump()}


@app.post("/api/plan/{day}/approve")
def approve(day: str):
    with _locks[day]:
        return pipeline.approve(_plan_or_404(day)).model_dump()


@app.post("/api/plan/{day}/reopen")
def reopen(day: str):
    with _locks[day]:
        plan = _plan_or_404(day)
        plan.status = "draft"
        db.save_plan(plan)
        db.log(day, "reopened", {})
        return plan.model_dump()


@app.post("/api/plan/{day}/send")
def send(day: str):
    with _locks[day]:
        plan = _plan_or_404(day)
        if plan.status == "draft":
            raise HTTPException(400, "Сначала утвердите план")
        res = bitrix.send(plan)
        if any(r["sent"] for r in res):
            plan.status = "sent"
            db.save_plan(plan)
        db.log(day, "sent", res)
        return res


@app.get("/api/plan/{day}/journal")
def journal(day: str):
    return db.journal(day)


@app.get("/driver/{day}/{driver_id}", response_class=HTMLResponse)
def driver(day: str, driver_id: str):
    plan = _plan_or_404(day)
    route = next((r for r in plan.routes if r.driver_id == driver_id), None)
    if not route:
        raise HTTPException(404, "У водителя нет маршрута на эту дату")
    return render.driver_sheet(plan, route)


# Точка входа для 1С (этап после MVP): обработка «Сформировать маршрут» шлёт сюда оплаченные заказы.
@app.post("/api/import/1c/{day}")
def import_1c(day: str, tasks: list[RawTask], x_token: str = Header("")):
    if not settings.import_token or x_token != settings.import_token:
        raise HTTPException(401, "Неверный токен")
    with _locks[day]:
        _ensure_not_approved(day)   # 1С не должна молча затирать утверждённый и отправленный план
        return pipeline.build_plan(day, tasks).model_dump(include={"date", "routes", "unassigned", "violations"})
