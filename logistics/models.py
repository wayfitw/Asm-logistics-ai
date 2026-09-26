"""Модели данных: вход из 1С, нормализованная точка маршрута, план."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ---------- Вход: то, что отдаёт 1С по кнопке «Сформировать маршрут» ----------

class CounterpartyAddress(BaseModel):
    kind: str = ""                 # «Фактический», «Склад», «Юридический»…
    text: str
    note: str = ""


class Counterparty(BaseModel):
    id: str
    name: str
    addresses: list[CounterpartyAddress] = []
    phones: list[str] = []
    contacts: list[str] = []       # «Юлия Красавина, главный бухгалтер» и т.п.
    work_hours: str = ""


class OrderLine(BaseModel):
    name: str
    qty: float = 1
    unit: str = "шт"
    weight_kg: float | None = None
    volume_m3: float | None = None


class RawTask(BaseModel):
    """Заявка: оплаченный заказ поставщику из 1С или ручная (срочная) задача логиста."""
    id: str
    source: Literal["1c", "manual"] = "1c"
    order_no: str = ""
    invoice_no: str = ""           # номер счёта для водителя
    amount: str = ""
    counterparty: Counterparty | None = None
    warehouse: str = ""            # склад получателя в 1С («Склад АСМ Фрезер»)
    lines: list[OrderLine] = []
    priority: str = ""             # приоритет из 1С
    text: str = ""                 # свободный текст (ручные задачи, комментарий к заказу)


# ---------- Нормализованная точка ----------

class Stop(BaseModel):
    id: str
    task_id: str
    kind: Literal["pickup", "delivery", "service"] = "pickup"
    title: str = ""                # короткое имя точки: «КОМУС (Химки)»
    address: str
    lat: float | None = None
    lon: float | None = None
    tw_start: str = "09:00"
    tw_end: str = "18:00"
    service_min: int = 25
    weight_kg: float = 0
    volume_m3: float = 0
    needs_closed_body: bool = False
    needs_europallet: bool = False
    urgent: bool = False
    requires_approval: bool = False   # «по согласованию» — не отправлять без подтверждения
    pair_with: str | None = None      # delivery ↔ pickup (забрать в А → выгрузить в Б)
    distinct_group: str | None = None # «нужно 2 машины» → копии точки на разных машинах
    vehicle_only: list[str] = []      # точка возможна только на этих машинах
    driver_only: list[str] = []
    task: str = ""                    # чистая формулировка задачи для водителя
    contact: str = ""
    invoice_no: str = ""
    order_no: str = ""
    warnings: list[str] = []
    needs_review: bool = False        # «требует проверки»: есть предупреждения или нет координат
    alt_addresses: list[str] = []     # другие склады/адреса контрагента — логист может выбрать


# ---------- План ----------

class PlannedStop(BaseModel):
    stop_id: str
    eta: str
    etd: str
    locked: bool = False


class Route(BaseModel):
    driver_id: str
    vehicle_id: str
    stops: list[PlannedStop] = []
    distance_km: float = 0
    duration_min: int = 0
    return_eta: str = ""
    load_kg_max: float = 0
    odometer_start: int | None = None   # пробег на начало дня (пока вручную)
    yandex_url: str = ""
    tsp: str = ""                       # итог проверки порядка задачей коммивояжёра (tsp.py)
    tsp_gain_min: int = 0               # сколько минут можно выиграть другим порядком (после ручных правок)


class Plan(BaseModel):
    date: str
    status: Literal["draft", "approved", "sent"] = "draft"
    stops: dict[str, Stop] = {}
    routes: list[Route] = []
    unassigned: list[str] = []
    excluded: list[str] = []                           # сняты логистом с сегодняшнего плана
    unavailable_vehicles: list[str] = []
    driver_overrides: dict[str, bool] = {}             # выход вне графика / отгул на эту дату
    questions: list[str] = []                          # вопросы ИИ к логисту по заявкам
    violations: list[str] = []                         # нарушения жёстких ограничений (checks.py)
    notes: list[str] = Field(default_factory=list)   # пояснения ИИ и предупреждения
    history: list[str] = Field(default_factory=list)  # правки логиста
