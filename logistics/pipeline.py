"""Сборка плана: заявки → ИИ-нормализация → геокодирование → кто работает → солвер → проверки → план."""
from __future__ import annotations

import json
from datetime import date

from . import checks, db, geo, llm, solver, tsp
from .config import driver_works, fmt_hm, hm, in_region, load_fleet, parse_hm
from .models import Plan, PlannedStop, RawTask, Route, Stop

load_plan = db.load_plan


def merged_overrides(fleet: dict, day: str, plan_overrides: dict | None = None) -> dict:
    """Переопределения графика: из fleet.yaml на дату + отметки логиста в плане (приоритетнее)."""
    base = dict((fleet.get("overrides") or {}).get(day, {}))
    base.update(plan_overrides or {})
    return {day: base}


def working_drivers(fleet: dict, day: date, plan_overrides: dict | None = None) -> list[dict]:
    ov = merged_overrides(fleet, day.isoformat(), plan_overrides)
    return [d for d in fleet["drivers"] if driver_works(d, day, ov)]


def prepare_stops(tasks: list[RawTask], fleet: dict) -> tuple[list[Stop], list[str]]:
    stops, questions = llm.normalize(tasks, fleet)
    by_task = {t.id: t for t in tasks}
    for s in stops:
        t = by_task.get(s.task_id)
        if t and t.counterparty and s.kind == "pickup":
            # все известные адреса КА, кроме юридического — чтобы логист мог выбрать другой склад
            s.alt_addresses = [a.text for a in t.counterparty.addresses
                               if a.text != s.address and "юрид" not in a.kind.lower()]
        if s.lat is None or s.lon is None:
            c = geo.geocode(s.address)
            if c and in_region(*c):
                s.lat, s.lon = c
            else:
                s.warnings.append("адрес не найден на карте (или найден не в Москве/МО) — уточните")
        s.needs_review = bool(s.warnings) or s.lat is None or s.requires_approval
    return stops, questions


def recalc_route(plan: Plan, r: Route, fleet: dict) -> list[str]:
    """Пересчёт ETA, км, загрузки и ссылки на карту для заданного порядка точек."""
    ids = [p.stop_id for p in r.stops]
    locked = {p.stop_id for p in r.stops if p.locked}
    base, shift = fleet["base"], fleet["shift"]
    planned, km, end, issues = solver.evaluate(ids, plan.stops, base, shift)
    for p in planned:
        p.locked = p.stop_id in locked
    r.stops, r.distance_km = planned, km
    r.return_eta, r.duration_min = fmt_hm(end), end - hm(shift["start"])
    depot = (base["lat"], base["lon"])
    r.yandex_url = geo.yandex_route_url([depot] + [(plan.stops[i].lat, plan.stops[i].lon) for i in ids] + [depot])
    load = peak = 0.0
    for i in ids:
        s = plan.stops[i]
        load += s.weight_kg if s.kind == "pickup" else -s.weight_kg if s.kind == "delivery" else 0
        peak = max(peak, load)
    r.load_kg_max = peak
    return issues


def tsp_review(plan: Plan, fleet: dict, apply: bool) -> None:
    """Вторая проверка каждого маршрута задачей коммивояжёра.
    apply=True (после автоматического расчёта) — лучший порядок применяется сразу;
    apply=False (после ручных правок) — порядок логиста не трогаем, только подсказываем выигрыш."""
    for r in plan.routes:
        if len(r.stops) < 2:
            r.tsp, r.tsp_gain_min = "порядок проверен (TSP)", 0
            continue
        ids = [p.stop_id for p in r.stops]
        res = tsp.best_order(ids, plan.stops, fleet["base"], fleet["shift"])
        current = hm(r.return_eta) if r.return_eta else 10**9
        gain = current - res.end_min
        if res.method == "infeasible":
            r.tsp, r.tsp_gain_min = "TSP: без нарушений окон этот набор точек не объехать", 0
        elif gain <= 0 or res.order == ids:
            r.tsp, r.tsp_gain_min = "порядок оптимален (проверено TSP)", 0
        elif apply:
            by_id = {p.stop_id: p for p in r.stops}
            r.stops = [by_id[i] for i in res.order]
            recalc_route(plan, r, fleet)
            r.tsp, r.tsp_gain_min = f"порядок улучшен TSP: возврат на {gain} мин раньше", 0
        else:
            r.tsp, r.tsp_gain_min = f"TSP: другой порядок быстрее на {gain} мин", gain


def finalize(plan: Plan, fleet: dict, apply_tsp: bool = False) -> Plan:
    tsp_review(plan, fleet, apply=apply_tsp)
    plan.violations = checks.validate(plan, fleet)
    db.save_plan(plan)
    return plan


def build_plan(day: str, tasks: list[RawTask], unavailable_vehicles: list[str] = (),
               driver_overrides: dict[str, bool] | None = None) -> Plan:
    fleet = load_fleet()
    db.log(day, "import", [t.model_dump(exclude_defaults=True) for t in tasks])
    stops, questions = prepare_stops(tasks, fleet)
    db.log(day, "normalized", {"stops": [s.model_dump() for s in stops], "questions": questions})
    plan = Plan(date=day, stops={s.id: s for s in stops}, questions=questions,
                unavailable_vehicles=list(unavailable_vehicles), driver_overrides=dict(driver_overrides or {}))
    solve_into(plan, fleet)
    try:
        if note := llm.explain(plan, fleet):
            plan.notes.insert(0, note)
    except Exception as e:   # пояснение — не критично
        plan.notes.append(f"⚠ Пояснение ИИ недоступно: {e}")
    finalize(plan, fleet, apply_tsp=True)
    db.log(day, "solved", plan.model_dump(include={"routes", "unassigned", "violations"}))
    return plan


def solve_into(plan: Plan, fleet: dict, locks: dict[str, str] | None = None) -> None:
    drivers = working_drivers(fleet, date.fromisoformat(plan.date), plan.driver_overrides)
    active = [s.model_copy(deep=True) for s in plan.stops.values()
              if not s.id.startswith("load:") and s.id not in plan.excluded]
    unplaceable = [s.id for s in active if s.lat is None]
    active = [s for s in active if s.lat is not None]
    for s in active:
        if locks and s.id in locks:
            s.driver_only = [locks[s.id]]
    res, _ = solver.best_plan(active, drivers, fleet["vehicles"], fleet["base"], fleet["shift"],
                              set(plan.unavailable_vehicles))
    for x in res.extra_stops:
        plan.stops[x.id] = x
    # введённый логистом пробег на начало дня не должен пропадать при пересчёте
    odo = {(r.driver_id, r.vehicle_id): r.odometer_start for r in plan.routes if r.odometer_start is not None}
    for r in res.routes:
        r.odometer_start = odo.get((r.driver_id, r.vehicle_id))
    for r in res.routes:
        for p in r.stops:
            p.locked = bool(locks and locks.get(p.stop_id) == r.driver_id)
    plan.routes, plan.unassigned = res.routes, res.unassigned + unplaceable
    plan.notes = [n for n in plan.notes if not n.startswith("⚠")]
    if not drivers:
        plan.notes.append("⚠ На эту дату по графику нет водителей — отметьте, кто выходит, в блоке «Машины и водители»")
    else:
        plan.notes += [f"⚠ {n}" for n in res.notes]


class PlanEditor:
    """Правки плана — и для кнопок интерфейса, и как инструменты ИИ-агента.
    Каждая правка пересчитывает ETA, прогоняет проверки, сохраняет план и пишет в журнал."""

    def __init__(self, plan: Plan, actor: str = "logist"):
        self.plan, self.fleet, self.actor = plan, load_fleet(), actor

    # --- helpers
    def _route(self, driver_id: str) -> Route:
        for r in self.plan.routes:
            if r.driver_id == driver_id:
                return r
        d = next((d for d in self.fleet["drivers"] if d["id"] == driver_id), None)
        if not d:
            raise ValueError(f"нет водителя {driver_id}")
        used = {r.vehicle_id for r in self.plan.routes} | set(self.plan.unavailable_vehicles)
        vid = next((v for v in d["vehicles"] if v not in used), None)
        if not vid:
            raise ValueError(f"у {driver_id} нет свободной машины")
        r = Route(driver_id=driver_id, vehicle_id=vid)
        self.plan.routes.append(r)
        return r

    def _linked(self, stop_id: str) -> list[str]:
        """Точка + её пара (забрать↔отвезти, погрузка на базе) — двигаются вместе, погрузка первой."""
        st = self.plan.stops
        ids = {stop_id}
        if st[stop_id].pair_with:
            ids.add(st[stop_id].pair_with)
        ids |= {s.id for s in st.values() if s.pair_with in ids}
        return sorted(ids, key=lambda i: 0 if st[i].kind == "pickup" else 1)

    def _recalc(self, r: Route) -> list[str]:
        return recalc_route(self.plan, r, self.fleet)

    def _detach(self, ids: list[str]) -> None:
        for r in self.plan.routes:
            before = len(r.stops)
            r.stops = [p for p in r.stops if p.stop_id not in ids]
            if r.stops and len(r.stops) != before:
                self._recalc(r)   # маршрут, из которого забрали точку, тоже пересчитываем
        self.plan.unassigned = [u for u in self.plan.unassigned if u not in ids]
        self.plan.excluded = [u for u in self.plan.excluded if u not in ids]

    # --- entry point
    def call(self, name: str, args: dict) -> dict:
        if self.plan.status != "draft" and name not in ("get_plan", "set_odometer"):
            raise ValueError("план уже утверждён — сначала вернуть в черновик")
        res = getattr(self, f"t_{name}")(**args)
        if name != "get_plan":
            self.plan.routes = [r for r in self.plan.routes if r.stops]
            self.plan.history.append(f"{self.actor}: {name} {json.dumps(args, ensure_ascii=False)}")
            finalize(self.plan, self.fleet, apply_tsp=name in ("replan", "set_driver"))
            db.log(self.plan.date, "edit", {"actor": self.actor, "tool": name, "args": args})
            res["violations"] = self.plan.violations
        return res

    # --- tools
    def t_get_plan(self) -> dict:
        st = self.plan.stops
        return {
            "date": self.plan.date,
            "routes": [{"driver_id": r.driver_id, "vehicle_id": r.vehicle_id, "return": r.return_eta, "km": r.distance_km,
                        "stops": [{"id": p.stop_id, "eta": p.eta, "title": st[p.stop_id].title, "kind": st[p.stop_id].kind,
                                   "window": f"{st[p.stop_id].tw_start}-{st[p.stop_id].tw_end}",
                                   "kg": st[p.stop_id].weight_kg, "locked": p.locked}
                                  for p in r.stops]} for r in self.plan.routes],
            "unassigned": [{"id": i, "title": st[i].title, "address": st[i].address, "warnings": st[i].warnings}
                           for i in self.plan.unassigned],
            "excluded": self.plan.excluded,
            "violations": self.plan.violations,
            "working_drivers": [d["id"] for d in working_drivers(self.fleet, date.fromisoformat(self.plan.date),
                                                                 self.plan.driver_overrides)],
        }

    def t_move_stop(self, stop_id: str, driver_id: str | None, position: int | None) -> dict:
        if stop_id not in self.plan.stops:
            raise ValueError(f"нет точки {stop_id}")
        ids = self._linked(stop_id)
        self._detach(ids)
        if driver_id is None:
            self.plan.excluded += [i for i in ids if not i.startswith("load:")]
            return {"ok": True, "excluded": ids}
        r = self._route(driver_id)
        new = [PlannedStop(stop_id=i, eta="", etd="", locked=True) for i in ids]
        if position is not None:
            k = max(0, min(position - 1, len(r.stops)))
            r.stops[k:k] = new
            return {"ok": True, "timing": self._recalc(r)}
        # лучшая позиция вставки — меньше нарушений, потом раньше возврат
        base, best = list(r.stops), None
        for k in range(len(base) + 1):
            r.stops = base[:k] + new + base[k:]
            issues = self._recalc(r)
            score = (len(issues), r.return_eta)
            if best is None or score < best[0]:
                best = (score, list(r.stops))
        r.stops = best[1]
        return {"ok": True, "timing": self._recalc(r)}

    def t_reorder_route(self, driver_id: str, stop_ids: list[str]) -> dict:
        r = self._route(driver_id)
        if sorted(stop_ids) != sorted(p.stop_id for p in r.stops):
            raise ValueError("нужно перечислить ровно те же точки, что в маршруте")
        by_id = {p.stop_id: p for p in r.stops}
        r.stops = [by_id[i] for i in stop_ids]
        return {"ok": True, "timing": self._recalc(r)}

    def t_update_stop(self, stop_id: str, address: str | None = None, tw_start: str | None = None,
                      tw_end: str | None = None, urgent: bool | None = None, task: str | None = None,
                      contact: str | None = None, service_min: int | None = None) -> dict:
        if stop_id not in self.plan.stops:
            raise ValueError(f"нет точки {stop_id}")
        s = self.plan.stops[stop_id]
        start = parse_hm(tw_start) if tw_start is not None else s.tw_start
        end = parse_hm(tw_end) if tw_end is not None else s.tw_end
        if start is None or end is None:
            raise ValueError("время укажите как ЧЧ:ММ, например 09:30")
        if start >= end:
            raise ValueError(f"окно пустое: начало {start} не раньше конца {end}")
        if service_min is not None and not 0 < int(service_min) <= 480:
            raise ValueError("время на точке — от 1 до 480 минут")
        s.tw_start, s.tw_end = start, end
        if urgent is not None:
            s.urgent = bool(urgent)
        if task is not None:
            s.task = task.strip()
        if contact is not None:
            s.contact = contact.strip()
        if service_min is not None:
            s.service_min = int(service_min)
        if address is not None and address.strip() and address.strip() != s.address:
            old = s.address
            c = geo.geocode(address.strip())
            if not c or not in_region(*c):
                raise ValueError(f"адрес «{address}» не найден на карте Москвы/МО — уточните написание")
            s.address, (s.lat, s.lon) = address.strip(), c
            s.alt_addresses = [x for x in [old] + s.alt_addresses if x != s.address]
            s.warnings = [w for w in s.warnings if "адрес" not in w]
            s.needs_review = bool(s.warnings) or s.requires_approval
        r = next((r for r in self.plan.routes if any(p.stop_id == stop_id for p in r.stops)), None)
        return {"ok": True, "timing": self._recalc(r) if r else []}

    def t_set_odometer(self, driver_id: str, km: int | None) -> dict:
        """Пробег машины на начало дня (ТЗ: пока вручную). Можно и после утверждения."""
        r = next((r for r in self.plan.routes if r.driver_id == driver_id), None)
        if not r:
            raise ValueError("у водителя нет маршрута на эту дату")
        if km is not None and not 0 <= int(km) < 5_000_000:
            raise ValueError("пробег — целое число километров")
        r.odometer_start = None if km is None else int(km)
        return {"ok": True}

    def t_set_driver(self, driver_id: str, works: bool) -> dict:
        """Выход вне графика (например, в субботу по согласованию) или отгул — и пересчёт."""
        if not any(d["id"] == driver_id for d in self.fleet["drivers"]):
            raise ValueError(f"нет водителя {driver_id}")
        self.plan.driver_overrides[driver_id] = bool(works)
        if not works:   # закрепления за невышедшим водителем снимаем
            for r in self.plan.routes:
                if r.driver_id == driver_id:
                    for p in r.stops:
                        p.locked = False
        return self.t_replan(self.plan.unavailable_vehicles)

    def t_optimize_order(self, driver_id: str) -> dict:
        """Переставить точки маршрута в оптимальный порядок (TSP), не меняя состав."""
        r = self._route(driver_id)
        before = r.return_eta
        res = tsp.best_order([p.stop_id for p in r.stops], self.plan.stops, self.fleet["base"], self.fleet["shift"])
        if res.method == "infeasible":
            return {"ok": False, "reason": "без нарушения окон эти точки одному водителю не объехать"}
        by_id = {p.stop_id: p for p in r.stops}
        r.stops = [by_id[i] for i in res.order]
        return {"ok": True, "timing": self._recalc(r), "return_before": before, "return_after": r.return_eta}

    def t_replan(self, unavailable_vehicles: list[str]) -> dict:
        self.plan.unavailable_vehicles = unavailable_vehicles
        locks = {p.stop_id: r.driver_id for r in self.plan.routes for p in r.stops if p.locked}
        solve_into(self.plan, self.fleet, locks)
        return {"ok": True, "unassigned": self.plan.unassigned,
                "routes": {r.driver_id: r.return_eta for r in self.plan.routes}}

    def add_tasks(self, tasks: list[RawTask]) -> dict:
        """Срочная заявка после расчёта: разбираем, добавляем, пересобираем с сохранением закреплённого."""
        db.log(self.plan.date, "import", [t.model_dump(exclude_defaults=True) for t in tasks])
        stops, questions = prepare_stops(tasks, self.fleet)
        for s in stops:
            self.plan.stops[s.id] = s
        self.plan.questions += questions
        return self.call("replan", {"unavailable_vehicles": self.plan.unavailable_vehicles}) | {
            "added": [s.id for s in stops]}


def approve(plan: Plan) -> Plan:
    fleet = load_fleet()
    finalize(plan, fleet)
    plan.status = "approved"
    db.save_plan(plan)
    db.log(plan.date, "approved", {"violations": plan.violations, "edits": len(plan.history)})
    return plan
