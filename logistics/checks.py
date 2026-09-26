"""Детерминированная проверка плана. Вызывается после солвера и после КАЖДОЙ правки (ручной или ИИ-агента).

LLM может ошибиться, солвер можно обойти ручной правкой — поэтому жёсткие ограничения
проверяются здесь, обычным кодом, и показываются логисту до утверждения.
"""
from __future__ import annotations

from collections import Counter
from datetime import date

from .config import driver_works, hm
from .models import Plan


def validate(plan: Plan, fleet: dict) -> list[str]:
    out: list[str] = []
    vehicles = {v["id"]: v for v in fleet["vehicles"]}
    drivers = {d["id"]: d for d in fleet["drivers"]}
    day = date.fromisoformat(plan.date)
    shift_end = hm(fleet["shift"]["end"])

    for vid, n in Counter(r.vehicle_id for r in plan.routes).items():
        if n > 1:
            out.append(f"{vehicles[vid]['name']}: назначена на {n} маршрута")
    for did, n in Counter(r.driver_id for r in plan.routes).items():
        if n > 1:
            out.append(f"{drivers[did]['name']}: {n} маршрута в один день")

    where: dict[str, str] = {}
    loaders: dict[str, set[str]] = {}
    for s in plan.stops.values():
        if s.kind == "pickup" and s.pair_with:
            loaders.setdefault(s.pair_with, set()).add(s.id)
    for r in plan.routes:
        d, v = drivers.get(r.driver_id), vehicles.get(r.vehicle_id)
        if not d or not v:
            out.append(f"Неизвестный водитель/машина: {r.driver_id}/{r.vehicle_id}")
            continue
        who = d["name"].split()[0]
        if r.vehicle_id not in d["vehicles"]:
            out.append(f"{who} не допущен к {v['name']}")
        ov = dict((fleet.get("overrides") or {}).get(plan.date, {}))
        ov.update(plan.driver_overrides)
        if not driver_works(d, day, {plan.date: ov}):
            out.append(f"{who} по графику не работает {plan.date} (нужно согласовать выход)")
        if r.vehicle_id in plan.unavailable_vehicles:
            out.append(f"{v['name']} отмечена недоступной")
        if r.return_eta and hm(r.return_eta) > shift_end:
            out.append(f"{who}: возврат на базу {r.return_eta}, смена до {fleet['shift']['end']}")

        load_w = load_v = 0.0
        seen: set[str] = set()
        for p in r.stops:
            s = plan.stops.get(p.stop_id)
            if not s:
                out.append(f"{who}: точка {p.stop_id} отсутствует в заявках")
                continue
            where[s.id] = r.driver_id
            name = s.title or s.address
            sign = 1 if s.kind == "pickup" else -1 if s.kind == "delivery" else 0
            load_w += sign * s.weight_kg
            load_v += sign * s.volume_m3
            if load_w > v["capacity_kg"] + 1e-6:
                out.append(f"{who}: перегруз после «{name}» — {load_w:.0f} кг при лимите {v['capacity_kg']} кг")
            if load_v > v["volume_m3"] + 1e-6:
                out.append(f"{who}: не влезает по объёму после «{name}» — {load_v:.1f} м³ из {v['volume_m3']} м³")
            if s.needs_closed_body and not v.get("closed_body", True):
                out.append(f"{who}: «{name}» нужен закрытый кузов, а {v['name']} — открытый борт")
            if s.needs_europallet and not v.get("fits_europallet", True):
                out.append(f"{who}: «{name}» на европаллете не входит в {v['name']}")
            if s.vehicle_only and r.vehicle_id not in s.vehicle_only:
                out.append(f"{who}: «{name}» только для {', '.join(s.vehicle_only)}")
            if s.driver_only and r.driver_id not in s.driver_only:
                out.append(f"{who}: «{name}» закреплена за {', '.join(s.driver_only)}")
            if p.eta and hm(p.eta) > hm(s.tw_end):
                out.append(f"{who}: «{name}» — прибытие {p.eta}, окно до {s.tw_end}")
            # пара всегда хранится на погрузке: pickup.pair_with = id выгрузки
            # (для доставки «со склада» погрузка — виртуальная точка load:<id> на базе)
            if s.kind == "delivery" and not (loaders.get(s.id, set()) & seen):
                out.append(f"{who}: «{name}» — выгрузка без погрузки перед ней в этом маршруте")
            seen.add(s.id)

    # пары «забрать→отвезти» — на одной машине
    for s in plan.stops.values():
        if s.kind == "pickup" and s.pair_with and s.id in where and s.pair_with in where \
                and where[s.id] != where[s.pair_with]:
            out.append(f"«{s.title}»: погрузка и выгрузка у разных водителей")
    # «нужно 2 машины» — копии на разных машинах
    groups: dict[str, list[str]] = {}
    for s in plan.stops.values():
        if s.distinct_group and s.id in where:
            groups.setdefault(s.distinct_group, []).append(where[s.id])
    for g, ds in groups.items():
        if len(ds) != len(set(ds)):
            out.append(f"«{plan.stops[g + '#1'].title if g + '#1' in plan.stops else g}»: нужны разные машины, а стоит на одной")

    for sid in plan.unassigned:
        s = plan.stops.get(sid)
        if s and s.urgent:
            out.append(f"Срочная заявка «{s.title or s.address}» не распределена")
    return list(dict.fromkeys(out))
