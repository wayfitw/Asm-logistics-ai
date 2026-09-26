"""Сценарии из плана разработки (§10). Офлайн: без OpenAI и без сети (матрица — по прямой)."""
from __future__ import annotations

import pytest

from logistics import checks, db, geo, llm, pipeline, solver
from logistics.config import load_fleet
from logistics.models import Plan, RawTask, Stop

FLEET = load_fleet()
# Точки по Москве/области (реальные координаты)
P = {
    "zhukovsky": (55.5697, 38.0437), "kolomenskaya": (55.6783, 37.6931), "balashikha": (55.8050, 37.9106),
    "khimki": (55.9191, 37.4195), "ryazansky": (55.7283, 37.7409), "kabelnaya": (55.7451, 37.7158),
    "reutov": (55.7646, 37.8471), "lyubertsy": (55.6666, 37.8890), "enthusiastov": (55.7943, 37.9267),
}


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    def fake_matrix(points):
        kms = [[round(geo._haversine_km(a, b) * 1.35, 1) for b in points] for a in points]
        return [[round(k / 25 * 60) for k in row] for row in kms], kms
    monkeypatch.setattr(geo, "matrix", fake_matrix)
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "t.db")
    monkeypatch.setenv("SOLVER_TIME_LIMIT", "1")
    monkeypatch.setattr(llm.settings.__class__, "llm_enabled", property(lambda self: False))


def stop(sid, where, kind="pickup", **kw) -> Stop:
    lat, lon = P[where]
    return Stop(id=sid, task_id=sid, kind=kind, title=sid, address=where, lat=lat, lon=lon, **kw)


def drivers(*ids):
    return [d for d in FLEET["drivers"] if d["id"] in ids]


def solve(stops, drv=("kolesnikov", "rybkin", "ushakov"), unavailable=()):
    res, _ = solver.best_plan(stops, drivers(*drv), FLEET["vehicles"], FLEET["base"], FLEET["shift"], set(unavailable))
    return res


def where(res, sid):
    return next(((r.driver_id, r.vehicle_id) for r in res.routes for p in r.stops if p.stop_id == sid), None)


def make_plan(stops, res, day="2026-09-24") -> Plan:
    plan = Plan(date=day, stops={s.id: s for s in stops + res.extra_stops}, routes=res.routes, unassigned=res.unassigned)
    plan.violations = checks.validate(plan, FLEET)
    return plan


# --- сценарии -------------------------------------------------------------

def test_urgent_short_window_is_served_in_time():
    stops = [stop("urgent", "kolomenskaya", urgent=True, tw_end="09:00"),
             stop("a", "khimki"), stop("b", "reutov"), stop("c", "lyubertsy")]
    res = solve(stops)
    assert where(res, "urgent")
    eta = next(p.eta for r in res.routes for p in r.stops if p.stop_id == "urgent")
    assert eta <= "09:00"


def test_cargo_heavier_than_staria_goes_to_foton_or_gazel():
    res = solve([stop("heavy", "khimki", weight_kg=1300)])
    assert where(res, "heavy")[1] in {"foton", "gazel"}


def test_europallet_never_in_staria():
    res = solve([stop("pal", "enthusiastov", weight_kg=180, needs_europallet=True)])
    assert where(res, "pal")[1] not in {"staria_1", "staria_2"}


def test_closed_body_never_on_open_gazel():
    stops = [stop(f"paper{i}", "khimki", weight_kg=900, needs_closed_body=True) for i in range(3)]
    res = solve(stops, drv=("rybkin", "ushakov"))
    assert all(where(res, s.id) is None or where(res, s.id)[1] != "gazel" for s in stops)


def test_driver_vehicle_compatibility_holds():
    stops = [stop(k, k) for k in ("khimki", "reutov", "lyubertsy", "kabelnaya", "balashikha")]
    res = solve(stops)
    allowed = {d["id"]: set(d["vehicles"]) for d in FLEET["drivers"]}
    assert all(r.vehicle_id in allowed[r.driver_id] for r in res.routes)
    assert not make_plan(stops, res).violations


def test_reserve_gazel_not_used_when_not_needed():
    res = solve([stop("a", "khimki", weight_kg=50), stop("b", "reutov", weight_kg=50)])
    assert all(r.vehicle_id != "gazel" for r in res.routes)


def test_same_area_orders_grouped_on_one_route():
    stops = [stop("r1", "reutov"), stop("r2", "reutov", tw_start="10:00"), stop("r3", "balashikha")]
    res = solve(stops)
    assert len({where(res, s.id)[0] for s in stops}) == 1


def test_pickup_then_delivery_on_same_vehicle_in_order():
    stops = [stop("oil", "kolomenskaya", pair_with="unload", weight_kg=200, tw_end="09:30"),
             stop("unload", "balashikha", kind="delivery", weight_kg=200)]
    res = solve(stops)
    route = next(r for r in res.routes if any(p.stop_id == "oil" for p in r.stops))
    ids = [p.stop_id for p in route.stops]
    assert ids.index("oil") < ids.index("unload")


def test_delivery_from_base_gets_load_stop_first():
    stops = [stop("to_reutov", "reutov", kind="delivery", weight_kg=40)]
    res = solve(stops)
    route = res.routes[0]
    assert [p.stop_id for p in route.stops] == ["load:to_reutov", "to_reutov"]
    assert not make_plan(stops, res).violations


def test_two_vehicles_needed_go_on_different_vehicles():
    stops = [stop("cont#1", "zhukovsky", distinct_group="cont", tw_end="11:00"),
             stop("cont#2", "zhukovsky", distinct_group="cont", tw_end="11:00")]
    res = solve(stops)
    assert where(res, "cont#1")[0] != where(res, "cont#2")[0]


def test_driver_day_off_is_respected():
    # 2026-09-26 — суббота: 5/2 не работают; Ушаков (2/2 с 23.09) — 26.09 выходной тоже
    day = pipeline.date.fromisoformat("2026-09-26")
    assert pipeline.working_drivers(FLEET, day) == []
    day = pipeline.date.fromisoformat("2026-09-27")   # воскресенье, у Ушакова рабочий день цикла
    assert [d["id"] for d in pipeline.working_drivers(FLEET, day)] == ["ushakov"]


def test_vehicle_unavailable_is_not_used():
    res = solve([stop("a", "khimki"), stop("b", "lyubertsy")], unavailable={"staria_1"})
    assert all(r.vehicle_id != "staria_1" for r in res.routes)
    assert all(r.driver_id != "kolesnikov" for r in res.routes)   # у Колесникова другой машины нет


def test_capacity_violation_detected_after_manual_edit():
    stops = [stop("h1", "khimki", weight_kg=700), stop("h2", "khimki", weight_kg=700)]
    res = solve(stops)
    plan = make_plan(stops, res)
    db.save_plan(plan)
    ed = pipeline.PlanEditor(plan)
    ed.call("move_stop", {"stop_id": "h1", "driver_id": "kolesnikov", "position": None})
    ed.call("move_stop", {"stop_id": "h2", "driver_id": "kolesnikov", "position": None})
    assert any("перегруз" in v for v in ed.plan.violations)


def test_manual_move_locks_and_survives_replan():
    stops = [stop(k, k) for k in ("khimki", "reutov", "lyubertsy", "kabelnaya")]
    res = solve(stops)
    plan = make_plan(stops, res)
    db.save_plan(plan)
    ed = pipeline.PlanEditor(plan)
    ed.call("move_stop", {"stop_id": "khimki", "driver_id": "ushakov", "position": None})
    ed.call("replan", {"unavailable_vehicles": []})
    assert where(ed.plan, "khimki")[0] == "ushakov"
    # и донор, и получатель пересчитаны: время возврата не раньше последней точки
    for r in ed.plan.routes:
        assert r.return_eta >= r.stops[-1].etd


def test_urgent_task_added_after_initial_plan():
    stops = [stop("a", "khimki"), stop("b", "reutov")]
    plan = make_plan(stops, solve(stops))
    db.save_plan(plan)
    ed = pipeline.PlanEditor(plan)
    # без ИИ — ручная задача с координатами в тексте
    res = ed.add_tasks([RawTask(id="U-1", source="manual", priority="Высокий",
                                text="Жуковский, 55.569692, 38.043729 забрать до 12")])
    assert res["added"] and where(ed.plan, res["added"][0])


def test_approved_plan_is_frozen():
    stops = [stop("a", "khimki")]
    plan = make_plan(stops, solve(stops))
    pipeline.approve(plan)
    with pytest.raises(ValueError):
        pipeline.PlanEditor(plan).call("replan", {"unavailable_vehicles": []})


def test_multiple_warehouses_flagged_for_review():
    from logistics.models import Counterparty, CounterpartyAddress
    t = RawTask(id="O", counterparty=Counterparty(id="k", name="ОКТАНТ", addresses=[
        CounterpartyAddress(kind="Склад", text="Москва, Рязанский проспект, 2"),
        CounterpartyAddress(kind="Склад", text="Люберцы, Котельнический проезд, 8")]))
    stops, _ = llm.normalize([t], FLEET)
    assert any("адрес" in w for w in stops[0].warnings)


def test_address_variants_expand_1c_abbreviations():
    v = geo._variants("111020, Московская обл., г. Балашиха, ш. Энтузиастов, д. 1Б, кв. 110")
    assert "Балашиха, шоссе Энтузиастов, 1" in v
    assert all("сква" not in x or "Москва" in x for x in geo._variants("г. Москва, ул. Наличная, 5"))
