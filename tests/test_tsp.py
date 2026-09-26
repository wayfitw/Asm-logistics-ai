"""TSP-проверка порядка точек: точность против полного перебора и встраивание в план."""
from __future__ import annotations

import itertools
import random

import pytest

from logistics import db, geo, llm, pipeline, solver, tsp
from logistics.config import load_fleet
from logistics.models import Plan, PlannedStop, Route, Stop

FLEET = load_fleet()
BASE, SHIFT = FLEET["base"], FLEET["shift"]


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    def fake_matrix(points):
        kms = [[round(geo._haversine_km(a, b) * 1.35, 1) for b in points] for a in points]
        return [[round(k / 25 * 60) for k in row] for row in kms], kms
    monkeypatch.setattr(geo, "matrix", fake_matrix)
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "t.db")
    monkeypatch.setattr(llm.settings.__class__, "llm_enabled", property(lambda self: False))


def rand_stops(rng, n, windows=True, pairs=0):
    out = {}
    for k in range(n):
        tw = ("09:00", "18:00")
        if windows and rng.random() < 0.4:
            h = rng.randint(9, 14)
            tw = (f"{h:02d}:00", f"{h + 3:02d}:00")
        out[f"s{k}"] = Stop(id=f"s{k}", task_id=f"s{k}", address="x", title=f"s{k}",
                            lat=55.6 + rng.random() * 0.3, lon=37.4 + rng.random() * 0.5,
                            tw_start=tw[0], tw_end=tw[1], service_min=15)
    for k in range(pairs):
        out[f"s{2 * k}"].pair_with = f"s{2 * k + 1}"
        out[f"s{2 * k + 1}"].kind = "delivery"
    return out


def brute_force(stops):
    """Лучшее время возврата перебором всех допустимых перестановок."""
    best = None
    for perm in itertools.permutations(stops):
        seen, ok = set(), True
        for i in perm:
            s = stops[i]
            if s.kind == "delivery" and any(p.pair_with == i and p.id not in seen for p in stops.values()):
                ok = False
                break
            seen.add(i)
        if not ok:
            continue
        _, _, end, issues = solver.evaluate(list(perm), stops, BASE, SHIFT)
        if not issues and (best is None or end < best):
            best = end
    return best


@pytest.mark.parametrize("seed", range(12))
def test_exact_tsp_matches_brute_force(seed):
    rng = random.Random(seed)
    stops = rand_stops(rng, 6, pairs=seed % 3)
    res = tsp.best_order(list(stops), stops, BASE, SHIFT)
    bf = brute_force(stops)
    if bf is None:
        assert res.method == "infeasible"
    else:
        assert res.method == "exact" and res.end_min == bf
        _, _, end, issues = solver.evaluate(res.order, stops, BASE, SHIFT)
        assert not issues and end == res.end_min   # порядок из DP даёт ровно заявленное время


def test_pickup_always_before_delivery():
    rng = random.Random(7)
    stops = rand_stops(rng, 8, windows=False, pairs=3)
    order = tsp.best_order(list(stops), stops, BASE, SHIFT).order
    for k in range(3):
        assert order.index(f"s{2 * k}") < order.index(f"s{2 * k + 1}")


def test_large_route_falls_back_to_ortools():
    rng = random.Random(1)
    stops = rand_stops(rng, 16, windows=False)
    for s in stops.values():   # 16 точек по 15 мин по всей Москве до 18:00 не объехать — делаем визиты короче
        s.service_min, s.lat, s.lon = 5, 55.72 + (s.lat - 55.6) / 5, 37.6 + (s.lon - 37.4) / 5
    res = tsp.best_order(list(stops), stops, BASE, SHIFT)
    assert res.method == "ortools" and sorted(res.order) == sorted(stops)
    _, _, end, issues = solver.evaluate(res.order, stops, BASE, SHIFT)
    _, _, end_given, _ = solver.evaluate(list(stops), stops, BASE, SHIFT)
    assert not issues and end <= end_given


def test_manual_bad_order_gets_hint_and_button_fixes_it():
    # точки на одной линии от базы: правильный порядок — по удалению, «плохой» — вперемешку
    pts = [(55.74, 37.80), (55.75, 37.88), (55.76, 37.96), (55.77, 38.04)]
    stops = {f"p{k}": Stop(id=f"p{k}", task_id=f"p{k}", title=f"p{k}", address="x", lat=a, lon=b, service_min=10)
             for k, (a, b) in enumerate(pts)}
    bad = ["p3", "p0", "p2", "p1"]
    plan = Plan(date="2026-09-24", stops=stops,
                routes=[Route(driver_id="kolesnikov", vehicle_id="staria_1",
                              stops=[PlannedStop(stop_id=i, eta="", etd="") for i in ["p0", "p1", "p2", "p3"]])])
    db.save_plan(plan)
    ed = pipeline.PlanEditor(plan)
    ed.call("reorder_route", {"driver_id": "kolesnikov", "stop_ids": bad})
    r = ed.plan.routes[0]
    assert [p.stop_id for p in r.stops] == bad          # ручной порядок логиста не тронут
    assert r.tsp_gain_min > 0 and "быстрее" in r.tsp    # но есть подсказка
    before = r.return_eta
    ed.call("optimize_order", {"driver_id": "kolesnikov"})
    r = ed.plan.routes[0]
    assert r.return_eta < before and r.tsp_gain_min == 0
    assert [p.stop_id for p in r.stops] in (["p0", "p1", "p2", "p3"], ["p3", "p2", "p1", "p0"])


def test_build_plan_routes_are_tsp_optimal():
    """После автоматического расчёта каждый маршрут не хуже точного TSP."""
    rng = random.Random(3)
    stops = list(rand_stops(rng, 9, windows=True).values())
    res, _ = solver.best_plan(stops, [d for d in FLEET["drivers"] if d["id"] != "ushakov"],
                              FLEET["vehicles"], BASE, SHIFT)
    plan = Plan(date="2026-09-24", stops={s.id: s for s in stops}, routes=res.routes, unassigned=res.unassigned)
    pipeline.tsp_review(plan, FLEET, apply=True)
    for r in plan.routes:
        exact = tsp.best_order([p.stop_id for p in r.stops], plan.stops, BASE, SHIFT)
        assert pipeline.hm(r.return_eta) <= exact.end_min
        assert "TSP" in r.tsp
