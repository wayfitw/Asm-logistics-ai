"""Задача коммивояжёра (TSP) — независимая проверка порядка точек внутри каждого маршрута.

Основной солвер (solver.py) решает VRP: кому какие точки и в каком порядке. Он эвристический
и ограничен по времени, поэтому порядок может быть не лучшим. Здесь для уже распределённых точек
одного водителя ищем порядок ТОЧНО — динамическим программированием Хелда–Карпа:

    состояние (посещённые точки, последняя точка) → самое раннее время готовности к выезду.

С окнами времени раннее время доминирует: если туда же можно попасть раньше — всё, что достижимо
позже, достижимо и отсюда (ждать разрешено). Значит DP даёт оптимум по времени возврата на базу
с учётом окон и правила «погрузка раньше выгрузки». Сложность O(2^n · n²): точно до ~13 точек,
больше — отдаём OR-Tools как TSP для одной машины.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import geo
from .config import hm
from .models import Stop

EXACT_LIMIT = 13


@dataclass
class TspResult:
    order: list[str]
    end_min: int            # время возврата на базу, минуты от полуночи
    km: float
    method: str             # "exact" | "ortools" | "infeasible"


def best_order(stop_ids: list[str], stops: dict[str, Stop], base: dict, shift: dict) -> TspResult:
    if not stop_ids:
        return TspResult([], hm(shift["start"]), 0.0, "exact")
    seq = [stops[i] for i in stop_ids]
    pts = [(base["lat"], base["lon"])] + [(s.lat, s.lon) for s in seq]
    tmat, kmat = geo.matrix(pts)
    if len(seq) <= EXACT_LIMIT:
        return _held_karp(seq, tmat, kmat, shift)
    return _ortools(seq, tmat, kmat, shift)


def _precedence(seq: list[Stop]) -> list[int]:
    """need[j] — битовая маска точек, которые обязаны быть посещены до j (погрузка до выгрузки)."""
    idx = {s.id: k for k, s in enumerate(seq)}
    need = [0] * len(seq)
    for k, s in enumerate(seq):
        if s.kind == "pickup" and s.pair_with in idx:
            need[idx[s.pair_with]] |= 1 << k
    return need


def _held_karp(seq, tmat, kmat, shift) -> TspResult:
    n, start, end_limit = len(seq), hm(shift["start"]), hm(shift["end"])
    tw = [(hm(s.tw_start), hm(s.tw_end)) for s in seq]
    svc = [s.service_min for s in seq]
    need = _precedence(seq)
    # узлы матрицы: 0 — база, 1..n — точки
    INF = 10**9
    ready = {}      # (mask, j) -> время окончания работ в j
    parent = {}
    for j in range(n):
        if need[j]:
            continue
        arr = max(start + tmat[0][j + 1], tw[j][0])
        if arr <= tw[j][1]:
            ready[(1 << j, j)] = arr + svc[j]
            parent[(1 << j, j)] = None
    for mask in range(1, 1 << n):
        for j in range(n):
            t = ready.get((mask, j))
            if t is None:
                continue
            for k in range(n):
                if mask >> k & 1 or (need[k] & mask) != need[k]:
                    continue
                arr = max(t + tmat[j + 1][k + 1], tw[k][0])
                if arr > tw[k][1]:
                    continue
                key, val = (mask | 1 << k, k), arr + svc[k]
                if val < ready.get(key, INF):
                    ready[key], parent[key] = val, j
    full = (1 << n) - 1
    best_j, best_end = None, INF
    for j in range(n):
        if (full, j) in ready:
            e = ready[(full, j)] + tmat[j + 1][0]
            if e < best_end:
                best_j, best_end = j, e
    if best_j is None or best_end > end_limit:
        return TspResult([s.id for s in seq], best_end if best_j is not None else INF, 0.0, "infeasible")
    order, mask, j = [], full, best_j
    while j is not None:
        order.append(j)
        mask, j = mask ^ (1 << j), parent[(mask, j)]
    order.reverse()
    km = kmat[0][order[0] + 1] + sum(kmat[a + 1][b + 1] for a, b in zip(order, order[1:])) + kmat[order[-1] + 1][0]
    return TspResult([seq[k].id for k in order], best_end, round(km, 1), "exact")


def _ortools(seq, tmat, kmat, shift) -> TspResult:
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    n = len(seq) + 1
    manager = pywrapcp.RoutingIndexManager(n, 1, 0)
    routing = pywrapcp.RoutingModel(manager)
    svc = [0] + [s.service_min for s in seq]
    cb = routing.RegisterTransitCallback(
        lambda a, b: tmat[manager.IndexToNode(a)][manager.IndexToNode(b)] + svc[manager.IndexToNode(a)])
    routing.SetArcCostEvaluatorOfAllVehicles(cb)
    routing.AddDimension(cb, 240, hm(shift["end"]), False, "Time")
    td = routing.GetDimensionOrDie("Time")
    td.CumulVar(routing.Start(0)).SetRange(hm(shift["start"]), hm(shift["start"]))
    for k, s in enumerate(seq, start=1):
        td.CumulVar(manager.NodeToIndex(k)).SetRange(hm(s.tw_start), hm(s.tw_end))
    idx = {s.id: k for k, s in enumerate(seq, start=1)}
    for s in seq:
        if s.kind == "pickup" and s.pair_with in idx:
            p, d = manager.NodeToIndex(idx[s.id]), manager.NodeToIndex(idx[s.pair_with])
            routing.AddPickupAndDelivery(p, d)
            routing.solver().Add(td.CumulVar(p) <= td.CumulVar(d))
    routing.AddVariableMinimizedByFinalizer(td.CumulVar(routing.End(0)))
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.seconds = 2
    sol = routing.SolveWithParameters(params)
    if not sol:
        return TspResult([s.id for s in seq], 10**9, 0.0, "infeasible")
    order, ix, km = [], routing.Start(0), 0.0
    while not routing.IsEnd(ix):
        nxt = sol.Value(routing.NextVar(ix))
        km += kmat[manager.IndexToNode(ix)][manager.IndexToNode(nxt)]
        if manager.IndexToNode(ix):
            order.append(seq[manager.IndexToNode(ix) - 1].id)
        ix = nxt
    return TspResult(order, sol.Min(td.CumulVar(routing.End(0))), round(km, 1), "ortools")
