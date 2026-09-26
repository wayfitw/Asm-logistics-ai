"""Оптимизация маршрутов (VRP с окнами, грузоподъёмностью и парами «забрать→отвезти») на OR-Tools.

ИИ сюда не лезет: математику делает солвер, он детерминирован и проверяем.
ИИ готовит для него чистые данные и объясняет/правит результат.
"""
from __future__ import annotations

import itertools
import os
from dataclasses import dataclass, field

from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from . import geo
from .config import fmt_hm, hm
from .models import PlannedStop, Route, Stop

URGENT_PENALTY = 1_000_000
NORMAL_PENALTY = 20_000
RESERVE_FIXED_COST = 400      # «стоимость» вывода резервной ГАЗели, в минутах
VEHICLE_FIXED_COST = 60


@dataclass
class Crew:
    driver: dict
    vehicle: dict


@dataclass
class SolveResult:
    routes: list[Route]
    unassigned: list[str]
    cost: int
    notes: list[str] = field(default_factory=list)
    extra_stops: list[Stop] = field(default_factory=list)   # виртуальные «погрузка на базе»


def crew_options(drivers: list[dict], vehicles: list[dict], unavailable: set[str]) -> list[list[Crew]]:
    """Все допустимые расстановки «водитель → машина» (без двух водителей на одной машине)."""
    vmap = {v["id"]: v for v in vehicles if v["id"] not in unavailable}
    choices = [[vid for vid in d["vehicles"] if vid in vmap] or [None] for d in drivers]
    out = []
    for combo in itertools.product(*choices):
        used = [c for c in combo if c]
        if len(used) != len(set(used)):
            continue
        out.append([Crew(d, vmap[c]) for d, c in zip(drivers, combo) if c])
    # сначала варианты без резерва
    out.sort(key=lambda crews: sum(c.vehicle.get("reserve", False) for c in crews))
    return out


def _allowed(stop: Stop, crew: Crew) -> bool:
    v = crew.vehicle
    if stop.vehicle_only and v["id"] not in stop.vehicle_only:
        return False
    if stop.driver_only and crew.driver["id"] not in stop.driver_only:
        return False
    if stop.needs_closed_body and not v.get("closed_body", True):
        return False
    if stop.needs_europallet and not v.get("fits_europallet", True):
        return False
    return stop.weight_kg <= v["capacity_kg"] and stop.volume_m3 <= v["volume_m3"]


def solve(stops: list[Stop], crews: list[Crew], base: dict, shift: dict,
          time_limit_s: int = 5) -> SolveResult:
    depot = (base["lat"], base["lon"])
    # Доставки без пары грузятся на базе: добавляем виртуальную точку «погрузка на базе».
    nodes: list[Stop | None] = [None]
    loads: list[Stop] = []
    paired = {s.pair_with for s in stops if s.kind == "pickup" and s.pair_with}
    for s in stops:
        if s.kind == "delivery" and s.id not in paired:
            loads.append(Stop(id=f"load:{s.id}", task_id=s.task_id, kind="pickup", title="Погрузка на базе",
                              task=f"Загрузить груз для: {s.title}",
                              address=base["address"], lat=base["lat"], lon=base["lon"],
                              tw_start=shift["start"], tw_end=s.tw_end, service_min=10,
                              weight_kg=s.weight_kg, volume_m3=s.volume_m3, pair_with=s.id,
                              vehicle_only=s.vehicle_only, driver_only=s.driver_only,
                              needs_closed_body=s.needs_closed_body, urgent=s.urgent))
    nodes += stops + loads
    idx_of = {n.id: i for i, n in enumerate(nodes) if n}
    pts = [depot] + [(n.lat, n.lon) for n in nodes[1:]]
    tmat, kmat = geo.matrix(pts)

    n, k = len(nodes), len(crews)
    manager = pywrapcp.RoutingIndexManager(n, k, 0)
    routing = pywrapcp.RoutingModel(manager)
    solver = routing.solver()

    def service(i: int) -> int:
        return nodes[i].service_min if nodes[i] else 0

    def transit(fi, ti):
        a, b = manager.IndexToNode(fi), manager.IndexToNode(ti)
        return tmat[a][b] + service(a)

    t_cb = routing.RegisterTransitCallback(transit)
    routing.SetArcCostEvaluatorOfAllVehicles(t_cb)
    day_start, day_end = hm(shift["start"]), hm(shift["end"])
    routing.AddDimension(t_cb, 240, day_end, False, "Time")
    time_dim = routing.GetDimensionOrDie("Time")
    # штраф за разницу «самый поздний возврат − старт»: не грузим одного водителя, пока другой стоит
    time_dim.SetGlobalSpanCostCoefficient(int(shift.get("balance_weight", 1)))

    def signed(attr):
        def cb(fi):
            i = manager.IndexToNode(fi)
            s = nodes[i]
            if not s:
                return 0
            val = getattr(s, attr) * (10 if attr == "volume_m3" else 1)   # объём в 0,1 м³
            return int(round(val)) if s.kind == "pickup" else -int(round(val)) if s.kind == "delivery" else 0
        return cb

    for attr, cap_key, name in (("weight_kg", "capacity_kg", "Weight"), ("volume_m3", "volume_m3", "Volume")):
        cb = routing.RegisterUnaryTransitCallback(signed(attr))
        mult = 10 if attr == "volume_m3" else 1
        routing.AddDimensionWithVehicleCapacity(cb, 0, [int(c.vehicle[cap_key] * mult) for c in crews], True, name)

    for v, crew in enumerate(crews):
        time_dim.CumulVar(routing.Start(v)).SetRange(day_start, day_start)
        time_dim.CumulVar(routing.End(v)).SetRange(day_start, day_end)
        routing.SetFixedCostOfVehicle(RESERVE_FIXED_COST if crew.vehicle.get("reserve") else VEHICLE_FIXED_COST, v)
        routing.AddVariableMinimizedByFinalizer(time_dim.CumulVar(routing.End(v)))

    for i, s in enumerate(nodes):
        if not s:
            continue
        ix = manager.NodeToIndex(i)
        time_dim.CumulVar(ix).SetRange(max(hm(s.tw_start), day_start), hm(s.tw_end))
        allowed = [v for v, c in enumerate(crews) if _allowed(s, c)]
        if not allowed:
            routing.ActiveVar(ix).SetValue(0)      # ни одна машина не подходит → в нераспределённые
        elif len(allowed) < k:
            # SetAllowedVehiclesForIndex в python-обёртке OR-Tools 9.15 не принимает list — режем домен напрямую
            routing.VehicleVar(ix).RemoveValues([v for v in range(k) if v not in allowed])
        routing.AddDisjunction([ix], URGENT_PENALTY if s.urgent else NORMAL_PENALTY)

    for s in nodes[1:]:
        if s.kind == "pickup" and s.pair_with and s.pair_with in idx_of:
            p, d = manager.NodeToIndex(idx_of[s.id]), manager.NodeToIndex(idx_of[s.pair_with])
            routing.AddPickupAndDelivery(p, d)
            solver.Add(routing.ActiveVar(p) == routing.ActiveVar(d))
            solver.Add(routing.VehicleVar(p) == routing.VehicleVar(d))
            solver.Add(time_dim.CumulVar(p) <= time_dim.CumulVar(d))

    groups: dict[str, list[int]] = {}
    for s in stops:
        if s.distinct_group:
            groups.setdefault(s.distinct_group, []).append(manager.NodeToIndex(idx_of[s.id]))
    for ixs in groups.values():
        for a, b in itertools.combinations(ixs, 2):
            # неактивная точка получает «фиктивную» машину, чтобы не мешать
            solver.Add(routing.VehicleVar(a) + (1 - routing.ActiveVar(a)) * 1000
                       != routing.VehicleVar(b) + (1 - routing.ActiveVar(b)) * 2000)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.seconds = time_limit_s
    sol = routing.SolveWithParameters(params)
    if not sol:
        return SolveResult([], [s.id for s in stops], 10**12, ["Солвер не нашёл допустимого решения"])

    routes, served = [], set()
    for v, crew in enumerate(crews):
        ix, planned, km, prev, load_max = routing.Start(v), [], 0.0, 0, 0
        wdim = routing.GetDimensionOrDie("Weight")
        while not routing.IsEnd(ix):
            nxt = sol.Value(routing.NextVar(ix))
            node, nnode = manager.IndexToNode(ix), manager.IndexToNode(nxt)
            km += kmat[node][nnode]
            if nodes[node]:
                t = sol.Min(time_dim.CumulVar(ix))
                planned.append(PlannedStop(stop_id=nodes[node].id, eta=fmt_hm(t), etd=fmt_hm(t + service(node))))
                served.add(nodes[node].id)
                load_max = max(load_max, sol.Value(wdim.CumulVar(nxt)))
            ix = nxt
        if not planned:
            continue
        end_t = sol.Min(time_dim.CumulVar(routing.End(v)))
        route_pts = [depot] + [pts[idx_of[p.stop_id]] for p in planned] + [depot]
        routes.append(Route(driver_id=crew.driver["id"], vehicle_id=crew.vehicle["id"],
                            stops=planned,
                            distance_km=round(km, 1), duration_min=end_t - day_start,
                            return_eta=fmt_hm(end_t), load_kg_max=load_max,
                            yandex_url=geo.yandex_route_url(route_pts)))
    unassigned = [s.id for s in stops if s.id not in served]
    return SolveResult(routes, unassigned, sol.ObjectiveValue(), [], loads)


def best_plan(stops: list[Stop], drivers: list[dict], vehicles: list[dict], base: dict, shift: dict,
              unavailable_vehicles: set[str] = frozenset()) -> tuple[SolveResult, list[Crew]]:
    """Перебираем расстановки водитель→машина (их единицы) и берём лучшую по целевой функции."""
    best, best_crews = None, []
    for crews in crew_options(drivers, vehicles, set(unavailable_vehicles)):
        if not crews:
            continue
        res = solve(stops, crews, base, shift, time_limit_s=int(os.getenv("SOLVER_TIME_LIMIT", "3")))
        if best is None or res.cost < best.cost:
            best, best_crews = res, crews
    if best is None:
        best = SolveResult([], [s.id for s in stops], 0, ["Сегодня нет доступных водителей/машин"])
    return best, best_crews


def evaluate(route_stop_ids: list[str], stops: dict[str, Stop], base: dict, shift: dict) -> tuple[list[PlannedStop], float, int, list[str]]:
    """Пересчёт ETA для порядка, заданного логистом вручную. Возвращает (точки, км, конец, нарушения)."""
    seq = [stops[i] for i in route_stop_ids]
    pts = [(base["lat"], base["lon"])] + [(s.lat, s.lon) for s in seq] + [(base["lat"], base["lon"])]
    tmat, kmat = geo.matrix(pts)
    t, km, out, issues = hm(shift["start"]), 0.0, [], []
    for k, s in enumerate(seq, start=1):
        t += tmat[k - 1][k] + (seq[k - 2].service_min if k > 1 else 0)
        km += kmat[k - 1][k]
        t = max(t, hm(s.tw_start))
        if t > hm(s.tw_end):
            issues.append(f"{s.title or s.address}: прибытие {fmt_hm(t)} позже окна до {s.tw_end}")
        out.append(PlannedStop(stop_id=s.id, eta=fmt_hm(t), etd=fmt_hm(t + s.service_min)))
    last = len(seq)
    end = t + (seq[-1].service_min if seq else 0) + tmat[last][last + 1]
    km += kmat[last][last + 1]
    if end > hm(shift["end"]):
        issues.append(f"Возврат на базу в {fmt_hm(end)} — позже конца смены {shift['end']}")
    return out, round(km, 1), end, issues
