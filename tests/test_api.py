"""API и защита от плохих данных: ошибки ввода → понятные 4xx, утверждённый план не затирается,
кривой ответ модели не роняет расчёт. Офлайн."""
from __future__ import annotations

import json
from types import SimpleNamespace as NS

import pytest
from fastapi.testclient import TestClient

from logistics import db, geo, llm
from logistics.config import DATA, load_fleet
from logistics.models import Counterparty, CounterpartyAddress, RawTask

FLEET = load_fleet()
SAMPLE = (DATA / "sample_tasks.json").read_bytes()
KNOWN = {  # «геокодер» без сети
    "Химки, Вашутинское шоссе, 36": (55.9191, 37.4195),
    "Москва, Рязанский проспект, 2": (55.7283, 37.7409),
    "Люберцы, Котельнический проезд, 8": (55.6666, 37.8890),
    "Минск, проспект Независимости, 1": (53.9000, 27.5600),
}


@pytest.fixture()
def client(monkeypatch, tmp_path):
    def fake_matrix(points):
        kms = [[round(geo._haversine_km(a, b) * 1.35, 1) for b in points] for a in points]
        return [[round(k / 25 * 60) for k in row] for row in kms], kms
    monkeypatch.setattr(geo, "matrix", fake_matrix)
    monkeypatch.setattr(geo, "geocode", lambda a: geo.coords_from_text(a) or KNOWN.get(a))
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "t.db")
    monkeypatch.setenv("SOLVER_TIME_LIMIT", "1")
    monkeypatch.setattr(llm.settings.__class__, "llm_enabled", property(lambda self: False))
    monkeypatch.setattr(llm.settings.__class__, "import_token", property(lambda self: "tok"))
    import app as A
    return TestClient(A.app)


D = "2026-10-01"   # четверг: работают все трое


def build(c, day=D, **form):
    return c.post(f"/api/plan/{day}/build", data={"sample": "true", **form})


def edit(c, tool, **args):
    return c.post(f"/api/plan/{D}/edit", json={"tool": tool, "args": args})


def test_bad_input_gives_clear_4xx_not_500(client):
    assert client.get("/api/state/2026-13-45").status_code == 400
    assert "ГГГГ-ММ-ДД" in client.get("/api/state/завтра").json()["detail"]
    r = client.post(f"/api/plan/{D}/build", files={"file": ("a.txt", b"x")})
    assert r.status_code == 400 and ".json" in r.json()["detail"]
    assert client.post(f"/api/plan/{D}/build", files={"file": ("a.json", b"{oops")}).status_code == 400
    assert build(client).status_code == 200
    assert edit(client, "move_stop", stop_id="00АН-002648:1").status_code == 400          # нет аргументов
    assert edit(client, "update_stop", stop_id="00АН-002648:1", weight_kg=5000).status_code == 400  # чужое поле
    r = edit(client, "update_stop", stop_id="00АН-002648:1", tw_start="9", tw_end="abc")
    assert r.status_code == 400 and "ЧЧ:ММ" in r.json()["detail"]
    r = edit(client, "update_stop", stop_id="00АН-002648:1", tw_start="15:00", tw_end="10:00")
    assert r.status_code == 400 and "пустое" in r.json()["detail"]
    r = client.post(f"/api/plan/{D}/tasks", json={"text": "  "})
    assert r.status_code == 400 and "Опишите" in r.json()["detail"]
    assert "повреждён" in client.post(f"/api/plan/{D}/build", files={"file": ("a.json", b"{oops")}).json()["detail"]
    assert client.post(f"/api/plan/{D}/build", files={"file": ("a.json", b'{"x": 1}')}).status_code == 400
    csv_one_col = "Адрес\nРеутов, 55.7646, 37.8471\n".encode()
    r = client.post("/api/plan/2026-10-02/build", files={"file": ("a.csv", csv_one_col)})
    assert r.status_code == 200 and len(r.json()["stops"]) == 1


def test_time_input_is_normalized(client):
    build(client)
    r = edit(client, "update_stop", stop_id="00АН-002648:1", tw_start="9.30", tw_end="14")
    assert r.status_code == 200
    s = client.get(f"/api/state/{D}").json()["plan"]["stops"]["00АН-002648:1"]
    assert (s["tw_start"], s["tw_end"]) == ("09:30", "14:00")


def test_approved_plan_is_not_overwritten(client):
    build(client)
    client.post(f"/api/plan/{D}/approve")
    assert build(client).status_code == 409
    r = client.post(f"/api/import/1c/{D}", json=json.loads(SAMPLE), headers={"x-token": "tok"})
    assert r.status_code == 409
    assert client.get(f"/api/state/{D}").json()["plan"]["status"] == "approved"
    # пробег на начало дня можно внести и после утверждения
    drv = client.get(f"/api/state/{D}").json()["plan"]["routes"][0]["driver_id"]
    assert edit(client, "set_odometer", driver_id=drv, km=61301).status_code == 200
    assert client.get(f"/api/state/{D}").json()["plan"]["routes"][0]["odometer_start"] == 61301
    assert edit(client, "replan", unavailable_vehicles=[]).status_code == 400   # а маршрут — нет


def test_odometer_survives_replan(client):
    build(client)
    plan = client.get(f"/api/state/{D}").json()["plan"]
    drv = plan["routes"][0]["driver_id"]
    edit(client, "set_odometer", driver_id=drv, km=150046)
    other = next(d["id"] for d in FLEET["drivers"] if d["id"] != drv)
    plan = edit(client, "set_driver", driver_id=other, works=False).json()["plan"]
    assert next(r for r in plan["routes"] if r["driver_id"] == drv)["odometer_start"] == 150046


def test_driver_can_be_called_out_on_weekend(client):
    sat = "2026-10-03"
    r = build(client, day=sat)
    assert r.status_code == 200 and r.json()["routes"] == []
    r = client.post(f"/api/plan/{sat}/edit", json={"tool": "set_driver", "args": {"driver_id": "rybkin", "works": True}})
    plan = r.json()["plan"]
    assert {x["driver_id"] for x in plan["routes"]} == {"rybkin"}
    assert not any("по графику не работает" in v for v in plan["violations"])
    fleet = client.get(f"/api/state/{sat}").json()["fleet"]
    assert next(d for d in fleet["drivers"] if d["id"] == "rybkin")["works_today"] is True


def test_driver_off_is_removed_from_plan(client):
    build(client)
    r = edit(client, "set_driver", driver_id="kolesnikov", works=False)
    assert "kolesnikov" not in {x["driver_id"] for x in r.json()["plan"]["routes"]}


def test_build_with_driver_overrides_from_form(client):
    r = build(client, day="2026-10-03", drivers="ushakov:0,rybkin:1")
    assert {x["driver_id"] for x in r.json()["routes"]} <= {"rybkin"} and r.json()["routes"]


def test_alternative_warehouses_offered_and_selectable(client):
    t = RawTask(id="O", counterparty=Counterparty(id="k", name="ОКТАНТ", addresses=[
        CounterpartyAddress(kind="Юридический", text="Москва, ул. Наличная, 5"),
        CounterpartyAddress(kind="Склад", text="Москва, Рязанский проспект, 2"),
        CounterpartyAddress(kind="Склад", text="Люберцы, Котельнический проезд, 8")]))
    from logistics import pipeline
    stops, _ = pipeline.prepare_stops([t], FLEET)
    s = stops[0]
    assert s.address == "Москва, Рязанский проспект, 2"
    assert s.alt_addresses == ["Люберцы, Котельнический проезд, 8"]   # юридический не предлагаем


def test_address_outside_moscow_is_rejected(client):
    build(client)
    r = edit(client, "update_stop", stop_id="00АН-002648:1", address="Минск, проспект Независимости, 1")
    assert r.status_code == 400 and "не найден" in r.json()["detail"]


def test_llm_output_is_sanitized(monkeypatch):
    base = {"local_id": "1", "task_id": "A", "kind": "pickup", "title": "t", "address": "x", "lat": 10.0, "lon": 10.0,
            "tw_start": "09.00", "tw_end": "к обеду", "service_min": 0, "weight_kg": -5, "volume_m3": 0,
            "needs_closed_body": False, "needs_europallet": False, "urgent": False, "requires_approval": False,
            "vehicles_needed": 50, "pair_with_local_id": None, "vehicle_ids": [], "driver_ids": [],
            "task": "", "contact": "", "invoice_no": "", "order_no": "", "warnings": []}
    ghost = dict(base, task_id="NOT-IN-INPUT")
    fake = NS(responses=NS(create=lambda **kw: NS(output_text=json.dumps({"stops": [base, ghost], "questions": []}))))
    monkeypatch.setattr(llm.settings.__class__, "llm_enabled", property(lambda self: True))
    monkeypatch.setattr(llm, "client", lambda: fake)
    tasks = [RawTask(id="A", source="manual", text="x"), RawTask(id="B", source="manual", text="Реутов забрать до 12")]
    stops, _ = llm.normalize(tasks, FLEET)
    a = [s for s in stops if s.task_id == "A"]
    assert len(a) == 4                                        # «нужно 50 машин» → не больше 4
    assert (a[0].tw_start, a[0].tw_end) == tuple(FLEET["shift"]["default_window"])
    assert a[0].lat is None and a[0].weight_kg == 0 and a[0].service_min >= 5
    assert any("окно" in w for w in a[0].warnings) and any("вне Москвы" in w for w in a[0].warnings)
    assert not any(s.task_id == "NOT-IN-INPUT" for s in stops)   # выдуманная заявка отброшена
    b = [s for s in stops if s.task_id == "B"]                   # пропущенная моделью — не потеряна
    assert b and any("ИИ пропустил" in w for w in b[0].warnings)
