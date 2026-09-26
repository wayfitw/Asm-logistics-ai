"""Контракт с OpenAI Responses API без сети: поддельный клиент отдаёт то, что отдал бы API."""
from __future__ import annotations

import json
from types import SimpleNamespace as NS

from logistics import db, geo, llm, pipeline
from logistics.config import load_fleet
from logistics.models import Plan, RawTask, Route, PlannedStop, Stop

FLEET = load_fleet()


class FakeResponses:
    def __init__(self, script):
        self.script, self.calls = list(script), []

    def create(self, **kw):
        self.calls.append(kw)
        return self.script.pop(0)


def fc(name, args, cid):
    return NS(type="function_call", name=name, arguments=json.dumps(args), call_id=cid)


def test_schema_is_strict_compatible():
    """Strict Structured Outputs: все поля required, additionalProperties=false на каждом объекте."""
    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                assert node["additionalProperties"] is False
                assert set(node["required"]) == set(node["properties"])
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
    walk(llm._normalize_schema(FLEET))
    for t in llm.AGENT_TOOLS:
        assert t["strict"] is True
        walk(t["parameters"])


def test_normalize_parses_structured_output(monkeypatch):
    item = {"local_id": "1", "task_id": "M-2", "kind": "pickup", "title": "Масло (Коломенская)",
            "address": "Москва, Коломенская ул., 16", "lat": None, "lon": None, "tw_start": "08:00", "tw_end": "09:00",
            "service_min": 20, "weight_kg": 200, "volume_m3": 0.5, "needs_closed_body": False, "needs_europallet": False,
            "urgent": True, "requires_approval": False, "vehicles_needed": 1, "pair_with_local_id": "2",
            "vehicle_ids": [], "driver_ids": [], "task": "Забрать масло", "contact": "Илья, +7 903 772-46-72",
            "invoice_no": "", "order_no": "", "warnings": []}
    item2 = item | {"local_id": "2", "kind": "delivery", "title": "Балашиха", "address": "Балашиха, Объездное ш., 8",
                    "pair_with_local_id": None, "tw_start": "09:00", "tw_end": "18:00", "task": "Выгрузить"}
    fake = FakeResponses([NS(output_text=json.dumps({"stops": [item, item2], "questions": ["Сколько бочек?"]}))])
    monkeypatch.setattr(llm.settings.__class__, "llm_enabled", property(lambda self: True))
    monkeypatch.setattr(llm, "client", lambda: NS(responses=fake))
    stops, q = llm.normalize([RawTask(id="M-2", source="manual", text="...")], FLEET)
    assert [s.id for s in stops] == ["M-2:1", "M-2:2"]
    assert stops[0].pair_with == "M-2:2" and stops[0].urgent
    assert q == ["Сколько бочек?"]
    fmt = fake.calls[0]["text"]["format"]
    assert fmt["type"] == "json_schema" and fmt["strict"] is True


def test_agent_loop_executes_tools_and_chains_responses(monkeypatch, tmp_path):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "t.db")
    monkeypatch.setattr(geo, "matrix", lambda pts: ([[10] * len(pts)] * len(pts), [[5.0] * len(pts)] * len(pts)))
    monkeypatch.setattr(llm.settings.__class__, "llm_enabled", property(lambda self: True))
    stops = {k: Stop(id=k, task_id=k, title=k, address=k, lat=55.7, lon=37.6) for k in ("a", "b")}
    plan = Plan(date="2026-09-24", stops=stops, routes=[
        Route(driver_id="kolesnikov", vehicle_id="staria_1", stops=[PlannedStop(stop_id="a", eta="09:00", etd="09:25"),
                                                                      PlannedStop(stop_id="b", eta="10:00", etd="10:25")])])
    fake = FakeResponses([
        NS(id="r1", output=[fc("move_stop", {"stop_id": "b", "driver_id": "ushakov", "position": None}, "c1")], output_text=""),
        NS(id="r2", output=[NS(type="message")], output_text="Перенёс «b» Ушакову."),
    ])
    monkeypatch.setattr(llm, "client", lambda: NS(responses=fake))
    ed = pipeline.PlanEditor(plan, actor="ai")
    answer = llm.PlanAgent(ed).run("b отдай Ушакову", FLEET)
    assert answer == "Перенёс «b» Ушакову."
    assert {r.driver_id for r in ed.plan.routes} == {"kolesnikov", "ushakov"}
    second = fake.calls[1]
    assert second["previous_response_id"] == "r1"
    out = second["input"][0]
    assert out["type"] == "function_call_output" and out["call_id"] == "c1"
    assert json.loads(out["output"])["ok"] is True
