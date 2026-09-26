"""Лист маршрута для водителя — в привычном формате «Логистического плана работ»,
плюс ссылка на маршрут в Яндекс.Картах. Открывается по ссылке, печатается, сохраняется в PDF."""
from __future__ import annotations

from html import escape

from .config import load_fleet
from .models import Plan, Route

KIND = {"pickup": "Загрузка", "delivery": "Выгрузка", "service": "Задача"}


def driver_sheet(plan: Plan, route: Route) -> str:
    fleet = load_fleet()
    d = next(x for x in fleet["drivers"] if x["id"] == route.driver_id)
    v = next(x for x in fleet["vehicles"] if x["id"] == route.vehicle_id)
    rows = []
    for n, p in enumerate(route.stops, 1):
        s = plan.stops[p.stop_id]
        flags = "".join([
            '<span class="f u">СРОЧНО</span>' if s.urgent else "",
            '<span class="f a">ПО СОГЛАСОВАНИЮ</span>' if s.requires_approval else ""])
        docs = " · ".join(x for x in (f"Заказ {s.order_no}" if s.order_no else "",
                                      f"Счёт {s.invoice_no}" if s.invoice_no else "") if x)
        rows.append(f"""<tr><td class="n">{n}</td><td class="t">{p.eta}<br><small>до {s.tw_end}</small></td>
          <td><b>{KIND[s.kind]}</b> {flags}<br>{escape(s.title)}<br><small>{escape(s.address)}</small></td>
          <td>{escape(s.task)}{f'<br><small>{escape(docs)}</small>' if docs else ''}</td>
          <td>{escape(s.contact)}</td></tr>""")
    status = {"draft": "ЧЕРНОВИК — не утверждён", "approved": "Утверждён", "sent": "Утверждён и отправлен"}[plan.status]
    return f"""<!doctype html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Маршрут {escape(d['name'].split()[0])} {plan.date}</title>
<style>
 body{{font:15px/1.4 system-ui,Segoe UI,Roboto,sans-serif;margin:0;padding:16px;color:#1b1b1b;background:#fff}}
 h1{{font-size:20px;margin:0 0 4px}} .meta{{color:#555;margin-bottom:12px}}
 .bar{{background:#f2a33a;padding:8px 12px;font-weight:600;border-radius:6px 6px 0 0}}
 table{{border-collapse:collapse;width:100%}} td,th{{border:1px solid #ccc;padding:6px 8px;vertical-align:top;text-align:left}}
 th{{background:#ffe94d}} td.n{{width:24px;text-align:center}} td.t{{white-space:nowrap}} small{{color:#666}}
 .f{{font-size:11px;padding:1px 5px;border-radius:4px;margin-left:4px;color:#fff}} .u{{background:#d33}} .a{{background:#7a5cff}}
 .btn{{display:inline-block;margin:12px 8px 0 0;padding:10px 14px;background:#fc0;color:#000;border-radius:8px;text-decoration:none;font-weight:600}}
 .st{{display:inline-block;padding:2px 8px;border-radius:4px;background:{'#fde2e2' if plan.status == 'draft' else '#dff5e1'}}}
 @media (max-width:640px){{table,tr,td{{display:block;width:auto}} th{{display:none}} tr{{border-bottom:2px solid #999;margin-bottom:6px}} td{{border:none}}}}
 @media print{{.btn{{display:none}}}}
</style></head><body>
<h1>Логистический план работ · {plan.date}</h1>
<div class="meta"><span class="st">{status}</span> · Выезд с базы 08:00 — {escape(fleet['base']['address'])}</div>
<div class="bar">{escape(d['name'])}, {escape(d.get('phone', ''))} · {escape(v.get('plate', ''))} {escape(v['name'])}
 · пробег на начало дня: {route.odometer_start if route.odometer_start is not None else '________'}</div>
<table><tr><th>#</th><th>Время</th><th>Адрес</th><th>Задача</th><th>Контакт</th></tr>{''.join(rows)}</table>
<p>Возврат на базу ≈ <b>{route.return_eta}</b> · {route.distance_km} км</p>
<a class="btn" href="{escape(route.yandex_url)}" target="_blank">Открыть маршрут в Яндекс.Картах</a>
<a class="btn" href="javascript:print()">Печать / PDF</a>
</body></html>"""
