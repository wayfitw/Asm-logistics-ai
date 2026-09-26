# АСМ · ИИ-планирование маршрутов (MVP)

Логист-снабженец нажимает кнопку, система берёт оплаченные заказы поставщикам и ручные задачи,
приводит их в порядок, раскладывает по водителям и машинам, логист правит результат и утверждает,
водитель получает лист маршрута.

## Главный принцип
**LLM понимает и объясняет, код гарантирует.**
- OpenAI Responses API (`logistics/llm.py`) разбирает кривые заявки в строгий JSON (Structured Outputs, `strict: true`),
  объясняет план и работает агентом правок через function calling.
- Оптимизация: OR-Tools (`logistics/solver.py`). Жёсткие ограничения проверяет `logistics/checks.py`
  после солвера и после каждой правки. Ни грузоподъёмность, ни допуск водителя к машине, ни окна LLM не решает.
- Любую правку ИИ-агент делает только через `PlanEditor` — тем же путём, что и кнопки интерфейса.

## Структура
- `config/fleet.yaml` — база, смена, машины, водители, графики. Бизнес-ограничения живут здесь, не в промптах.
- `logistics/models.py` — RawTask (вход) → Stop (точка) → Plan/Route (результат).
- `logistics/tsp.py` — вторая проверка порядка внутри маршрута: точный TSP с окнами (Хелд–Карп, ≤13 точек), дальше OR-Tools.
  После авторасчёта применяется (`finalize(apply_tsp=True)`), после ручных правок — только подсказка `tsp_gain_min`.
- `logistics/pipeline.py` — build_plan, PlanEditor (move_stop / reorder_route / update_stop / optimize_order / replan / add_tasks), approve.
- `logistics/geo.py` — геокодер (Яндекс или Nominatim + очистка сокращений 1С), матрица OSRM × TRAFFIC_FACTOR.
- `logistics/db.py` — SQLite: планы и журнал (import → normalized → solved → edit/agent → approved → sent).
- `logistics/importer.py` — JSON / CSV / XLSX. `logistics/render.py` — лист водителя. `logistics/bitrix.py` — заготовка.
- `app.py` + `static/index.html` — рабочее место логиста.

## Модель задач
- Заказ поставщику = `pickup` у поставщика, груз едет на базу.
- «Забрать А → выгрузить Б» = pickup с `pair_with` = id delivery. Пара всегда хранится на погрузке.
- Доставка без пары грузится на базе: солвер сам добавляет виртуальную точку `load:<id>`.
- «Нужно 2 машины» = копии `…#1`, `…#2` с общим `distinct_group`.
- `needs_review` — показывать логисту: есть warnings, нет координат или «по согласованию».

## Команды
```
python -m pip install -r requirements.txt
python -m pytest -q                      # офлайн, без ключа и сети
python -m uvicorn app:app --reload       # http://localhost:8000
```

## Ввод и ошибки
- Ошибки ввода — `ValueError` с текстом для логиста → 400 (обработчик в app.py). Дата в URL проверяется middleware.
- Утверждённый план пересобрать нельзя (409) — сначала «Вернуть в черновик». Пробег (`set_odometer`) можно и после утверждения.
- Вывод модели чистит `llm._sanitize`: время через `config.parse_hm`, координаты через `config.in_region`.
- Выход водителя на дату — `Plan.driver_overrides` (поверх `overrides` из fleet.yaml).

## Правила
- Ключи только в `.env` (см. `.env.example`), в код не класть.
- Новое ограничение = правило в solver.py + проверка в checks.py + тест-сценарий.
- Меняешь схему/инструменты LLM — прогони `tests/test_llm_contract.py` (strict-схемы, цикл function calling).
- Google Sheets как основу не используем (решение в плане разработки). 1С и Bitrix — после MVP.
