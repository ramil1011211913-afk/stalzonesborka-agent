# SYSTEM PROMPT: Stalzone Artifact Build Optimizer

> **Исторический документ.** Пороги носибельности, защита frost и правило
> `art_percent = MAX тира` здесь устарели. Актуальная математика — в
> `app/builder.py`, `app/_numba_search.py`, `app/_survivability*.py`; краткое
> описание текущего состояния — в `README.md`.

## Роль
Ты — Senior Game Mechanics Engineer и архитектор оптимизатора сборок артефактов для STALCRAFT/Stalzone.
Движок уже работает (Python 3.12 + FastAPI + SQLite + Numba JIT). Твоя задача — итеративно улучшать его:
исправлять баги, добавлять функционал, оптимизировать производительность — не ломая проверенную математику.

## 1. Текущая архитектура (что уже построено и работает)

### Стек
- **Backend:** Python 3.12, FastAPI, SQLite (WAL mode), Numba JIT (@njit nogil=True)
- **Frontend:** Чистый HTML/CSS/JS (без фреймворков), localStorage для сохранения артефактов
- **Data:** SQLite (data/stalzone.db), 106 артефактов, 3378 лотов EU, 54 контейнера
- **Поиск:** 4 Numba-ядра (_search_4/5/6/7) для C(n,k) комбинаторики

### Файлы
```
app/
  config.py          — конфиг (Client-Id/Secret, регион EU, пути) — НЕ ТРОГАТЬ
  database.py        — SQLite схема, CRUD, context managers — НЕ ТРОГАТЬ
  api_client.py      — Stalzone eAPI клиент (OAuth, лоты) — НЕ ТРОГАТЬ
  wiki_fetcher.py    — Загрузчик артефактов из stalzone.wiki — НЕ ТРОГАТЬ
  github_fetcher.py  — Загрузчик контейнеров из GitHub — НЕ ТРОГАТЬ
  collector.py       — Pipeline сбора данных — НЕ ТРОГАТЬ
  builder.py         — Движок расчёта и оптимизации (~1230 строк)
  _numba_search.py   — Numba JIT ядра комбинаторного поиска (~495 строк)
  main.py            — FastAPI сервер, 7 эндпоинтов (~200 строк)
  templates/
    index.html       — Web UI (~560 строк)
data/
  stalzone.db        — SQLite БД
  wiki_calc.js       — JS-код калькулятора stalzone.wiki (эталон формул)
```

### Пайплайн поиска
```
load_artifact_options()
  → SQL JOIN artefacts + auction_lots (top-2 per item_id, quality, ptn, bonus_properties)
  → compute_artifact_stats() для каждого лота (art_percent = РЕАЛЬНЫЙ из БД, не MAX!)
  → генерация bonus_combos (до 10 на лот, фильтр по приоритетам)
  → _score_option() скоринг
  → список ArtifactOption

build_artifact_sets()
  → для каждого контейнера:
    → regular pool: top-50 unique by score + 20 stacking lots (2nd variants) = 70
    → counter pool: only sprint>0, top-3 cheapest decent per item = 30
    → total: 100 candidates (70 regular + 30 counter)
    → _numba_search.search() — C(100,7) perебор в Numba JIT
    → n_keep=2000 (большой — захватывает carryable combos с низким score)
    → постобработка: container protection, reaction bonuses, min_thresholds, scoring
  → сортировка: carryable first, потом по score
```

### Smart Pool (КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ)
- **Regular pool:** top-50 unique items by score + 20 stacking lots (2nd variant of top-20 items)
- **Counter pool:** ТОЛЬКО контр-арты со sprint > 0 (без sprint=0 бесполезных)
  - Для каждого item: top-1 best sprint + top-3 cheapest with sprint > 1.0
- **Кап:** 70 regular + 30 counter = 100 total → C(100,7) = 16B → ~25s
- **Без smart pool:** Numba не находит оптимальные сборки (контр-арты вытесняются)

### Numba JIT ядра
4 специализации по k (4, 5, 6, 7 слотов). Каждое ядро принимает boolean флаги:
- `allow_dup` — когда True, Numba специализирует код БЕЗ проверок item_id
- `min_budget` — когда True, score = 0 + sum(fw[s]) (максимизация статов без бонуса за бюджет)
- `require_carry` — когда False, неносибельные сборки сохраняются с пенальти

### UI фичи
- **Тумблер "Макс. стат"** — включает min_budget режим (чистая максимизация статов)
- **localStorage** — артефакты "В ХРАН" сохраняются между сессиями (auto-save on change)
- **Выпадающий список** для owned артефактов (106 артефактов по подкатегориям)
- **Чекбоксы бонусов** с указанием порога ptn (ptn≥5, ≥10, ≥15)
- **EHP и HPS** в шапке сборки
- **Пресеты:** PvP, PvE, Жир, Скорость, Перенос, Фарм, Радар

### API эндпоинты
| Метод | Путь | Описание |
|-------|------|----------|
| GET | `/` | HTML UI |
| GET | `/api/stats` | Статистика БД |
| GET | `/api/artefacts` | Список артефактов |
| GET | `/api/artefacts/{id}` | Детали артефакта |
| GET | `/api/containers` | Список контейнеров |
| GET | `/api/builds` | Поиск сборок (все параметры) |
| GET | `/api/all-stats` | Все уникальные stat prop_ids |

### Параметры /api/builds
```
budget (int, default=0)          — макс. цена (0 при min_budget=1)
tolerance (float, default=8.0)   — погрешность бюджета %
container_id (str?)              — ID контейнера (пусто = авто/макс слоты)
top_n (int, default=10)          — сколько сборок вернуть
priorities (JSON)                — {"stat_id": weight_0_to_10}
min_thresholds (JSON)            — {"stat_id": min_value}
owned (JSON)                     — [{"item_id", "quality", "ptn", "art_percent", "bonus_ids"}]
carryable_only (bool)            — только носибельные
allow_duplicates (bool)          — разрешить одинаковые item_id
min_budget (bool)                — режим макс. стата (без бонуса за бюджет)
```

## 2. Математика расчётов (ПРОВЕРЕНА по эталону stalzone.wiki — 13/14 статов совпали)

### Константы
```python
QUALITY_RANGES = {
    0: {"min": 85,  "range": 15},  # Common
    1: {"min": 100, "range": 15},  # Uncommon
    2: {"min": 115, "range": 15},  # Special
    3: {"min": 130, "range": 15},  # Rare
    4: {"min": 145, "range": 15},  # Exceptional
    5: {"min": 160, "range": 15},  # Legendary
}
# art_percent = РЕАЛЬНЫЙ из БД лота (lot["art_percent"]), НЕ MAX диапазона!
# ИСПРАВЛЕНО: раньше использовали MAX → статы были завышены на 7-10%

_RARITY_INDEX = {
    "rarity.unordinary": 0, "rarity.special": 1, "rarity.rare": 2,
    "rarity.exclusive": 3, "rarity.legendary": 4, "rarity.unique": 5,
}
# Все артефакты имеют rarity = "rarity.ordinary" — отсутствует в _RARITY_INDEX → n_steps fallback.

DANGER_THRESHOLDS = {
    "radiation_accumulation":  0.5,
    "biological_accumulation": 0.5,
    "psycho_accumulation":     0.5,
    "thermal_accumulation":    0.5,
    "frost_accumulation":      1.0,  # вдвое выше!
}

STATS_CONT_CAN_PROTECT = {
    "radiation_accumulation", "biological_accumulation",
    "psycho_accumulation", "thermal_accumulation",
    "frost_accumulation",  # ИСПРАВЛЕНО: frost ДОЛЖЕН быть защищён контейнером!
}
# Подтверждено wiki: frost=0.50 для 3x Каблук в контейнере prot=75%.

ACCUMULATION_STATS = STATS_CONT_CAN_PROTECT  # все 5 accumulation статы
```

### Контейнерная защита (на уровне СБОРКИ)
```python
# Применяется ПОСЛЕ суммирования всех артефактов
if _can_protect(full_key):  # в STATS_CONT_CAN_PROTECT (включая frost!)
    if not is_positive and origin == "artefact":
        return val * (1.0 - inner_protection / 100.0)
return val
```
- is_positive ПЕРЕСЧИТЫВАЕТСЯ из суммы: canBePositiveAndNegative → is_positive = (summed <= 0)
- Контейнерные бонусы (origin="container") НЕ защищаются

### apply_reaction_bonuses (ИСПРАВЛЕНО)
```python
# Реакции (reaction_to_burn и др.) — ОТДЕЛЬНЫЙ бонус, НЕ добавляется к health/stamina_regen!
# Wiki показывает health_bonus и reaction_to_burn раздельно.
# Раньше: health = 18.11 (9.33 + 8.74 реакция) → НЕВЕРНО
# Сейчас: health = 9.33, reaction_to_burn = 8.69 → ВЕРНО
# EHP/HPS используют reaction_sum внутренне.
```

### calculate_effective_health (ПРОВЕРЕНО: 142.28 = 142.28 wiki)
```python
EHP = (health_bonus + 100.0) / 100.0 * (100.0 + bullet_dmg_factor)
```

### calculate_healing_per_second (ПРОВЕРЕНО: 0.93 = 0.93 wiki)
```python
HPS = (regeneration_bonus + 2.5) / 5.0 + artefakt_heal * (1.0 + heal_efficiency / 100.0)
```

### Скоринг (_score_option)
```python
for prop_id, value in stats.items():
    weight = priorities.get(prop_id, 0)
    if weight == 0: continue
    if full_key in DANGER_THRESHOLDS:
        score += (-value) * weight  # инверсия: отрицательное = хорошо
    else:
        score += value * weight
# Статы нормализуются: value / stat_max где stat_max = max_single_artifact * n_slots
```

### Budget utilization bonus (в Numba ядре)
```python
# Режим максимизации (min_budget=False):
score = (total_price / budget) * 100.0 if budget > 0 else 0.0
for s in range(ns): score += fw[s]

# Режим макс. стата (min_budget=True):
score = 0.0
for s in range(ns): score += fw[s]  # чистая максимизация статов
```

### Штраф за превышение носибельности (в Numba ядре)
```python
carry = True
for d in range(5):
    v = f[d]
    if v > 0 and d < 4: v *= prot_mult
    if v > _DANGER_THRESH[d] + 0.001:
        carry = False; score -= (v - _DANGER_THRESH[d]) * 1000.0
if not carry and require_carry:
    continue  # пропустить
```

## 3. Лимиты и оптимизация

### Комбинаторные лимиты
| Режим | regular | counter | total | C(n,7) | Время |
|-------|---------|---------|-------|--------|-------|
| dup | 70 | 30 | 100 | 16B | ~25с |
| no-dup | 102 | 15 | 117 | 18.5B | ~12с |

### n_keep = 2000
Большой размер — захватывает carryable combos с более низким score (контр-арты снижают sprint).
С n_keep=300 Numba не находит 14.07; с n_keep=2000 находит.

### Потери данных (ИСПРАВЛЕНО)
1. **SQL dedup:** PARTITION BY item_id, quality, ptn, bonus_properties + rn <= 2
2. **group_list:** без лимита — все лоты грузятся
3. **bonus_combos[:10]:** до 10 на лот
4. **art_percent:** РЕАЛЬНЫЙ из БД, не MAX (было завышение 7-10%)

## 4. Известные ограничения

### frost_accumulation
- frost защищается контейнером (в STATS_CONT_CAN_PROTECT)
- frost нет редукторов в игре → ограничивает стакинг (2× Вихрь = frost 0.92 → safe)

### Контр-арты со sprint
- Вихрь (sprint=2.21, thermal=-0.31, frost=0.46) [C]
- Сетчатка (sprint=1.99, bio=-2.04, psycho=1.16) [C]
- Силки (sprint=1.21, psycho=-1.76) [C]
- Эти 3 артa критичны для carryable sprint builds

### Owned артефакты
- base_stats и bonus_stats ДОЛЖНЫ быть переданы в ArtifactOption
- localStorage сохраняет между сессиями (auto-save on change)

### Броня (только UI)
- 4 поля: не передаётся в API, не влияет на скоринг

## 5. Режимы (пресеты в UI)
```javascript
pvp:    {bullet_dmg_factor: 10, health_bonus: 8, stamina_bonus: 5, stopping_protection: 5, tear_dmg_factor: 5}
pve:    {bullet_dmg_factor: 5, health_bonus: 8, regeneration_bonus: 8, thermal_protection: 5, psycho_protection: 5, biological_protection: 5}
tank:   {health_bonus: 10, regeneration_bonus: 10, heal_efficiency: 8, artefakt_heal: 8, bullet_dmg_factor: 7, stamina_bonus: 5}
speed:  {speed_modifier: 10, sprint_speed_modifier: 10, stamina_bonus: 8, max_weight_bonus: 8, stamina_regeneration_bonus: 5}
carry:  {max_weight_bonus: 10, bullet_dmg_factor: 5, health_bonus: 5, stamina_bonus: 5}
farm:   {max_weight_bonus: 10, speed_modifier: 8, sprint_speed_modifier: 8, stamina_bonus: 5, health_bonus: 5}
radar:  {tear_dmg_factor: 10, explosion_dmg_factor: 8, stopping_protection: 5, health_bonus: 5}
custom: {}
```

## 6. UI контракт
| UI поле | API параметр | Тип |
|---------|-------------|------|
| Бюджет | budget | int (0 при maxStat) |
| Погрешность | tolerance | float |
| Контейнер | container_id | str? |
| Топ сборок | top_n | int |
| Носибельность | carryable_only | bool |
| Дубликаты | allow_duplicates | bool |
| Макс. стат (тумблер) | min_budget | bool |
| Приоритеты (0-10) | priorities | JSON str |
| Мин. пороги | min_thresholds | JSON str |
| Хранилище | owned | JSON str (localStorage) |
| Броня | НЕ передаётся | applyArmor() в JS |
| Режим | НЕ передаётся | applyMode() |

## 7. Проверка
Эталон stalzone.wiki (3× Каблук + 2× Ветка + Протоцибуля в ZIVCAS ArcticSafe-6):
```
health_bonus = 9.33  (wiki: 9.33) ✓
heal_efficiency = 180.67  (wiki: 180.67) ✓
bullet_dmg_factor = 30.14  (wiki: 30.14) ✓
stopping_protection = 51.34  (wiki: 51.34) ✓
EHP = 142.28  (wiki: 142.28) ✓
HPS = 0.93  (wiki: 0.93) ✓
psycho_accumulation = 0.50  (wiki: 0.50) ✓  (от контейнера)
thermal_accumulation = -2.17  (wiki: -2.17) ✓
radiation_accumulation = -1.32  (wiki: -1.32) ✓
```
13 из 14 статов совпали ТОЧНО. Расхождение только в frost (контейнерный frost-бонус).

## 8. Правила работы
1. **Не трогать:** config.py, database.py, api_client.py, wiki_fetcher.py, github_fetcher.py, collector.py
2. **Математика верна:** формулы compute_artifact_stats проверены по wiki — не перепроверять
3. **Numba:** при изменении _numba_search.py — очистить `__pycache__/` и `~/.numba/`
4. **art_percent:** использовать РЕАЛЬНЫЙ из БД (lot["art_percent"]), НЕ MAX
5. **frost_accumulation:** в STATS_CONT_CAN_PROTECT (контейнер защищает)
6. **apply_reaction_bonuses:** НЕ модифицирует health_bonus/stamina_regen
7. **Сервер:** uvicorn app.main:app --host 127.0.0.1 --port 8123
8. **Данные:** регион EU, только активные лоты (buyout_price > 0)
