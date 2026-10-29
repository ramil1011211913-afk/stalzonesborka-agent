# Экспертная инструкция: дальнейшая работа над оптимизатором

## Контекст: что нашли при тесте «пулестой живучесть вес 50-60, бюджет 10М, бочонок»

Build #1: Score=1.370, EHP=136.7, Price=10.28М
- Бубльгум (bullet=31, thermal=2.5) — 3.65М
- Яйцо ×2 (weight=28+28=55.5, radiation=2.5+2.5) — 1.58М
- Морозец (health=4.33, frost=1.0) — 3.33М
- Дезинтегратор + Протоцибуля + Чертов еж — филлеры

Все 5 сборок: ядро Бубльгум+Яйцо×2+Морозец одинаковое, отличаются только 1-2 филлера.

---

## Фаза 1: stat_max — нереалистичная нормализация (КРИТИЧНО)

**Проблема:** `stat_max[pid] = max_single_artifact × n_slots` (стр. 1112).
Для health_bonus: если Морозец единственный health-арт в пуле, `stat_max = 4.33 × 7 = 30.33`.
Но 7 Морозец невозможно носить — frost = 7.0, предел 1.0.

**Эффект:** EHP-скординг делит на `max_ehp_gain`, вычисленный из нереалистичных stat_max.
Score = `ehp_gain / max_ehp_gain × weight` = 36.7 / (огромное число) = мало → score=1.37.

**Решение:** stat_max должен учитывать carryable constraints.
Варианты:
- A) `stat_max = max_single_artifact × min(n_slots, carryable_max_for_stat)`.
  Для frost-статов: carryable_max = threshold / per_art_value (Морозец frost=1.0 → max 1 арт).
- B) `stat_max = sum of top-n_slots artifacts by stat value` (реалистичный максимум).
- C) Использовать EHP gain от best-case build, а не от impossible theoretical max.

**Рекомендация:** B — `stat_max = sum of top-n_slots artifact values for this stat`.
Это даёт реалистичный верхний предел, достижимый в принципе.

---

## Фаза 2: frost_accumulation=1.0 — нет soft penalty (ВАЖНО)

**Проблема:** Морозец имеет frost=1.0 (точный максимум носибельности).
- `check_carryability`: `val > threshold + 0.001` → 1.0 > 1.001 = False → carryable ✓
- В scoring: frost=1.0 инвертируется в -1.0 → «хороший» стат. Но 1.0 — это ПРЕДЕЛ, не «хорошо».
- Hard penalty (стр. 1337-1338): `if val > threshold: score -= (val - threshold) × 1000`.
  Но 1.0 не > 1.0 → нет штрафа.

**Эффект:** Оптимизатор не отличает frost=0.01 (безопасно) от frost=1.0 (на пределе).
Выбор Морозец = frost=1.0 = блокировка ВСЕХ других frost-артов.
Оптимизатор не знает об этом ограничении при выборе кандидатов.

**Решение:** Добавить soft penalty в `_score_build`:
```python
# Near-threshold penalty: builds at 80-100% of danger threshold get diminishing score
if fkey in DANGER_THRESHOLDS:
    threshold = DANGER_THRESHOLDS[fkey]
    if 0 < val <= threshold:
        score -= (val / threshold) * weight * 0.3  # linear penalty up to 30% of weight
```
Это даст приоритет сборкам с меньшим frost, даже если все carryable.

---

## Фаза 3: Пул кандидатов — мало diversity для health (ВАЖНО)

**Проблема:** `load_artifact_options` берёт top-N артефактов по каждому priority-стату (стр. 1179).
Для health_bonus: если в БД мало артефактов с health > 0, пул сужается до 1-2 вариантов.

**Эффект:** Морозец — единственный сильный health-арт. Без альтернативы все сборки идут через Морозец.
Нет diversity → нет альтернативных стратегий (например, 2 дешёвых health-арта вместо 1 дорогого).

**Решение:**
1. Проверить сколько артефактов в БД имеют health_bonus > 0:
   `SELECT item_id, name_ru, stats->'health_bonus' FROM artefacts WHERE ...`
2. Если мало — рассмотреть arts с косвенным health (heal_efficiency, regeneration_bonus).
3. Добавить в пул arts с health > 0 даже если они не top-N по score, но имеют уникальные combo (health+counter, health+weight).

---

## Фаза 4: Score=1.37 — правильный ли EHP-скординг?

**Проверка формулы:**
```python
# Текущий EHP scoring (стр. 478-492):
ehp = (1 + health/100) × (100 + bullet)  # = (1 + 4.33/100) × 131.02 = 136.7
ehp_gain = ehp - 100  # = 36.7
max_ehp = (1 + stat_max_health/100) × (100 + stat_max_bullet)  # нереалистично большой
max_ehp_gain = max_ehp - 100
score += (ehp_gain / max_ehp_gain) × (hp_w + bul_w) / 2  # ≈ 36.7 / huge × 10 ≈ мало
```

**Проблема:** max_ehp_gain огромен из-за stat_max = max_single × 7.
Реальный max EHP для 7-slot build с carryable constraints ≈ 200-250 (не 500+).

**Решение:** После фикса stat_max (Фаза 1) score должен стать реалистичнее.
Проверить: score лучшей сборки должен быть в диапазоне 5-15, не 1.37.

---

## Фаза 5: Дополнительные улучшения (после основных фиксов)

### 5.1 Price efficiency в скоринге
Дешёвый филлер (Протоцибуля 14К, Кисель 9К) занимает слот, не давая priority-статов.
Скоринг должен учитывать opportunity cost слота:
```python
# Штраф за слот без priority-статов
slot_efficiency = sum(stats.get(p, 0) for p in priorities) / max_single_priority
if slot_efficiency < 0.05:
    score -= 0.5  # пустой слот
```

### 5.2 Diversity в результатах
5 одинаковых сборок — бесполезно. Нужно:
- Pareto frontier уже есть (стр. 566-595), но не работает: все сборки доминируют друг друга по разным статам.
- Добавить constraint: в top-10 результатов не более 3 сборок с одинаковым набором из 5+ общих артефактов.

### 5.3 Обновление аукциона
Проверить актуальность данных в stalzone.db:
```sql
SELECT MAX(created_at) FROM auction_lots;
SELECT COUNT(*) FROM auction_lots WHERE buyout_price > 0;
```
Если данные устарели — запустить collector.py для обновления.

---

## Порядок выполнения

1. **Фаза 1** — фикс stat_max (без него score всегда будет занижен)
2. **Фаза 2** — soft penalty для frost/danger stats (Морозец блокирует сборку)
3. **Фаза 3** — расширение пула health-кандидатов (diversity)
4. **Фаза 4** — верификация score после фаз 1-3
5. **Фаза 5** — улучшения (price efficiency, diversity, auction refresh)

Каждую фазу: тест на том же запросе «пулестой живучесть вес 50-60, 10М, бочонок».
Ожидаемый результат: score > 5, diverse сборки, ни одна не на frost=1.0.
