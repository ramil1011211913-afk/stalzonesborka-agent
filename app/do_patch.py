import re
import shutil
from pathlib import Path

p = Path("app/builder.py") if Path("app/builder.py").exists() else Path("builder.py")
bak = p.with_name("builder.py.bak")

# Восстанавливаем оригинал из бэкапа
if bak.exists():
    shutil.copy(bak, p)
    print("[1/4] Файл восстановлен из бэкапа")

content = p.read_text(encoding="utf-8")

# 1. Функция _get_max_carryable_count (без явного типа ArtifactOption, чтобы избежать NameError)
t1 = """def _is_counter_art(opt) -> bool:
    \"\"\"Check if an option is a counter-art: has negative accumulation (protection).\"\"\"
    for pid, val in opt.stats.items():
        fkey = _to_full_key(pid)
        if fkey in DANGER_THRESHOLDS and val < 0:
            return True
    return False"""

r1 = t1 + """

def _get_max_carryable_count(opt, n_slots: int) -> int:
    \"\"\"Возвращает максимальное кол-во штук артефакта без перегруза по заражению.\"\"\"
    max_count = n_slots
    for pid, val in opt.stats.items():
        fkey = _to_full_key(pid)
        if fkey in DANGER_THRESHOLDS and val > 0:
            limit = DANGER_THRESHOLDS[fkey]
            allowed = int(limit // val) if val > 0 else n_slots
            max_count = min(max_count, max(1, allowed))
    return max_count"""

if "_get_max_carryable_count" not in content:
    if t1 in content:
        content = content.replace(t1, r1, 1)
        print("[2/4] Функция расчета лимита штучности добавлена")

# 2. Исправление stat_max
t2 = """        for pid in priorities:
            fkey = _to_full_key(pid)
            vals = []
            _seen_iids: set[str] = set()
            for o in sorted(pool + owned_opts, key=lambda o: -o.stats.get(pid, 0)):
                if o.item_id in _seen_iids:
                    continue
                v = o.stats.get(pid, 0)
                if fkey in DANGER_THRESHOLDS:
                    v = -v
                if v > 0:
                    _seen_iids.add(o.item_id)
                    vals.append(v)
            vals.sort(reverse=True)
            stat_max[pid] = sum(vals[:n_slots]) if vals else 1.0"""

r2 = """        for pid in priorities:
            fkey = _to_full_key(pid)
            vals = []
            _seen_iids: set[str] = set()
            for o in sorted(pool + owned_opts, key=lambda o: -o.stats.get(pid, 0)):
                if o.item_id in _seen_iids:
                    continue
                v = o.stats.get(pid, 0)
                if fkey in DANGER_THRESHOLDS:
                    v = -v
                if v > 0:
                    _seen_iids.add(o.item_id)
                    max_pcs = _get_max_carryable_count(o, n_slots) if fkey not in DANGER_THRESHOLDS else 1
                    vals.extend([v] * max_pcs)
            vals.sort(reverse=True)
            stat_max[pid] = sum(vals[:n_slots]) if vals else 1.0"""

if t2 in content:
    content = content.replace(t2, r2, 1)
    print("[3/4] Расчет stat_max обновлен")

# 3. Расширение кандидатов для живучести
t3 = """        _prio_keys = list(priorities.keys())
        regular = [o for o in pool if
                   any(o.stats.get(p, 0) > 0 for p in _prio_keys) or not
                   any(o.stats.get(d, 0) < -0.5 for d in
                       ("psycho_accumulation", "biological_accumulation",
                        "thermal_accumulation", "radiation_accumulation"))]"""

r3 = """        _prio_keys = list(priorities.keys())
        if "health_bonus" in priorities:
            _prio_keys.extend(["heal_efficiency", "regeneration_bonus", "artefakt_heal"])
        regular = [o for o in pool if
                   any(o.stats.get(p, 0) > 0 for p in _prio_keys) or not
                   any(o.stats.get(d, 0) < -0.5 for d in
                       ("psycho_accumulation", "biological_accumulation",
                        "thermal_accumulation", "radiation_accumulation"))]"""

if t3 in content:
    content = content.replace(t3, r3, 1)
    print("[4/4] Кандидаты выживаемости расширены")

# 4. Мягкие штрафы soft penalty
content = re.sub(
    r"^(\s*)score -= \(val / threshold\) \* 0\.5",
    r"\1w = priorities.get(prop_id, 5.0)\n\1score -= (val / threshold) * w * 0.3",
    content,
    flags=re.MULTILINE
)
content = re.sub(
    r"^(\s*)total_score -= \(val / threshold\) \* 0\.5",
    r"\1w = priorities.get(prop_id, 5.0)\n\1total_score -= (val / threshold) * w * 0.3",
    content,
    flags=re.MULTILINE
)

p.write_text(content, encoding="utf-8")
print("\n[УСПЕХ] Файл builder.py успешно обновлен без ошибок NameError!")