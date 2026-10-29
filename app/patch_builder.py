python -c "
code = '''import shutil
from pathlib import Path

builder_path = Path(\"app/builder.py\")
if not builder_path.exists():
    print(\"[ERROR] Файл app/builder.py не найден! Убедись, что находишься в папе C:\\\\myprofrma\\\\stalzonesborka\")
    exit(1)

# Резервная копия
backup_path = Path(\"app/builder.py.bak_patch\")
shutil.copy(builder_path, backup_path)
print(f\"[1/5] Создана резервная копия: {backup_path}\")

content = builder_path.read_text(encoding=\"utf-8\")

# 1. Добавление функции _get_max_carryable_count
target_1 = \"\"\"def _is_counter_art(opt) -> bool:
    \"\"\"Check if an option is a counter-art: has negative accumulation (protection).\"\"\"
    for pid, val in opt.stats.items():
        fkey = _to_full_key(pid)
        if fkey in DANGER_THRESHOLDS and val < 0:
            return True
    return False\"\"\"

replacement_1 = target_1 + \"\"\"

def _get_max_carryable_count(opt: ArtifactOption, n_slots: int) -> int:
    \"\"\"Возвращает максимальное кол-во штук артефакта без перегруза по заражению.\"\"\"
    max_count = n_slots
    for pid, val in opt.stats.items():
        fkey = _to_full_key(pid)
        if fkey in DANGER_THRESHOLDS and val > 0:
            limit = DANGER_THRESHOLDS[fkey]
            allowed = int(limit // val) if val > 0 else n_slots
            max_count = min(max_count, max(1, allowed))
    return max_count\"\"\"

if \"_get_max_carryable_count\" not in content:
    if target_1 in content:
        content = content.replace(target_1, replacement_1, 1)
        print(\"[2/5] Добавлена функция расчета лимита штучности артефактов\")
    else:
        print(\"[WARN] Точка вставки _is_counter_art не найдена\")
else:
    print(\"[INFO] Функция _get_max_carryable_count уже существует\")

# 2. Исправление stat_max
target_2 = \"\"\"        for pid in priorities:
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
            stat_max[pid] = sum(vals[:n_slots]) if vals else 1.0\"\"\"

replacement_2 = \"\"\"        for pid in priorities:
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
            stat_max[pid] = sum(vals[:n_slots]) if vals else 1.0\"\"\"

if target_2 in content:
    content = content.replace(target_2, replacement_2, 1)
    print(\"[3/5] Обновлен расчет нормализации stat_max\")
else:
    print(\"[WARN] Блок stat_max не найден\")

# 3. Расширение кандидатов
target_3 = \"\"\"        _prio_keys = list(priorities.keys())
        regular = [o for o in pool if
                   any(o.stats.get(p, 0) > 0 for p in _prio_keys) or not
                   any(o.stats.get(d, 0) < -0.5 for d in
                       (\"psycho_accumulation\", \"biological_accumulation\",
                        \"thermal_accumulation\", \"radiation_accumulation\"))]\"\"\"

replacement_3 = \"\"\"        _prio_keys = list(priorities.keys())
        if \"health_bonus\" in priorities:
            _prio_keys.extend([\"heal_efficiency\", \"regeneration_bonus\", \"artefakt_heal\"])
        regular = [o for o in pool if
                   any(o.stats.get(p, 0) > 0 for p in _prio_keys) or not
                   any(o.stats.get(d, 0) < -0.5 for d in
                       (\"psycho_accumulation\", \"biological_accumulation\",
                        \"thermal_accumulation\", \"radiation_accumulation\"))]\"\"\"

if target_3 in content:
    content = content.replace(target_3, replacement_3, 1)
    print(\"[4/5] Расширен выбор кандидатов для режима выживаемости\")
else:
    print(\"[WARN] Блок кандидатов не найден\")

# 4. Штрафы soft penalty
content = content.replace(\"score -= (val / threshold) * 0.5\", \"w = priorities.get(prop_id, 5.0)\\n                            score -= (val / threshold) * w * 0.3\")
content = content.replace(\"total_score -= (val / threshold) * 0.5\", \"w = priorities.get(prop_id, 5.0)\\n                total_score -= (val / threshold) * w * 0.3\")
print(\"[5/5] Настроены динамические мягкие штрафы пороговых статов\")

builder_path.write_text(content, encoding=\"utf-8\")
print(\"\\n[SUCCESS] Файл app/builder.py успешно обновлен!\")
'''
Path('patch_builder.py').write_text(code, encoding='utf-8')
print('Создан файл patch_builder.py')
"
