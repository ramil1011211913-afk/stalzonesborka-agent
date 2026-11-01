"""
FastAPI web server — REST API + web interface for Stalzone artifact collector.

Endpoints:
  GET  /                        — Web UI (HTML)
  GET  /api/stats               — Database statistics
  GET  /api/artefacts           — List all artifacts
  GET  /api/artefacts/{id}      — Single artifact detail with bonus stats
  GET  /api/containers          — List all containers/backpacks
  GET  /api/builds              — Find artifact builds within budget
"""
import json
import logging
from pathlib import Path
import threading

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .builder import build_artifact_sets, check_auction_freshness

# Enabled only after the separate experimental release gate passes.
MAX_QUALITY_RELEASED = Path(__file__).with_name('_max_quality_release.json').is_file()
from .config import Config
from .database import get_artefact_stats_summary, get_connection, init_db
from ._app_state import application_store, auction_job, DEFAULT_LEASE_SECONDS

log = logging.getLogger(__name__)

app = FastAPI(title="Stalzone Artifact Builder", version="2.0.0")

init_db()

STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

_llm_system_prompt: str | None = None


def _persist_owned_artifacts(owned):
    """Shared application-session state, committed atomically across workers."""
    application_store(Config.DATA_DIR).save_owned(owned)


@app.get('/api/owned-artifacts')
async def owned_artifacts():
    store = application_store(Config.DATA_DIR)
    try:
        store.import_legacy(Path(Config.DATA_DIR) / 'last_owned.json')
    except (ValueError, UnicodeError) as error:
        return JSONResponse(status_code=422, content={'error': str(error)})
    return store.owned()


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent / "templates" / "index.html"
    if html_path.exists():
        return HTMLResponse(
            html_path.read_text(encoding="utf-8").replace(
                '<option value="max_quality">Максимальное качество (MILP)</option>',
                '<option value="max_quality">Максимальное качество (MILP)</option>' if MAX_QUALITY_RELEASED else ''),
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )
    return HTMLResponse("<h1>Stalzone Artifact Builder</h1>")


@app.get("/favicon.ico")
async def favicon():
    """Return empty favicon to avoid 404."""
    from fastapi.responses import Response
    return Response(content=b"", media_type="image/x-icon")

@app.get("/.well-known/appspecific/com.chrome.devtools.json")
async def chrome_devtools():
    """Return empty DevTools config to avoid Chrome 404 noise."""
    return {}

@app.get("/api/stats")
async def api_stats():
    return get_artefact_stats_summary()


@app.get("/api/artefacts")
async def api_artefacts(subcategory: str | None = None):
    conn = get_connection()
    if subcategory:
        rows = conn.execute(
            "SELECT id, name_ru, name_en, subcategory, color FROM artefacts WHERE subcategory=? ORDER BY name_ru",
            (subcategory,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, name_ru, name_en, subcategory, color FROM artefacts ORDER BY subcategory, name_ru"
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.get("/api/artefacts/{item_id}")
async def api_artefact_detail(item_id: str):
    conn = get_connection()
    art = conn.execute(
        "SELECT id, name_ru, name_en, subcategory, color, stats, bonus_stats, icon_url FROM artefacts WHERE id=?",
        (item_id,),
    ).fetchone()
    if not art:
        return JSONResponse({"error": "not found"}, status_code=404)

    # Price summary per quality from active lots
    lots_summary = conn.execute("""
        SELECT quality, MIN(buyout_price) as min_price, COUNT(*) as lot_count
        FROM auction_lots WHERE item_id=? AND region=?
        GROUP BY quality ORDER BY quality
    """, (item_id, Config.REGION)).fetchall()

    result = dict(art)
    result["stats"] = json.loads(result["stats"]) if result["stats"] else []
    result["bonus_stats"] = json.loads(result["bonus_stats"]) if result["bonus_stats"] else []
    result["lots_by_quality"] = [dict(r) for r in lots_summary]
    conn.close()
    return result


@app.get("/api/containers")
async def api_containers(item_type: str | None = None, min_slots: int | None = None):
    conn = get_connection()
    query = "SELECT id, name_ru, name_en, item_type, color, weight, slots, inner_protection, effectiveness, bonus_stats, icon_url FROM containers"
    conditions = []
    params = []
    if item_type:
        conditions.append("item_type=?")
        params.append(item_type)
    if min_slots is not None:
        conditions.append("slots>=?")
        params.append(min_slots)
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY slots DESC, name_ru"
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.get("/api/armors")
async def api_armors(category: str | None = None):
    """List all available armors from DB."""
    conn = get_connection()
    if category:
        rows = conn.execute("SELECT * FROM armors WHERE category=? ORDER BY name_ru", (category,)).fetchall()
    else:
        rows = conn.execute("SELECT * FROM armors ORDER BY category, name_ru").fetchall()
    import json as _j
    result = []
    for r in rows:
        result.append({
            "id": r["id"], "name_ru": r["name_ru"], "category": r["category"],
            "rank": r["rank"], "level": r["level"],
            "stats": _j.loads(r["stats"]) if r["stats"] else {},
            "icon_url": r["icon_url"],
        })
    conn.close()
    return result

@app.get("/api/builds")
def api_builds(
    budget: int = Query(0, description="Artifact purchase cap; armor/container excluded; min_budget with zero budget removes cap", ge=0),
    tolerance: float = Query(8.0, ge=0, le=50, description="Budget tolerance %"),
    container_id: str | None = Query(None, description="Specific container ID"),
    container_ids: str | None = Query(None, description='JSON list of owned container IDs, e.g. ["rgoz","v667"]'),
    top_n: int = Query(10, ge=1, le=50, description="Top N builds"),
    # Priority sliders 0-10 — pass as JSON string for flexibility
    priorities: str | None = Query(None, description='JSON: {"health_bonus": 10, "bullet_dmg_factor": 8}'),
    min_thresholds: str | None = Query(None, description='JSON: {"health_bonus": 15}'),
    max_thresholds: str | None = Query(None, description='JSON: {"max_weight_bonus": 60}'),
    owned: str | None = Query(None, description='JSON: [{"item_id":"ljn2","quality":3,"ptn":15,"art_percent":137.5,"bonus_ids":["HEALTH_BONUS"]}]'),
    carryable_only: bool = Query(True, description="Filter to carryable builds only"),
    allow_duplicates: bool = Query(False, description="Allow same item_id multiple times"),
    min_budget: bool = Query(False, description="Minimize price subject to constraints; budget=0 removes the cap"),
    conservative: bool = Query(False, description="Backward-compatible alias for unknown_percent=minimum"),
    unknown_percent: str | None = Query(None, pattern="^(minimum|midpoint|maximum)$",
        description="Point used only for unknown quality percentages: minimum (guarantee), midpoint (expected) or maximum (optimistic); known lot percentages stay actual"),
    guaranteed_carryable: bool = Query(False, description="Only builds carryable at MAX art% of tier"),
    roll_optimize: bool = Query(False, description="Plan exact hundredth rolls for unknown percentages; maximum/midpoint start at tier maximum, minimum keeps the guaranteed minimum baseline"),
    exclude: str | None = Query(None, description='JSON: ["y5ro"] — item_ids to exclude from builds'),
    prompt_text: str | None = Query(None, description="Original natural language prompt from AI parser"),
    armor_id: str | None = Query(None, description="Armor ID (e.g. rwgmg for Гончий). Stats added to builds."),
    armor_ids: str | None = Query(None, description="JSON alternative armor IDs"),
    armor_level: int = Query(15, ge=0, le=15),
    force_refresh: bool = Query(False, description="Force auction data refresh before building"),
    optimization_mode: str = Query("balanced", pattern="^(balanced|lexicographic|min_budget)$"),
    priority_order: str | None = Query(None, description="JSON ordered stat list for lexicographic mode"),
    objective_scales: str | None = Query(None, description="JSON optional fixed positive stat scales"),
    objective_tolerance: float = Query(1e-6, gt=0),
    min_budget_timeout: float = Query(30., ge=0, le=30),
    max_quality: bool = Query(False, description="Explicit experimental MILP objective maximizer"),
    quality_timeout: float = Query(45., gt=0, le=300),
    deep_optimization: bool = Query(False),
    stock_only: bool = Query(False, description="Use only unupgraded (+0) auction and owned artifacts"),
    primary_only: bool = Query(False, description="Deep pure EHP only; price is a constraint, not an objective"),
):
    """Find optimal artifact builds within budget.

    Priorities: {prop_id: weight_0_to_10} — stats to maximize.
    Min thresholds: {prop_id: min_value} — builds below filtered out.
    Owned: list of {item_id, quality, ptn, art_percent, bonus_ids} — price=0.

    Synchronous (not async) so FastAPI runs it in a threadpool, preventing
    event-loop blocking during long combinatorial searches.
    """
    try:
        if deep_optimization and not max_quality:
            raise ValueError('Deep optimization requires max_quality=true')
        from ._quality_time import request_seconds
        quality_timeout = request_seconds(quality_timeout, deep_optimization)
        prio = json.loads(priorities) if priorities else {}
        mins = json.loads(min_thresholds) if min_thresholds else {}
        maxs = json.loads(max_thresholds) if max_thresholds else {}
        owned_list = json.loads(owned) if owned else []
        exclude_list = json.loads(exclude) if exclude else []
        order = json.loads(priority_order) if priority_order else None
        scales = json.loads(objective_scales) if objective_scales else None
        armor_list = json.loads(armor_ids) if armor_ids else None
        container_ids_list = None
        if container_ids is not None:
            parsed_containers = json.loads(container_ids)
            if (not isinstance(parsed_containers, list) or not parsed_containers or
                    any(not isinstance(cid, str) or not cid.strip() for cid in parsed_containers)):
                raise ValueError("container_ids must be a nonempty JSON list of container IDs")
            container_ids_list = []
            for cid in parsed_containers:
                cid = cid.strip()
                if cid not in container_ids_list:
                    container_ids_list.append(cid)
        if container_id == "":
            container_id = None
        if container_id is not None and container_ids_list is not None:
            raise ValueError("Choose container_id or container_ids")
        from ._quality_scenario import resolve_mode, roll_baseline
        unknown_mode = resolve_mode(conservative, unknown_percent)
        if roll_optimize and min_budget:
            raise ValueError('roll_optimize is not supported with min_budget')
        if roll_optimize and guaranteed_carryable:
            raise ValueError('roll_optimize cannot be combined with guaranteed_carryable')
        effective_unknown = roll_baseline(unknown_mode, roll_optimize)
        if armor_list is not None and (not isinstance(armor_list, list) or not armor_list or
                                      any(a is not None and not isinstance(a, str) for a in armor_list)):
            raise ValueError("armor_ids must be a nonempty JSON list of IDs or null")
        if armor_id is not None and armor_list is not None:
            raise ValueError("Choose armor_id or armor_ids")
        from ._objective import Objective
        mode = "min_budget" if min_budget else optimization_mode
        Objective.create(prio, scales, mode, order, objective_tolerance)
        from ._survivability import validate as validate_survivability
        validate_survivability(prio, mins, maxs, max_quality=max_quality, mode=mode)
        from ._ehp_primary_only import validate_request
        validate_request(primary_only, deep_optimization, max_quality, prio, mode, min_budget)
        if primary_only and tolerance != 0:
            raise ValueError('primary_only requires tolerance=0: strict purchase budget')
        if not isinstance(owned_list, list) or any(not isinstance(a, dict) for a in owned_list):
            raise ValueError('Owned artifacts must be a JSON list of records')
        json.dumps(owned_list, allow_nan=False)
    except (ValueError, TypeError) as e:
        return JSONResponse(
            status_code=422,
            content={"error": str(e)},
        )

    # Validate owned artifact art_percent range
    # Validate owned artifact art_percent range — default to quality tier max if 0/missing
    for oa in owned_list:
        ap = oa.get("art_percent", 0)
        oa["_art_percent_known"] = isinstance(ap, (int, float)) and ap != 0
        if not isinstance(ap, (int, float)) or ap == 0:
            # Default: max of quality tier
            from .api_client import QUALITY_RANGES
            qr = QUALITY_RANGES.get(oa.get("quality", 0), QUALITY_RANGES[0])
            oa["art_percent"] = qr["min"] + qr["range"]
        elif ap < 85 or ap > 200:
            return JSONResponse(
                status_code=422,
                content={"error": f"Invalid art_percent {ap} for owned artifact {oa.get('item_id', '?')}"},
            )

    # ── Auto-refresh stale auction data ──
    freshness = check_auction_freshness(Config.REGION)
    auction_refreshing = False
    refresh = application_store(Config.DATA_DIR).status(auction_job(Config.REGION))
    if not freshness["is_fresh"] or force_refresh or refresh['state'] == 'expired':
        _start_auction_refresh()
    auction_refreshing = application_store(Config.DATA_DIR).status(auction_job(Config.REGION))['running']

    try:
        engine = build_artifact_sets
        quality_options = {}
        if primary_only:
            quality_options['primary_only'] = True
        if stock_only:
            quality_options['stock_only'] = True
        if max_quality:
            if not MAX_QUALITY_RELEASED:
                return JSONResponse(status_code=422, content={"error": "Experimental quality gate has not passed"})
            from ._max_quality import build_max_quality_sets
            engine = build_max_quality_sets
            quality_options['quality_timeout'] = quality_timeout
            if deep_optimization:
                quality_options['deep_optimization'] = True
        builds = engine(
            budget=budget,
            tolerance_pct=tolerance,
            priorities=prio,
            min_thresholds=mins,
            max_thresholds=maxs,
            top_n=top_n,
            region=Config.REGION,
            container_id=container_id,
            container_ids=container_ids_list,
            owned_artifacts=owned_list,
            carryable_only=carryable_only,
            allow_duplicates=allow_duplicates,
            min_budget=min_budget,
            conservative=(unknown_mode == 'minimum'),
            unknown_percent=unknown_mode,
            roll_optimize=roll_optimize,
            guaranteed_carryable=guaranteed_carryable,
            exclude_item_ids=exclude_list,
            optimization_mode=mode, priority_order=order,
            objective_scales=scales, objective_tolerance=objective_tolerance,
            armor_id=armor_id, armor_ids=armor_list, armor_level=armor_level,
            min_budget_timeout=min_budget_timeout,
            **quality_options,
        )
    except (ValueError, TypeError) as error:
        return JSONResponse(status_code=422, content={"error": str(error)})


    if owned is not None:
        _persist_owned_artifacts(owned_list)

    # ── Full log: request params + all builds with artifacts and stats ──
    import datetime as _dt, os as _os
    _log_dir = _os.path.join(_os.path.dirname(__file__), "..", "data", "build_logs")
    _os.makedirs(_log_dir, exist_ok=True)
    _ts = _dt.datetime.now()
    _log_entry = {
        "timestamp": _ts.isoformat(),
        "request": {
            "budget": budget,
            "effective_budget": int(budget * (1 + tolerance / 100)),
            "tolerance_pct": tolerance,
            "container_id": container_id,
            "container_ids": container_ids_list,
            "container_selection": ("owned" if container_ids_list else
                                    "single" if container_id else "auto"),
            "top_n": top_n,
            "priorities": prio,
            "optimization_mode": mode,
            "priority_order": order,
            "min_thresholds": mins,
            "max_thresholds": maxs,
            "owned_count": len(owned_list),
            "owned_artifacts": owned_list,
            "carryable_only": carryable_only,
            "allow_duplicates": allow_duplicates,
            "min_budget": min_budget,
            "conservative": unknown_mode == 'minimum',
            "unknown_percent": effective_unknown,
            "requested_unknown_percent": unknown_mode,
            "roll_optimize": roll_optimize,
            "roll_optimize_baseline": effective_unknown if roll_optimize else None,
            "exclude": exclude_list,
            "prompt_text": prompt_text or "",
            "armor_id": armor_id, "armor_ids": armor_list, "armor_level": armor_level,
            "budget_scope": "artifacts",
            "region": Config.REGION,
            "guaranteed_carryable": guaranteed_carryable,
            "objective_scales": scales, "objective_tolerance": objective_tolerance,
            "min_budget_timeout": min_budget_timeout,
        },
        "result": {
            "search_status": getattr(builds, 'search_status', None),
            "roll_plan_search": next((b.roll_plan_search for b in builds if b.roll_plan_search), None),
            "builds_found": len(builds),
            "builds": [b.to_dict() for b in builds],
        },
    }
    # Filename: YYYY-MM-DD_HH-MM-SS_<budget>_<builds>builds.json
    if max_quality:
        _log_entry['request'].update(max_quality=True, quality_timeout=quality_timeout)
        if deep_optimization:
            _log_entry['request']['deep_optimization'] = True
    if primary_only:
        _log_entry['request']['primary_only'] = True
    if stock_only:
        _log_entry['request']['stock_only'] = True
    _prio_short = "_".join(f"{k}{v}" for k, v in sorted(prio.items(), key=lambda x: -x[1])[:3]) if prio else "default"
    import uuid as _uuid
    _log_name = f"{_ts.strftime('%Y-%m-%d_%H-%M-%S-%f')}_{_uuid.uuid4().hex}_{budget//1000000}M_{len(builds)}b_{_prio_short}.json"
    _log_path = _os.path.join(_log_dir, _log_name)
    with open(_log_path, "x", encoding="utf-8") as _lf:
        json.dump(_log_entry, _lf, ensure_ascii=False, indent=2)
    log.info("Build log saved to %s (%d builds)", _log_path, len(builds))

    return {
        "budget": budget,
        "effective_budget": int(budget * (1 + tolerance / 100)),
        "tolerance_pct": tolerance,
        "armor_id": armor_id,
        "container_id": container_id,
        "container_ids": container_ids_list,
        "container_selection": ("owned" if container_ids_list else
                                "single" if container_id else "auto"),
        "priorities": prio,
        "optimization_mode": mode,
        "min_thresholds": mins,
        "max_thresholds": maxs,
        "owned_count": len(owned_list),
        "exclude": exclude_list,
        "conservative": unknown_mode == 'minimum',
        "unknown_percent": effective_unknown,
        "requested_unknown_percent": unknown_mode,
        "roll_optimize": roll_optimize,
        "roll_optimize_baseline": effective_unknown if roll_optimize else None,
        "roll_plan_search": next((b.roll_plan_search for b in builds if b.roll_plan_search), None),
        "allow_duplicates": allow_duplicates,
        "carryable_only": carryable_only,
        "builds_found": len(builds),
        "search_status": getattr(builds, 'search_status', None),
        "builds": [b.to_dict() for b in builds],
        "log_file": _log_path.replace("\\", "/"),
        "auction_fresh": freshness["is_fresh"],
        "auction_last_collected": freshness["last_collected"],
        "auction_age_hours": freshness["age_hours"],
        "auction_refreshing": auction_refreshing,
        "last_emission": freshness.get("last_emission"),
        **({'stock_only': True} if stock_only else {}),
    }


@app.get("/api/all-stats")
async def api_all_stats():
    """Return all unique stat prop_ids and their Russian names across all artefacts."""
    conn = get_connection()
    seen = {}
    for row in conn.execute("SELECT stats, bonus_stats FROM artefacts"):
        stats = json.loads(row["stats"]) if row["stats"] else []
        for s in stats:
            pid = s.get("prop_id", "")
            if pid and pid not in seen:
                seen[pid] = s.get("name_ru", pid)
        bonus = json.loads(row["bonus_stats"]) if row["bonus_stats"] else []
        for s in bonus:
            pid = s.get("prop_id", "")
            if pid and pid not in seen:
                seen[pid] = s.get("name_ru", pid)
    conn.close()
    from ._survivability import REGISTRY
    from ._derived_stats import NAMES as DERIVED_NAMES
    seen.update({k:v[0] for k,v in REGISTRY.items()})
    seen.update(DERIVED_NAMES)
    return {"stats": seen}

REFRESH_LEASE_SECONDS = DEFAULT_LEASE_SECONDS

def _run_auction_refresh(lease):
    """Background auction data refresh - concurrent (8 workers, limit=1000)."""
    try:
        from .collector import collect_auction_data
        collect_auction_data(max_lots=1000, item_source='artefacts', lease=lease)
        log.info("Auction refresh done")
    except Exception as e:
        lease.finish(str(e))
        log.error("Auction refresh error: %s", e)


def _start_auction_refresh():
    lease = application_store(Config.DATA_DIR).claim(auction_job(Config.REGION), REFRESH_LEASE_SECONDS)
    if lease is None:
        return False
    try:
        threading.Thread(target=_run_auction_refresh, args=(lease,), daemon=True).start()
    except Exception as error:
        lease.finish(str(error))
        raise
    return True


@app.post("/api/refresh-auction")
async def refresh_auction():
    """Start background auction data refresh. Returns immediately."""
    started = _start_auction_refresh()
    state = application_store(Config.DATA_DIR).status(auction_job(Config.REGION))
    if not started:
        return JSONResponse({"status": "already_running", "done": state['done'], "total": state['total']})
    _conn = get_connection()
    _cnt = _conn.execute("SELECT COUNT(*) FROM artefacts").fetchone()[0]
    _conn.close()
    return {"status": "started", "total_items": _cnt}


@app.get("/api/refresh-status")
async def refresh_status():
    """Poll auction refresh progress."""
    return application_store(Config.DATA_DIR).status(auction_job(Config.REGION))

@app.post("/api/parse-prompt")
async def parse_prompt(body: dict = None):
    """Parse natural language build description using GLM-5.2 LLM with deterministic fallback."""
    import re, json as _json, os
    import aiohttp

    if body is None:
        return JSONResponse(status_code=400, content={"error": "empty body"})
    text = body.get("text", "").strip()
    if not text:
        return JSONResponse(status_code=400, content={"error": "empty text"})

    # ── Deterministic keyword parser (fallback) ──
    def _deterministic_parse(txt: str) -> dict:
        t = txt.lower()
        result = {"priorities": {}, "allow_duplicates": False, "conservative": False,
                  "unknown_percent": "midpoint",
                  "carryable_only": True, "guaranteed_carryable": False,
                  "container_id": None, "budget": 0}

        stat_keywords = [
            ("total_speed", ["общая скорость", "общую скорость", "общей скорости", "total speed"]),
            ("bullet_dmg_factor", ["пулестойкость", "пулестой", "пулестойк", "пуля", "bullet"]),
            ("health_bonus", ["живучесть", "здоровье", "health", "хп"]),
            ("max_weight_bonus", ["переносимый вес", "переносимый", "вес", "weight"]),
            ("speed_modifier", ["скорость", "speed"]),
            ("stamina_bonus", ["выносливость", "стамина", "stamina"]),
            ("regeneration_bonus", ["регенерац", "regen", "реген"]),
            ("stopping_protection", ["стойкость", "stopping", "стоп"]),
        ]
        for stat_id, keywords in stat_keywords:
            if stat_id == "speed_modifier" and "total_speed" in result["priorities"]:
                continue
            for kw in keywords:
                if kw in t:
                    result["priorities"][stat_id] = 10
                    break

        m = re.search(r'(\d+)\s*(млн|миллион|кк|kk)', t)
        if m:
            result["budget"] = int(m.group(1)) * 1000000
        else:
            m = re.search(r'бюджет[^0-9]*(\d+)', t)
            if m:
                val = int(m.group(1))
                result["budget"] = val if val > 1000 else val * 1000000

        if "бочонок" in t or "rgoz" in t:
            result["container_id"] = "rgoz"
        elif "тюк" in t or "v667" in t:
            result["container_id"] = "v667"
        elif "разгрузк" in t or "adr" in t or "dkln" in t:
            result["container_id"] = "dkln"
        elif "берлог" in t:
            result["container_id"] = "g35n"
        elif "кокон" in t:
            result["container_id"] = "9y4q"
        elif "улей" in t or "p92d" in t:
            result["container_id"] = "p92d"
        elif "zivcas" in t or "арктик" in t or "512o" in t:
            result["container_id"] = "512o"
        elif "кзс" in t or "kzs" in t:
            result["container_id"] = "j33l"
        elif "вьюк" in t:
            result["container_id"] = "51zo"
        elif "сума" in t or "трансформер" in t or "jmp6" in t:
            result["container_id"] = "jmp6"
        elif "спортивн" in t or "odq0" in t:
            result["container_id"] = "odq0"
        elif "сот" in t and ("контейнер" in t or "o3d5" in t):
            result["container_id"] = "o3d5"

        if "дубликат" in t and ("можно" in t or "разреш" in t or "да" in t):
            result["allow_duplicates"] = True
        if any(w in t for w in ["гарант", "гарантия", "худший", "без перекрут", "без перекрута", "conservative", "консерват"]):
            result["conservative"] = True
            result["unknown_percent"] = "minimum"
        elif any(w in t for w in ["ожидаем", "в среднем", "середин", "expected"]):
            result["conservative"] = False
            result["unknown_percent"] = "midpoint"
        elif any(w in t for w in ["максимум", "макс. процент", "лучший ролл", "maximum"]):
            result["conservative"] = False
            result["unknown_percent"] = "maximum"
        if any(w in t for w in ["переносимый", "носибельност", "носибел"]):
            result["carryable_only"] = True
        if any(w in t for w in ["гарант носиб", "гарантир носиб", "гарантированная носиб", "гарантированную носиб"]):
            result["guaranteed_carryable"] = True
            result["conservative"] = True
            result["unknown_percent"] = "minimum"

        # Combat: "лоб в лоб", "танк", "приведенка", "ehp" → bullet + health
        if any(w in t for w in ["лоб в лоб", "лоб-в-лоб", "танк", "приведенк", "приведенное здоровье", "приведён", "ehp", "эхп", "пвп", "pvp"]):
            if "bullet_dmg_factor" not in result["priorities"]:
                result["priorities"]["bullet_dmg_factor"] = 10
            if "health_bonus" not in result["priorities"]:
                result["priorities"]["health_bonus"] = 8
        # "бегать быстро" → speed emphasis
        if "бегать" in t and ("скор" in t or "быстр" in t) and "total_speed" not in result["priorities"]:
            result["priorities"]["speed_modifier"] = 10
            result["priorities"]["sprint_speed_modifier"] = 7
        # "не ловить радиацию" → max threshold
        if "не ловить рад" in t or "без радиац" in t:
            result.setdefault("max_thresholds", {})["radiation_accumulation"] = 0.0

        min_thresholds = {}
        m = re.search(r'от\s+(\d+(?:\.\d+)?)\s*(?:до\s+\d+(?:\.\d+)?)?\s*кг', t)
        if m and ("вес" in t or "перенос" in t):
            min_thresholds["max_weight_bonus"] = float(m.group(1))
        m2 = re.search(r'от\s+(\d+(?:\.\d+)?)\s*(?:до\s+\d+(?:\.\d+)?)?\s*пуле', t)
        if m2:
            min_thresholds["bullet_dmg_factor"] = float(m2.group(1))
        m3 = re.search(r'от\s+(\d+(?:\.\d+)?)\s*(?:до\s+\d+(?:\.\d+)?)?\s*жив', t)
        if m3:
            min_thresholds["health_bonus"] = float(m3.group(1))
        if min_thresholds:
            result["min_thresholds"] = min_thresholds

        max_thresholds = {}
        m = re.search(r'до\s+(\d+(?:\.\d+)?)\s*кг', t)
        if m and ("вес" in t or "перенос" in t):
            max_thresholds["max_weight_bonus"] = float(m.group(1))
        if max_thresholds:
            result["max_thresholds"] = max_thresholds

        # Lower priority to 2 for stats that have min/max thresholds (range constraint,
        # not a primary objective). E.g., "вес от 50 до 60" → weight=2, not 10.
        _range_stats = set(min_thresholds.keys()) | set(max_thresholds.keys())
        for pid in _range_stats:
            if pid in result["priorities"] and result["priorities"][pid] >= 10:
                result["priorities"][pid] = 2

        # Exclude artifacts: "без яйца" / "исключить X" → resolve names to item_ids
        _exclude_names = []
        _exc_match = re.findall(r'(?:без|исключ[иы]ть?)\s+([а-яё]+)', t)
        if _exc_match:
            _name_map = {
                "яйца": "y5ro", "яйцо": "y5ro", "яйцем": "y5ro",
                "бубльгум": "qodk", "морозец": "ljn2", "смольник": "49zn",
                "корка": "8m30", "жильник": "l1m2", "креветка": "8k30",
            }
            for name in _exc_match:
                _id = _name_map.get(name.lower())
                if _id and _id not in _exclude_names:
                    _exclude_names.append(_id)
        if _exclude_names:
            result["exclude"] = _exclude_names
        result["llm_parsed"] = False
        return result

    # ── LLM system prompt (cached, built from DB) ──
    global _llm_system_prompt
    if _llm_system_prompt is None:
        stats_map = {}
        conn = get_connection()
        for r in conn.execute("SELECT stats, bonus_stats FROM artefacts WHERE stats IS NOT NULL").fetchall():
            for s in (_json.loads(r["stats"]) if r["stats"] else []):
                pid = s.get("prop_id", "")
                name = s.get("name_ru", pid)
                if pid and pid not in stats_map:
                    stats_map[pid] = name
            if r["bonus_stats"]:
                for s in _json.loads(r["bonus_stats"]):
                    pid = s.get("prop_id", "")
                    name = s.get("name_ru", pid)
                    if pid and pid not in stats_map:
                        stats_map[pid] = name
        cont_map = {}
        for r in conn.execute("SELECT id, name_ru, slots FROM containers").fetchall():
            cont_map[r["id"]] = f'{r["name_ru"]} ({r["slots"]} сл.)'
        conn.close()

        stat_list = '\n'.join(f'  "{pid}": "{name}"' for pid, name in sorted(stats_map.items()))
        cont_list = '\n'.join(f'  "{cid}": "{name}"' for cid, name in sorted(cont_map.items()))
        _llm_system_prompt = f"""Ты — парсер сборок артефактов для игры STALCRAFT.
Разбери описание сборки на естественном русском языке и верни JSON.

Доступные статы (prop_id → название):
{stat_list}

Доступные контейнеры:
{cont_list}

Верни JSON:
{{"priorities": {{"prop_id": вес_0_10}}, "budget": число_рублей, "container_id": "id_или_null",
  "allow_duplicates": bool, "conservative": bool, "unknown_percent": "minimum"|"midpoint"|"maximum",
  "carryable_only": bool,
  "min_thresholds": {{"prop_id": мин}}, "max_thresholds": {{"prop_id": макс}},
  "exclude": ["item_id"]}}
 exclude: пустой массив [] если нет исключений, иначе массив item_id.

Правила: пулестой→bullet_dmg_factor, живучесть→health_bonus, вес→max_weight_bonus,
скорость→speed_modifier, общая скорость→total_speed (speed_modifier + sprint_speed_modifier),
выносливость→stamina_bonus, стойкость→stopping_protection.
50млн/50кк→50000000. бочонок→rgoz.
"Гарант. мин." / "гарантированный минимум" / "худший" / "без перекрута" → unknown_percent:"minimum", conservative:true.
"Ожидаемо" / "в среднем" / "середина" → unknown_percent:"midpoint", conservative:false.
"Максимум" / "лучший ролл" / "макс. процент" → unknown_percent:"maximum", conservative:false.
Если режим не указан — unknown_percent:"midpoint".
unknown_percent означает: известный процент лота сохраняется, а выбранная точка диапазона качества применяется только к неизвестным процентам.
от X до Y кг→min+max thresholds (НЕ conservative — это ограничение диапазона, separate feature).
ВАЖНО: если указан диапазон (от/до) для стата — поставь ему приоритет 2-3, не 0 и не 10.
Приоритет 0 = стат не попадёт в пул кандидатов, оптимизатор не найдёт артефакты.
Приоритет 10 = оптимизатор пожертвует главными статами ради этого.
Пример: "пулестой живучесть вес от 50 до 60" → bullet_dmg_factor=10, health_bonus=10, max_weight_bonus=2
carryable_only: true если "носибельност" или "переносим". Если "дубликаты можно" — носибельность может быть false (тяжёлые сборки).
guaranteed_carryable: true если "гарантированная носибельность" или "гарант носиб" — проверяет носибельность при МАКСИМАЛЬНОМ % качества, не только минимальном. Вместе с conservative=true.
"Лоб в лоб", "танк", "приведенка", "ehp", "пвп" → bullet_dmg_factor=10, health_bonus=8 (боевая сборка).
"Бегать быстро" → speed_modifier=10, sprint_speed_modifier=7.
"Разгрузка" → dkln, "Берлога-6" → g35n, "Улей" → p92d, "Кокон" → 9y4q, "Бочонок" → rgoz.
Исключение артефактов: если пользователь пишет "без яйца" или "исключить X" — добавь item_id в "exclude" (массив строк). Яйцо=y5ro, Бубльгум=qodk, Морозец=ljn2, Смольник=49zn.
Верни "exclude": ["item_id1", "item_id2"] если есть исключения, иначе не включай поле."""

    # ── Try GLM-5.2 via DashScope API ──
    api_key = os.getenv("DASHSCOPE_API_KEY", "")
    if not api_key:
        # Try loading from omp config
        omp_env = os.path.expanduser("~/.omp/agent/.env")
        if os.path.exists(omp_env):
            from dotenv import load_dotenv as _ld
            _ld(omp_env)
            api_key = os.getenv("DASHSCOPE_API_KEY", "")

    if api_key:
        try:
            timeout = aiohttp.ClientTimeout(total=120.0)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "glm-5.2",
                        "messages": [
                            {"role": "system", "content": _llm_system_prompt},
                            {"role": "user", "content": text},
                        ],
                        "temperature": 0.1,
                        "max_tokens": 2048,
                        "response_format": {"type": "json_object"},
                    },
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        content = data["choices"][0]["message"]["content"]
                        parsed = _json.loads(content)
                        parsed.setdefault("priorities", {})
                        parsed.setdefault("budget", 0)
                        parsed.setdefault("container_id", None)
                        parsed.setdefault("allow_duplicates", False)
                        parsed.setdefault("conservative", False)
                        parsed.setdefault("unknown_percent",
                                          "minimum" if parsed.get("conservative") else "midpoint")
                        if parsed.get("unknown_percent") not in ("minimum", "midpoint", "maximum"):
                            parsed["unknown_percent"] = ("minimum" if parsed.get("conservative")
                                                         else "midpoint")
                        parsed["conservative"] = parsed["unknown_percent"] == "minimum"
                        parsed.setdefault("carryable_only", True)
                        parsed.setdefault("guaranteed_carryable", False)
                        parsed.setdefault("exclude", [])
                        parsed["llm_parsed"] = True
                        return parsed
                    else:
                        err_text = await resp.text()
                        log.warning("LLM API error %d: %s", resp.status, err_text[:200])
        except Exception as e:
            log.warning("LLM parse failed, falling back to deterministic: %s", e)

    # ── Fallback: deterministic parser ──
    result = _deterministic_parse(text)
    result["llm_parsed"] = False
    return result


# ──────────────────────── Lunar-zone Bridge ────────────────────────

@app.get("/api/export-lunarzone")
def api_export_lunarzone(
    budget: int = Query(0, ge=0),
    tolerance: float = Query(8.0, ge=0, le=50),
    container_id: str | None = Query(None),
    top_n: int = Query(1, ge=1, le=10),
    priorities: str | None = Query(None),
    min_thresholds: str | None = Query(None),
    max_thresholds: str | None = Query(None),
    owned: str | None = Query(None),
    carryable_only: bool = Query(True),
    min_budget: bool = Query(False),
    conservative: bool = Query(False),
    unknown_percent: str | None = Query(None, pattern="^(minimum|midpoint|maximum)$"),
    exclude: str | None = Query(None),
    armor_id: str | None = Query(None),
    armor_ids: str | None = Query(None),
    armor_level: int = Query(15, ge=0, le=15),
    build_index: int = Query(0, ge=0, description="Which build to export (0=top)"),
    optimization_mode: str = Query("balanced", pattern="^(balanced|lexicographic|min_budget)$"),
    priority_order: str | None = Query(None),
    objective_scales: str | None = Query(None),
    objective_tolerance: float = Query(1e-6, gt=0),
    max_quality: bool = Query(False),
    quality_timeout: float = Query(45., gt=0, le=300),
    deep_optimization: bool = Query(False),
    allow_duplicates: bool = Query(False),
    stock_only: bool = Query(False),
    primary_only: bool = Query(False),
):
    """Export optimizer build to lunar-zone JSON format."""
    from .lunar_bridge import export_to_lunarzone
    try:
        if deep_optimization and not max_quality:
            raise ValueError('Deep optimization requires max_quality=true')
        from ._quality_time import request_seconds
        quality_timeout = request_seconds(quality_timeout, deep_optimization)
        prio = json.loads(priorities) if priorities else {}
        mins = json.loads(min_thresholds) if min_thresholds else {}
        maxs = json.loads(max_thresholds) if max_thresholds else {}
        owned_list = json.loads(owned) if owned else []
        exclude_list = json.loads(exclude) if exclude else []
        order = json.loads(priority_order) if priority_order else None
        scales = json.loads(objective_scales) if objective_scales else None
        armor_list = json.loads(armor_ids) if armor_ids else None
        from ._quality_scenario import resolve_mode
        unknown_mode = resolve_mode(conservative, unknown_percent)
        if armor_list is not None and (not isinstance(armor_list, list) or not armor_list or
                                      any(a is not None and not isinstance(a, str) for a in armor_list)):
            raise ValueError("armor_ids must be a nonempty JSON list of IDs or null")
        if armor_id is not None and armor_list is not None:
            raise ValueError("Choose armor_id or armor_ids")
        from ._objective import Objective
        mode = "min_budget" if min_budget else optimization_mode
        Objective.create(prio, scales, mode, order, objective_tolerance)
        from ._survivability import validate as validate_survivability
        validate_survivability(prio, mins, maxs, max_quality=max_quality, mode=mode)
        from ._ehp_primary_only import validate_request
        validate_request(primary_only, deep_optimization, max_quality, prio, mode, min_budget)
        if primary_only and tolerance != 0:
            raise ValueError('primary_only requires tolerance=0: strict purchase budget')
    except (ValueError, TypeError) as error:
        return JSONResponse(status_code=422, content={"error": str(error)})

    try:
        engine = build_artifact_sets
        quality_options = {}
        if primary_only:
            quality_options['primary_only'] = True
        if stock_only:
            quality_options['stock_only'] = True
        if max_quality:
            if not MAX_QUALITY_RELEASED:
                return JSONResponse(status_code=422, content={"error": "Experimental quality gate has not passed"})
            from ._max_quality import build_max_quality_sets
            engine = build_max_quality_sets
            quality_options.update(quality_timeout=quality_timeout, allow_duplicates=allow_duplicates)
            if deep_optimization:
                quality_options['deep_optimization'] = True
            from .api_client import QUALITY_RANGES
            if not isinstance(owned_list, list) or any(not isinstance(a, dict) for a in owned_list):
                raise ValueError('Owned artifacts must be a JSON list of records')
            for artifact in owned_list:
                if not artifact.get('art_percent'):
                    tier = QUALITY_RANGES.get(artifact.get('quality', 0), QUALITY_RANGES[0])
                    artifact['art_percent'] = tier['min'] + tier['range']
        builds = engine(
            budget=budget, tolerance_pct=tolerance, priorities=prio,
            min_thresholds=mins, max_thresholds=maxs, top_n=top_n,
            region=Config.REGION, container_id=container_id,
            owned_artifacts=owned_list, carryable_only=carryable_only,
            min_budget=min_budget, conservative=(unknown_mode == 'minimum'),
            unknown_percent=unknown_mode,
            exclude_item_ids=exclude_list,
            optimization_mode=mode, priority_order=order,
            objective_scales=scales, objective_tolerance=objective_tolerance,
            armor_id=armor_id, armor_ids=armor_list, armor_level=armor_level,
            **quality_options,
        )
    except (ValueError, TypeError) as error:
        return JSONResponse(status_code=422, content={"error": str(error)})


    if not builds or build_index >= len(builds):
        return JSONResponse(status_code=404, content={"error": "No build found at index"})

    lz_json = export_to_lunarzone(builds[build_index], armor_id)
    if max_quality:
        # The quality path exports actual unlocked bonuses, not the catalog's
        # default unlock order. Legacy export behavior is kept unchanged.
        from .builder import _normalize_bonus_id
        chosen = builds[build_index]
        for option, exported in zip(chosen.artifacts, lz_json['build_data']['artifacts']):
            unlocked = {_normalize_bonus_id(key.upper()) for key in option.unlocked_bonus_ids}
            exported['selectedSecondary'] = [bonus['name_ru'] for bonus in option.bonus_stats
                if bonus.get('name_ru') and _normalize_bonus_id(bonus.get('prop_id', '').upper()) in unlocked]
            exported['quality'] = option.art_percent
        lz_json['_optimizer_meta']['search_status'] = getattr(builds, 'search_status', None)
        if 'effective_health' in chosen.combined_stats:
            data = chosen.to_dict()
            lz_json['_optimizer_meta'].update(derived_stats=data['derived_stats'],
                calculation_provenance=data['calculation_provenance'], objective=chosen.objective_config,
                constraints=dict(minimum=mins, maximum=maxs),
                scenario=dict(conservative=(unknown_mode == 'minimum'), unknown_percent=unknown_mode,
                              stock_only=stock_only))
        lz_json['_optimizer_meta']['physical_items'] = [dict(auction_lot_id=o.auction_lot_id,
            physical_id=o.physical_id, price=o.buyout_price, bonus_ids=o.unlocked_bonus_ids) for o in chosen.artifacts]
    return {
        "lunarzone_json": lz_json,
        "lunar_url": f"https://lunar-zone.com/calculator",
        "build_count": len(builds),
        "exported_index": build_index,
    }


@app.get("/api/import-lunarzone")
def api_import_lunarzone(build_id: str = Query(..., description="Lunar-zone build UUID")):
    """Import a lunar-zone build by UUID. Returns parsed artifacts + stats."""
    from .lunar_bridge import import_lunarzone_build
    parsed = import_lunarzone_build(build_id)
    if not parsed:
        return JSONResponse(status_code=404, content={"error": "Build not found on lunar-zone"})
    return parsed


@app.get("/api/verify-lunarzone")
def api_verify_lunarzone(build_id: str = Query(..., description="Lunar-zone build UUID")):
    """Import a lunar-zone build AND compute stats using our formulas.

    Fetches the build, converts artifacts to specs, computes stats directly
    (no search — uses exact artifacts from the build), applies armor.
    """
    from .lunar_bridge import import_lunarzone_build, compute_specific_build
    parsed = import_lunarzone_build(build_id)
    if not parsed:
        return JSONResponse(status_code=404, content={"error": "Build not found on lunar-zone"})

    # Convert parsed artifacts to spec format
    artifacts_spec = []
    for art in parsed.get("artifacts", []):
        artifacts_spec.append({
            "item_id": art["item_id"],
            "quality": art["quality"],
            "ptn": art["ptn"],
            "art_percent": art["art_percent"],
            "bonus_props": art.get("bonus_props", []),
        })

    # Compute stats directly (no search — uses ALL artifacts from the build)
    container_id = parsed.get("container_id")
    armor_id = parsed.get("armor_id")
    build = compute_specific_build(artifacts_spec, container_id, armor_id, parsed.get("armor_level", 15))

    computed = build.to_dict() if build else None
    return {
        **parsed,
        "computed_stats": computed,
        "stats_match": computed is not None,
    }

@app.get("/api/template-search")
def api_template_search(
    build_id: str = Query(..., description="Lunar-zone build UUID"),
    exact_ptn: bool = Query(True, description="Require exact ptn match"),
):
    """Template mode: search auction for lots matching a lunar-zone build template.

    For each artifact in the build, finds auction lots with the same item_id,
    quality tier, bonus properties, and ptn level. Returns matching lots with
    prices and computed stats (matching lunar-zone at 1000% precision).
    """
    from .lunar_bridge import search_template_lots
    result = search_template_lots(build_id, exact_ptn=exact_ptn)
    if not result:
        return JSONResponse(status_code=404, content={"error": "Build not found on lunar-zone"})
    return result

@app.get("/api/template-build")
def api_template_build(
    build_id: str = Query(..., description="Lunar-zone build UUID — template to match"),
    budget: int = Query(0, ge=0, description="Max price in rubles"),
    container_id: str | None = Query(None, description="Override container ID (default: template's container)"),
    armor_id: str | None = Query(None, description="Override armor ID (default: template's armor)"),
    tolerance: float = Query(8.0, ge=0, le=50, description="Budget tolerance %"),
    relax_quality: bool = Query(False, description="Allow any quality — find cheapest lots of same item_ids"),
):
    """Template mode: greedily build from a lunar-zone template within budget.

    relax_quality=True: find cheapest lots of same item_ids regardless of quality/bonuses.
    Preserves the template's exact composition — NOT the general optimizer.
    """
    from .lunar_bridge import build_from_template
    result = build_from_template(build_id, budget, container_id, armor_id, tolerance, relax_quality=relax_quality)
    if not result:
        return JSONResponse(status_code=404, content={"error": "Build not found on lunar-zone"})
    return result

@app.get("/api/public-builds")
def api_public_builds(
    offset: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
):
    """List public builds from lunar-zone."""
    from .lunar_bridge import list_public_builds
    return list_public_builds(offset=offset, limit=limit)


@app.get("/api/price-history")
def api_price_history(
    item_id: str = Query(..., description="Artifact item_id"),
    quality: int | None = Query(None, description="Filter by quality tier 0-5"),
    limit: int = Query(100, ge=1, le=500),
):
    """Return auction price history for an artifact + current lowest buyout."""
    conn = get_connection()
    if quality is not None:
        rows = conn.execute(
            "SELECT price, quality, art_percent, sold_time, ptn "
            "FROM auction_history WHERE item_id=? AND region=? AND quality=? "
            "ORDER BY sold_time DESC LIMIT ?",
            (item_id, Config.REGION, quality, limit)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT price, quality, art_percent, sold_time, ptn "
            "FROM auction_history WHERE item_id=? AND region=? "
 "ORDER BY sold_time DESC LIMIT ?",
            (item_id, Config.REGION, limit)
        ).fetchall()
    current = conn.execute(
        "SELECT MIN(buyout_price) as min_price FROM auction_lots "
        "WHERE item_id=? AND region=?",
        (item_id, Config.REGION)
    ).fetchone()
    conn.close()
    # Reverse to chronological order (oldest first)
    history = [{"price": r["price"], "quality": r["quality"],
                "art_percent": round(r["art_percent"], 1), "sold_time": r["sold_time"],
                "ptn": r["ptn"]} for r in reversed(rows)]
    prices = [h["price"] for h in history]
    return {
        "item_id": item_id,
        "history": history,
        "current_min_price": current["min_price"] if current and current["min_price"] else None,
        "count": len(history),
        "min_price": min(prices) if prices else 0,
        "max_price": max(prices) if prices else 0,
        "avg_price": round(sum(prices) / len(prices)) if prices else 0,
    }

# ──────────────────────── Meta-builds optimizer ────────────────────────

@app.get("/api/meta-builds")
def api_meta_builds(
    budget: int = Query(50_000_000, ge=1_000, description="Budget in rubles"),
    style: str = Query("balanced", description="Playstyle: pvp/tank/speed/heal/balanced/carry"),
    priorities: str | None = Query(None, description='JSON overrides — {"bullet_dmg_factor": 10}'),
    top_n: int = Query(10, ge=1, le=30, description="Max builds to return"),
    carryable_only: bool = Query(True, description="Only carryable builds"),
    allow_duplicates: bool = Query(False, description="Allow same item_id multiple times"),
    containers: str | None = Query(None, description='JSON: ["l362","dkln"] — explicit container IDs'),
    armor_id: str | None = Query(None),
    armor_ids: str | None = Query(None),
    armor_level: int = Query(15, ge=0, le=15),
    use_cache: bool = Query(True, description="Try pre-computed cache first"),
    stock_only: bool = Query(False),
    primary_only: bool = Query(False),
):
    """Find the BEST builds for any budget — auto-selects containers.

    Tries top-5 containers ranked by playstyle-aligned bonus scoring,
    runs beam search for each, returns Pareto-optimal builds.

    Playstyles: pvp, tank, speed, heal, balanced, carry.
    """
    if primary_only:
        return JSONResponse(status_code=422, content={'error':'primary_only requires Deep Max Quality EHP'})
    from .meta_optimizer import auto_optimize, get_cached_meta

    if use_cache and not stock_only and armor_id is None and armor_ids is None and priorities is None and containers is None and carryable_only and not allow_duplicates:
        cached = get_cached_meta(budget, style)
        if cached:
            cached = [b for b in cached if b.get("total_price", float("inf")) <= int(budget * 1.08)]
        if cached:
            return {
                "budget": budget,
                "style": style,
                "cached": True,
                "builds": cached[:top_n],
            }

    try:
        prio = json.loads(priorities) if priorities is not None else None
        if priorities is not None:
            from ._objective import Objective
            Objective.create(prio)
            from ._survivability import validate as validate_survivability
            validate_survivability(prio, None, None, max_quality=False, mode='balanced')
        container_list = json.loads(containers) if containers is not None else None
        if containers is not None and (not isinstance(container_list, list) or not container_list or
                                      any(not isinstance(cid, str) or not cid for cid in container_list)):
            raise ValueError("containers must be a nonempty JSON list of IDs")
        armor_list = json.loads(armor_ids) if armor_ids else None
        result = auto_optimize(
            budget=budget,
            armor_id=armor_id, armor_ids=armor_list, armor_level=armor_level,
            priorities=prio,
            style=style if not prio else None,
            top_n=top_n,
            carryable_only=carryable_only,
            allow_duplicates=allow_duplicates,
            containers_override=container_list,
            **({'stock_only': True} if stock_only else {}),
        )
    except (ValueError, TypeError) as error:
        return JSONResponse(status_code=422, content={"error": str(error)})


    return {
        "budget": result["budget"],
        "style": result["style"],
        "priorities": result["priorities"],
        "containers_tried": result["containers_tried"],
        "builds_found": len(result["builds"]),
        "builds": result["builds"],
        "elapsed_sec": result["elapsed_sec"],
        "cached": False,
    }


@app.get("/api/budget-sweep")
def api_budget_sweep(
    min_budget: int = Query(1_000_000, ge=1_000, description="Min budget"),
    max_budget: int = Query(100_000_000, ge=1_000, description="Max budget"),
    steps: int = Query(8, ge=2, le=20, description="Number of budget points"),
    style: str = Query("balanced", description="Playstyle"),
    top_n: int = Query(3, ge=1, le=10, description="Builds per step"),
    carryable_only: bool = Query(True),
    allow_duplicates: bool = Query(False),
    armor_id: str | None = Query(None),
    armor_level: int = Query(15, ge=0, le=15),
    stock_only: bool = Query(False),
    primary_only: bool = Query(False),
):
    """Run optimization at multiple budget points — shows meta progression.

    Returns best build at each budget tier (logarithmic spacing).
    """
    if primary_only:
        return JSONResponse(status_code=422, content={'error':'primary_only requires Deep Max Quality EHP'})
    from .meta_optimizer import budget_sweep

    try:
        result = budget_sweep(
            min_budget=min_budget,
            max_budget=max_budget,
            steps=steps,
            style=style,
            top_n=top_n,
            carryable_only=carryable_only,
            allow_duplicates=allow_duplicates,
            armor_id=armor_id, armor_level=armor_level,
            **({'stock_only': True} if stock_only else {}),
        )
    except (ValueError, TypeError) as error:
        return JSONResponse(status_code=422, content={"error": str(error)})

    return result


@app.get("/api/playstyles")
def api_playstyles():
    """List available playstyle presets."""
    from .meta_optimizer import PLAYSTYLES
    return {"playstyles": PLAYSTYLES}
