"""Explicit experimental objective maximizer. Existing search modes never call it."""
import math
import sqlite3
import time
from contextlib import nullcontext
from . import _max_quality_model as limits
from ._max_quality_process import quality_slot, run_model
from ._quality_time import DEEP_SECONDS, DEEP_NODE_LIMIT, VALIDATION_RESERVE, request_seconds


class QualityResults(list):
    def __init__(self, builds=(), *, state='UNKNOWN', reason='unknown', models=(), timings=None,
                 conservative=False, unknown_percent=None):
        super().__init__(builds)
        text = {'OPTIMAL': 'Численный оптимум полного ранга доказан в проверенных подготовленных пулах.',
                'FEASIBLE': 'Допустимая сборка найдена; оптимальность полного ранга не доказана.',
                'INFEASIBLE': 'Несовместность доказана только в проверенных подготовленных пулах.',
                'UNKNOWN': 'Поиск не смог подтвердить наличие или отсутствие допустимой сборки.'}
        explanations = {
            'unsupported_guaranteed_or_conservative': 'MILP пока не поддерживает отдельную гарантированную носимость. Расчёт при минимальном проценте не включает эту функцию.',
            'unsupported_or_derived_stat': 'Запрошено неизвестное или нелинейное свойство; ограничение не было отброшено.',
            'nonlinear_protected_accumulation_objective': 'Оптимизация защищённых накоплений пока не поддерживается.',
            'quality_worker_busy': 'Другой запрос уже выполняет MILP. Повторите после его завершения.',
            'request_time_limit': 'Общий лимит времени исчерпан.',
            'time_or_node_limit': 'Solver остановлен по лимиту времени или узлов.',
            'solution_limit': 'HiGHS остановлен по внутреннему лимиту поиска. Полный оптимум не доказан.',
            'node_limit': 'Достигнут лимит узлов MILP. Полный оптимум не доказан.',
            'time_limit': 'Достигнут лимит времени solver. Полный оптимум не доказан.',
            'solver_error': 'Solver сообщил ошибку исполнения или протокола; его статус сохранён в диагностике.',
            'secondary_infeasible': 'Вторичная модель несовместна; это не доказательство несовместности исходного запроса.',
            'dependency_unavailable': 'SciPy/HiGHS недоступен в текущем окружении.',
            'cancelled': 'Вычисление отменено.',
            'memory_limit': 'Достигнут лимит памяти отдельного процесса.',
            'profile_limit': 'Слишком много контейнерных и броневых профилей. Выберите конкретный контейнер или меньше альтернатив.',
            'complete_full_rank': 'Все цели и сравнение по цене подтверждены в указанном scope.',
            'survivability_numeric_limit': 'Производные характеристики выходят за поддерживаемый численный диапазон.',
            'survivability_matrix_limit': 'Полная модель производных характеристик не помещается в допустимый размер. Эвристическое усечение не применялось.',
            'unsupported_survivability_lexicographic': 'Для производных характеристик поддержан только Balanced в Max Quality / Deep.',
            'unproved_primary': 'Численная граница не подтвердила максимум основной цели.',
            'restricted_subproblem_infeasible': 'Модель с границей от стартового свидетеля несовместна; это не доказательство несовместности исходной задачи. Свидетель проверяется отдельно.',
        }
        self.search_status = dict(state=state, reason=reason, backend='max_quality_milp', experimental=True,
            scope='verified_fully_described_prepared_profiles', global_optimality_proven=False,
            numerical_certificate=True,
            optimality_proven=state=='OPTIMAL', infeasibility_proven=state=='INFEASIBLE',
            models=list(models), timings=timings or {}, message=state + ': ' + text[state] + ' ' + explanations.get(reason, 'Код завершения: ' + reason),
            limits=dict(seconds=limits.DEFAULT_SECONDS, model_seconds=limits.MODEL_SECONDS,
                candidates=limits.MAX_CANDIDATES, profiles=limits.MAX_PROFILES, rows=limits.MAX_ROWS,
                nonzeros=limits.MAX_NONZEROS, nodes=limits.MAX_NODES, worker_memory_bytes=limits.MEMORY_BYTES,
                simultaneous_heavy_tasks=1))
        from ._quality_scenario import metadata, resolve_mode, scenario_active
        mode = resolve_mode(conservative, unknown_percent)
        self.unknown_percent = mode
        if scenario_active(mode) or unknown_percent is not None:
            calculation = metadata(mode)
            self.search_status['calculation'] = calculation
            self.search_status['message'] += ' ' + calculation['message']


def _build_once(budget, *, quality_timeout=limits.DEFAULT_SECONDS, deep_optimization=False, primary_only=False, cancel_event=None, _deadline=None, _market=None, _primary_node_budget=50000, **request):
    """Maximize balanced or binned lexicographic rank, then price; one result.

    No implicit Beam fallback. Unsupported scenarios return UNKNOWN, explicitly.
    Proof scope is the verified prepared profiles, never the entire auction.
    """
    from . import builder as b
    from ._feasibility_fallback import current_candidates
    from ._physical_items import physical_key
    from ._quality_scenario import resolve_mode
    mode = resolve_mode(bool(request.get('conservative')), request.get('unknown_percent'))
    request['unknown_percent'] = mode
    scenario = mode != 'maximum'
    start = time.monotonic()
    context = {}
    def response(*args, **kwargs):
        result = QualityResults(*args, unknown_percent=mode, **kwargs)
        from . import _survivability as derived
        if derived.requested(request.get('priorities'), request.get('min_thresholds'), request.get('max_thresholds')):
            result.search_status['derived_metrics'] = derived.provenance()
            result.search_status['limits'].update(base_rows=limits.MAX_ROWS, base_nonzeros=limits.MAX_NONZEROS,
                variables=derived.MAX_VARIABLES, rows=derived.MAX_ROWS, nonzeros=derived.MAX_NONZEROS)
        if context.get('candidate_audit'):
            result.search_status['candidate_preparation'] = {
                k: v for k, v in context['candidate_audit'].items()
                if k not in ('deadline', 'max_input_candidates')}
            if context['candidate_audit'].get('unknown_exclusions'):
                result.search_status['message'] += ' Лоты с неизвестными свойствами исключены; сертификат относится только к поддержанному пулу.'
        if primary_only:
            from ._ehp_primary_only import annotate_status
            annotate_status(result.search_status, bool(result))
        return result
    quality_timeout = request_seconds(quality_timeout, deep_optimization)
    deadline = _deadline if _deadline is not None else start + quality_timeout
    if cancel_event is not None and cancel_event.is_set():
        return response(reason='cancelled')
    if request.get('min_budget') or request.get('optimization_mode', 'balanced') not in ('balanced', 'lexicographic'):
        raise ValueError('Max quality supports balanced or lexicographic objectives, not min_budget')
    if any(k.startswith('_') for k in request):
        raise ValueError('Private search hooks are not quality request parameters')
    if request.get('guaranteed_carryable'):
        return response(reason='unsupported_guaranteed_or_conservative')
    if scenario:
        from ._quality_scenario import current_views, physical_digest, metadata, source
        # These views retain their original descriptors; never compare a scenario
        # percentage with the stored physical percentage or weaken that check.
        verify = current_views
    else:
        verify = current_candidates
    if not request.get('priorities') or not any(v > 0 for v in request['priorities'].values()):
        raise ValueError('Max quality needs at least one positive priority')
    with (nullcontext(True) if _market is not None else quality_slot()) as acquired:
        if not acquired:
            return response(reason='quality_worker_busy' if acquired is False else 'quality_worker_unavailable')
        context = dict(deadline=deadline, max_profiles=limits.MAX_PROFILES,
                       max_input_candidates=limits.MAX_INPUT_CANDIDATES)
        b.build_artifact_sets(budget, _quality_context=context, **request)
        preparation = time.monotonic()-start
        if _market is not None and _market.changed():
            return response(reason='market_changed', timings=dict(preparation_seconds=preparation))
        if context.get('error'):
            return response(reason=context['error'], timings=dict(preparation_seconds=preparation))
        objective = context.get('objective')
        if objective is None or not context.get('profiles'):
            return response(reason='no_prepared_profiles')
        from ._survivability import requested, normalized_objective
        if requested(request.get('priorities'), request.get('min_thresholds'), request.get('max_thresholds')):
            if objective.mode != 'balanced':
                return response(reason='unsupported_survivability_lexicographic')
            try:
                objective = normalized_objective(objective, context['profiles'], request.get('objective_scales'))
            except (AssertionError, OverflowError, ValueError):
                return response(reason='survivability_numeric_limit')
            context['objective'] = objective
        models = []; builds = []; complete = True; candidate_total = 0; validation_seconds = 0.
        minimum = request.get('min_thresholds') or {}; maximum = request.get('max_thresholds') or {}
        duplicates = request.get('allow_duplicates', False); carry = request.get('carryable_only', True)
        region = context['request']['region']; effective_budget = context['budget']
        for container, original in context['profiles']:
            if primary_only and _primary_node_budget<=0:
                models.append(dict(state='UNKNOWN',reason='node_limit',priority_optimality_proven=False))
                complete=False;break
            if time.monotonic() >= deadline or (cancel_event is not None and cancel_event.is_set()):
                complete = False
                models.append(dict(state='UNKNOWN', reason='cancelled' if cancel_event and cancel_event.is_set() else 'request_time_limit'))
                break
            current = verify(original, b.get_connection, region, b._normalize_bonus_id)
            candidates = [o for o in current if limits.descriptor_supported(o)]
            if any(o.is_owned and (not limits.descriptor_supported(o)) for o in original):
                complete = False; models.append(dict(state='UNKNOWN', reason='unsupported_owned_descriptor')); continue
            owned = {}
            ambiguous = False
            for o in candidates:
                if o.is_owned:
                    physical = source(o) if scenario else o
                    value = (physical.item_id, physical.quality, physical.ptn, physical.art_percent, tuple(sorted(physical.unlocked_bonus_ids)))
                    key = physical_key(o)
                    ambiguous |= key in owned and owned[key] != value
                    owned[key] = value
            candidate_total += len(candidates)
            if ambiguous or candidate_total > limits.MAX_TOTAL_CANDIDATES:
                complete = False; models.append(dict(state='UNKNOWN', reason='ambiguous_owned_identity' if ambiguous else 'total_candidate_limit')); continue
            from ._quality_candidates import exact_reduce
            verified_before_reduction = len(candidates)
            candidates, coverage = exact_reduce(candidates, container.slots, duplicates)
            coverage.update(scope='full_supported_physical_profile' if context.get('candidate_audit', {}).get('coverage') == 'all_supported_physical_offers' else 'verified_candidate_provider_profile',
                            physical_or_descriptor_rejections=len(original)-verified_before_reduction)
            try:
                model = limits.make_model(candidates, container, objective, effective_budget, minimum, maximum, duplicates, carry)
            except limits.Unsupported as error:
                complete = False; models.append(dict(state='UNKNOWN', reason=str(error), candidate_coverage=coverage)); continue
            if len(candidates) < container.slots:
                models.append(dict(state='INFEASIBLE', reason='not_enough_verified_candidates', pool_sha256=model['pool_sha256'], candidate_coverage=coverage)); continue
            if _market is not None and _market.changed():
                return response(reason='market_changed', models=models, timings=dict(preparation_seconds=preparation,
                    validation_seconds=validation_seconds, worker_seconds=sum(m.get('worker_seconds',0) for m in models)))
            if deep_optimization:
                model = dict(model, model_seconds=DEEP_SECONDS, node_limit=DEEP_NODE_LIMIT, time_policy='deep_remaining', stage_reserve_seconds=limits.MODEL_SECONDS)
            if primary_only:
                from ._quality_scenario import source as original_source
                fields = ('auction_lot_id','physical_id','owned_instance_id','is_owned','item_id','quality','ptn','art_percent','stats_random','unlocked_bonus_ids','buyout_price')
                model = dict(model, primary_only=True, node_limit=min(model["node_limit"], _primary_node_budget), physical_descriptors=[{k:getattr(original_source(o),k,None) for k in fields} for o in candidates],
                    market_generation=_market.initial if _market is not None else None)
            result = run_model(model, deadline - (VALIDATION_RESERVE if deep_optimization else .1), cancel_event)
            if primary_only:
                records=result.get('stages', [])
                charged=(sum(s['node_count'] if isinstance(s.get('node_count'),int) and s['node_count']>=0 else s.get('node_limit',model['node_limit']) for s in records if s.get('kind') in ('primary_seed','priority')) if any(s.get('kind')=='priority' for s in records) else model['node_limit'])
                _primary_node_budget=max(0,_primary_node_budget-charged)
            t = time.monotonic(); valid_stages = []; invalid = False; validation_details = []
            # Native recomputation and market recheck replace every solver claim.
            for stage in result.get('stages', []):
                if 'indices' not in stage:
                    continue
                ids = stage['indices']
                if len(ids) != container.slots or len(set(ids)) != len(ids) or any(not isinstance(i, int) or not 0 <= i < len(candidates) for i in ids):
                    validation_details.append(dict(stage=stage.get('stage'), reason='invalid_solver_indices'))
                    invalid = True; continue
                combo = [candidates[i] for i in ids]
                verified = verify(combo, b.get_connection, region, b._normalize_bonus_id)
                if len(verified) != len(combo):
                    validation_details.append(dict(stage=stage.get('stage'), reason='physical_offer_validation_failed',
                        accepted=len(verified), required=len(combo), local_ids=[o.auction_lot_id for o in combo]))
                    invalid = True; continue
                checked = []
                b._process_combo(combo, container, sum(o.buyout_price for o in combo), context['request']['priorities'],
                    minimum, checked, effective_budget, objective=objective, max_thresholds=maximum,
                    require_carry=carry, allow_duplicates=duplicates)
                if not checked or not b._final_valid(checked[0], effective_budget, minimum, maximum, carry, False, duplicates):
                    validation_details.append(dict(stage=stage.get('stage'), reason='game_validation_failed'))
                    invalid = True; continue
                build = checked[0]; build._search_request = dict(context['request'], max_quality=True, quality_timeout=quality_timeout)
                if deep_optimization:
                    build._search_request['deep_optimization'] = True
                if primary_only:
                    build._search_request['primary_only'] = True
                b._decorate_armor(build)
                valid_stages.append((stage, build)); builds.append(build)
            validation_seconds += time.monotonic()-t
            proved = result.get('optimal', False) and not invalid and bool(valid_stages)
            if proved:
                final = valid_stages[-1][1]
                if objective.mode == 'balanced':
                    proved = all(objective.rank(final.combined_stats) >= objective.rank(x.combined_stats) for _, x in valid_stages)
                else:
                    ranks = objective.rank(final.combined_stats)
                    proved = tuple(result.get('certified_bins', ())) == ranks
                proved = proved and all(objective.sort_key(final) <= objective.sort_key(x) for _, x in valid_stages)
            if time.monotonic() >= deadline or (cancel_event is not None and cancel_event.is_set()):
                proved = False
            state = 'OPTIMAL' if proved else 'FEASIBLE' if valid_stages else 'INFEASIBLE' if result.get('infeasible') and not invalid else 'UNKNOWN'
            if state not in ('OPTIMAL', 'INFEASIBLE'):
                complete = False
            models.append(dict(result, state=state, container_id=container.item_id,
                armor_id=(getattr(container, '_armor', None) or {}).get('id'),
                candidates=len(candidates), omitted_candidates=len(original)-verified_before_reduction,
                candidate_coverage=coverage,
                **(dict(formulation=model['survivability_contract'], variables=model['variables'],
                        base_rows=model['base_rows'], base_nonzeros=model['base_nonzeros']) if model.get('survivability_contract') else {}),
                pool_sha256=model['pool_sha256'], rows=len(model['rows']), nonzeros=model['nonzeros'],
                validation_failed=invalid, worker_reason=result.get('reason', 'unknown'),
                validation_reason=validation_details[0]['reason'] if validation_details else 'passed',
                validation_details=validation_details))
            if deep_optimization:
                models[-1]['node_limit'] = model['node_limit']
            models[-1]['objective'] = objective.metadata()
            models[-1]['conditions'] = dict(slots=container.slots, budget=effective_budget,
                minimum=minimum, maximum=maximum, carry=carry, allow_duplicates=duplicates,
                container_effectiveness=container.effectiveness, container_protection=container.inner_protection,
                bonuses=container.bonus_stats, armor_level=(getattr(container, '_armor', None) or {}).get('level'))
            if scenario:
                models[-1]['physical_pool_sha256'] = physical_digest(candidates)
                models[-1]['conditions']['calculation'] = metadata(mode)
        builds.sort(key=(lambda build:(-build.combined_stats['effective_health'],tuple(sorted(str(physical_key(o)) for o in build.artifacts)))) if primary_only else objective.sort_key)
        winner = builds[:1]
        state = 'OPTIMAL' if winner and complete else 'FEASIBLE' if winner else 'INFEASIBLE' if models and complete else 'UNKNOWN'
        for build in winner:
            for option in build.artifacts:
                hidden = context.get('hidden_effects', {}).get(option.item_id)
                if hidden and not any(w['item_id']==option.item_id for w in build.hidden_effect_warnings):
                    build.hidden_effect_warnings.append(dict(item_id=option.item_id, name_ru=option.name_ru, **hidden))
        result = response(winner, state=state, reason='complete_full_rank' if complete and winner else models[-1].get('reason', 'unknown') if models else 'unknown',
            models=models, timings=dict(preparation_seconds=preparation, validation_seconds=validation_seconds,
                worker_seconds=sum(m.get('worker_seconds', 0) for m in models), total_seconds=time.monotonic()-start))
        result.search_status['limits']['seconds'] = quality_timeout
        if primary_only:
            from ._ehp_primary_only import annotate_status
            annotate_status(result.search_status, bool(result))
        return result


def build_max_quality_sets(budget, *, quality_timeout=limits.DEFAULT_SECONDS, deep_optimization=False, primary_only=False, cancel_event=None, **request):
    """At most two attempts under one slot and one original request deadline.

    A result is current only as of the last successful market/physical check;
    future auction availability is never guaranteed. Commits during preparation
    invalidate mixed reads, so only an unchanged attempt may produce a certificate.
    """
    from . import builder as b
    from ._quality_market import MarketWatch, MarketUnavailable, MIN_RETRY_SECONDS
    from .config import Config
    from ._ehp_primary_only import validate_request
    validate_request(primary_only, deep_optimization, True, request.get('priorities'), request.get('optimization_mode', 'balanced'), request.get('min_budget', False))
    if primary_only and request.get('tolerance_pct', 8.) != 0:
        raise ValueError('primary_only requires tolerance_pct=0: strict purchase budget')
    quality_timeout = request_seconds(quality_timeout, deep_optimization)
    # Preserve preflight behaviour without accessing the market for unsupported requests.
    if any(k.startswith('_') for k in request):
        raise ValueError('Private search hooks are not quality request parameters')
    if request.get('min_budget') or request.get('optimization_mode', 'balanced') not in ('balanced', 'lexicographic'):
        raise ValueError('Max quality supports balanced or lexicographic objectives, not min_budget')
    from ._quality_scenario import resolve_mode
    mode = resolve_mode(bool(request.get('conservative')), request.get('unknown_percent'))
    def response(reason, models=()):
        result = QualityResults(reason=reason, models=models, unknown_percent=mode)
        result.search_status['limits']['seconds'] = quality_timeout
        if deep_optimization:
            result.search_status['deep_optimization'] = True
            result.search_status['limits'].update(model_seconds=DEEP_SECONDS, validation_reserve_seconds=VALIDATION_RESERVE,
                time_policy='remaining_budget_with_later_stage_reserve', later_stage_reserve_seconds=limits.MODEL_SECONDS, nodes=DEEP_NODE_LIMIT)
        if primary_only:
            from ._ehp_primary_only import annotate_status
            annotate_status(result.search_status, bool(result))
        return result
    if request.get('guaranteed_carryable'):
        return response('unsupported_guaranteed_or_conservative')
    if not request.get('priorities') or not any(v > 0 for v in request['priorities'].values()):
        raise ValueError('Max quality needs at least one positive priority')
    if cancel_event is not None and cancel_event.is_set():
        return response('cancelled')
    start = time.monotonic(); deadline = start + quality_timeout; history = []; primary_nodes_left=DEEP_NODE_LIMIT
    with quality_slot() as acquired:
        if not acquired:
            return response('quality_worker_busy' if acquired is False else 'quality_worker_unavailable')
        try:
            with MarketWatch(b.get_connection, request.get('region') or Config.REGION) as market:
                for attempt in range(2):
                    if market.initial['running']:
                        result = response('market_refresh_in_progress')
                        break
                    result = _build_once(budget, quality_timeout=quality_timeout, cancel_event=cancel_event,
                        deep_optimization=deep_optimization, primary_only=primary_only, _deadline=deadline, _market=market, _primary_node_budget=primary_nodes_left, **request)
                    if primary_only:
                        for entry in result.search_status['models']:
                            for stage in entry.get('stages', []):
                                if stage.get('kind') in ('primary_seed','priority'):
                                    used=stage.get('node_count')
                                    primary_nodes_left=max(0,primary_nodes_left-(used if isinstance(used,int) and used>=0 else stage.get('node_limit',primary_nodes_left)))
                            if 'stages' in entry and not any(s.get('kind')=='priority' for s in entry.get('stages', [])):
                                # A killed solver may have spent every unreported node.
                                primary_nodes_left=0
                    changed = market.changed()
                    info = dict(attempt=attempt, market_before=market.initial, market_after=market.latest,
                        market_changed=changed, models=result.search_status['models'], timings=dict(result.search_status['timings']))
                    history.append(info)
                    if not changed:
                        break
                    # Discard every previous build and proof; never remap local IDs.
                    result = response('market_changed', info['models'])
                    if primary_only and primary_nodes_left<=0:
                        break
                    remaining = deadline-time.monotonic()
                    if attempt == 1 or market.latest['running'] or not market.latest['tracked']:
                        break
                    if cancel_event is not None and cancel_event.is_set():
                        result = response('cancelled', info['models']); break
                    if remaining < MIN_RETRY_SECONDS:
                        result = response('market_changed_insufficient_time', info['models']); break
                    market.initial = market.latest
                    market.changed_seen = False
        except (MarketUnavailable, sqlite3.Error):
            result = response('market_state_unavailable', history[-1]['models'] if history else ())
    status = result.search_status
    if deep_optimization:
        status['deep_optimization'] = True
        status['limits'].update(seconds=DEEP_SECONDS, model_seconds=DEEP_SECONDS,
            time_policy='remaining_budget_with_later_stage_reserve', validation_reserve_seconds=VALIDATION_RESERVE,
            later_stage_reserve_seconds=limits.MODEL_SECONDS, nodes=DEEP_NODE_LIMIT)
    status['attempts'] = history
    status['retry_count'] = max(0, len(history)-1)
    status['worker_reason'] = (status['models'][-1].get('worker_reason', status['models'][-1].get('reason')) if status['models'] else None)
    status['validation_reason'] = ('market_changed' if status['reason'].startswith('market_') else
        status['models'][-1].get('validation_reason') if status['models'] else None)
    if status['state']=='UNKNOWN' and status['validation_reason'] not in (None, 'passed', 'market_changed'):
        status['reason'] = status['validation_reason']
        status['message'] = 'UNKNOWN: Модель завершена, но финальная проверка сборки не пройдена. '+status['validation_reason']
    if status['reason'].startswith('market_'):
        status['message'] = ('UNKNOWN: Не удалось подтвердить согласованное состояние аукциона. Повторите запрос.'
            if status['reason']=='market_state_unavailable' else 'UNKNOWN: Аукцион обновился во время поиска. Повторите запрос.')
    if deep_optimization and status['reason'] in ('solver_error', 'time_or_node_limit') and status['validation_reason'] in (None, 'passed'):
        if any((stage.get('node_count') or 0) >= item.get('node_limit', DEEP_NODE_LIMIT) for item in status['models'] for stage in item.get('stages', ())):
            # Some SciPy/HiGHS versions expose max-nodes as status 4/16.
            # Preserve every raw status/message and worker_reason, clarify the cap.
            status['reason'] = 'node_limit'
            status['message'] = status['state'] + ': Достигнут лимит узлов MILP. Оптимальность не доказана.'
    if (status['state'] == 'FEASIBLE' and status['reason'] in ('solution_limit', 'node_limit', 'time_limit')
            and status['validation_reason'] == 'passed' and len(status['models']) == 1
            and status['models'][0].get('priority_optimality_proven')):
        status['message'] += ' Основная цель доказана в проверенном пуле; минимальная стоимость не доказана.'
    status['timings'].update(total_seconds=time.monotonic()-start,
        preparation_seconds=sum(x['timings'].get('preparation_seconds',0) for x in history),
        validation_seconds=sum(x['timings'].get('validation_seconds',0) for x in history),
        worker_seconds=sum(m.get('worker_seconds',0) for x in history for m in x['models']))
    if primary_only:
        from ._ehp_primary_only import annotate_status
        annotate_status(result.search_status, bool(result))
    return result
