"""Private short-lived HiGHS worker. Never imports app.main or opens a database."""
import json
import math
import os
import re
from pathlib import Path
import sys
import time
from ._quality_time import stage_seconds


def _finite_number(value):
    """Optional solver metadata is not a certificate when absent or malformed."""
    if value is None or isinstance(value, (str, bytes, bool)):
        return None
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _stop_reason(status, message, nodes, node_limit):
    # SciPy 1.18 maps native kSolutionLimit=16 to status4, while preserving
    # its numeric native code only in this standardized message. This parses
    # termination provenance, NEVER the existence or validity of an incumbent.
    match = re.search(r'\bHiGHS Status (\d+):', message)
    native = int(match.group(1)) if match else None
    if status == 2:
        # This SciPy mapping also labels native ModelError=2 as infeasible.
        return ('infeasible' if native in (None, 8) else 'solver_error'), native
    if status in (1, 4):
        if native == 16:
            return ('node_limit' if nodes is not None and nodes >= node_limit else 'solution_limit'), native
        if status == 1:
            return ('time_limit' if native == 13 else 'time_or_node_limit'), native
    return 'solver_error', native


def memory_limit(size):
    # Windows process commit cap, plus parent RSS watchdog and termination.
    if os.name == 'nt':
        import ctypes as c
        from ctypes import wintypes as w
        class Basic(c.Structure):
            _fields_ = [('PerProcessUserTimeLimit', c.c_longlong), ('PerJobUserTimeLimit', c.c_longlong),
                        ('LimitFlags', w.DWORD), ('MinimumWorkingSetSize', c.c_size_t),
                        ('MaximumWorkingSetSize', c.c_size_t), ('ActiveProcessLimit', w.DWORD),
                        ('Affinity', c.c_size_t), ('PriorityClass', w.DWORD), ('SchedulingClass', w.DWORD)]
        class IO(c.Structure):
            _fields_ = [(name, c.c_ulonglong) for name in ('ReadOperationCount', 'WriteOperationCount',
                'OtherOperationCount', 'ReadTransferCount', 'WriteTransferCount', 'OtherTransferCount')]
        class Extended(c.Structure):
            _fields_ = [('BasicLimitInformation', Basic), ('IoInfo', IO), ('ProcessMemoryLimit', c.c_size_t),
                        ('JobMemoryLimit', c.c_size_t), ('PeakProcessMemoryUsed', c.c_size_t),
                        ('PeakJobMemoryUsed', c.c_size_t)]
        kernel = c.WinDLL('kernel32', use_last_error=True)
        kernel.CreateJobObjectW.restype = w.HANDLE
        kernel.GetCurrentProcess.restype = w.HANDLE
        kernel.SetInformationJobObject.argtypes = [w.HANDLE, c.c_int, c.c_void_p, w.DWORD]
        kernel.AssignProcessToJobObject.argtypes = [w.HANDLE, w.HANDLE]
        job = kernel.CreateJobObjectW(None, None)
        info = Extended(); info.BasicLimitInformation.LimitFlags = 0x100; info.ProcessMemoryLimit = size
        if not job or not kernel.SetInformationJobObject(job, 9, c.byref(info), c.sizeof(info)) or not kernel.AssignProcessToJobObject(job, kernel.GetCurrentProcess()):
            raise RuntimeError('memory_limit_unavailable')
        return job  # Keep the handle alive until process exit.
    else:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (3 * 1024**3, 3 * 1024**3))


def solve(model, output, *, solver=None):
    if model.get('primary_only'):
        from ._ehp_primary_only import solve as primary_solve
        try:
            return primary_solve(model, output, solver=solver)
        except RuntimeError as error:
            if 'std::bad_alloc' in str(error):
                raise MemoryError('std::bad_alloc') from error
            raise
    if model.get('survivability_contract'):
        from ._survivability_worker import solve as derived_solve
        return derived_solve(model, output, solver=solver)
    import numpy as np
    from scipy.optimize import milp, Bounds, LinearConstraint
    from scipy.sparse import csr_matrix
    solver = solver or milp
    # Python's monotonic clock is shared across processes on this machine.
    # Startup/import time must not restart the request's remaining budget.
    deadline = model.get('deadline', time.monotonic() + model['seconds'])
    rows = [list(r['values']) for r in model['rows']]
    lower = [-np.inf if r['lo'] is None else r['lo'] for r in model['rows']]
    upper = [np.inf if r['hi'] is None else r['hi'] for r in model['rows']]
    n = model['n']; history = []; certificates = []; bins = []
    state = dict(reason='time_limit', optimal=False, infeasible=False, stages=history,
                 pool_sha256=model.get('pool_sha256'), certified_bins=bins,
                 memory_limit_kind=model.get('memory_limit_kind'), process_id=os.getpid())

    def save():
        temporary = str(output) + '.tmp'
        Path(temporary).write_text(json.dumps(state, allow_nan=False), encoding='utf8')
        os.replace(temporary, output)

    objectives = [(-np.array(v, dtype=float), c) for v, c in zip(model['goals'], model['constants'])]
    objectives.append((np.array(model['prices'], dtype=float) / 1e6, 0.))
    for stage, (cost, constant) in enumerate(objectives):
        remaining = stage_seconds(model, stage, len(objectives), deadline - time.monotonic())
        if remaining <= 0:
            save(); return state
        stage_start = time.monotonic()
        r = solver(cost, integrality=np.ones(n), bounds=Bounds(0, 1),
                   constraints=LinearConstraint(csr_matrix(np.array(rows)), lower, upper),
                   options=dict(time_limit=remaining, node_limit=model['node_limit'], mip_rel_gap=0.))
        bound = _finite_number(getattr(r, 'mip_dual_bound', None))
        gap = _finite_number(getattr(r, 'mip_gap', None))
        record = dict(stage=stage, kind='cost' if stage == len(objectives)-1 else 'priority',
                      time_limit_seconds=remaining, elapsed_seconds=time.monotonic()-stage_start,
                      solver_status=int(r.status), solver_message=str(r.message)[:1000],
                      bound=bound, gap=gap)
        record['objective_upper_bound'] = -record['bound'] + constant if record['kind']=='priority' and record['bound'] is not None else None
        record['cost_lower_bound'] = record['bound'] * 1e6 if record['kind']=='cost' and record['bound'] is not None else None
        try:
            primal = np.asarray(r.x, dtype=float) if getattr(r, 'x', None) is not None else None
        except (TypeError, ValueError):
            primal = None
        if (primal is not None and primal.shape == (n,) and np.isfinite(primal).all()
                and np.max(abs(primal-np.rint(primal)), initial=0) <= 1e-5
                and np.all((np.rint(primal) >= 0) & (np.rint(primal) <= 1))):
            selected = np.flatnonzero(primal > .5)
            record['indices'] = selected.tolist()
            record['primal_objective'] = float(cost @ (primal > .5))
        nodes = _finite_number(getattr(r, 'mip_node_count', None))
        record['node_count'] = int(nodes) if nodes is not None and nodes >= 0 and nodes.is_integer() else None
        history.append(record)
        if 'indices' not in record or r.status != 0:
            reason, native = _stop_reason(r.status, record['solver_message'], record['node_count'], model['node_limit'])
            record.update(termination_reason=reason, highs_status=native)
            infeasible = reason == 'infeasible' and stage == 0 and 'indices' not in record
            state.update(infeasible=infeasible,
                         reason='secondary_infeasible' if reason == 'infeasible' and stage > 0 else reason,
                         priority_optimality_proven=bool(model['goals']) and len(certificates) == len(model['goals']),
                         cost_optimality_proven=False)
            save(); return state
        x = np.zeros(n); x[record['indices']] = 1
        if stage == len(objectives)-1:
            paid = sum(model['prices'][i] for i in record['indices'])
            state['optimal'] = record['bound'] is not None and math.ceil(record['bound'] * 1e6 - 1e-6) >= paid
            state['reason'] = 'finished' if state['optimal'] else 'unproved_cost_tie'
            save(); return state
        vector = -cost
        target = float(vector @ x)
        if model['mode'] == 'lexicographic':
            value = target + constant; chosen_bin = math.floor(value / model['tolerance'] + .5)
            bound_bin = math.floor((-record['bound'] + constant) / model['tolerance'] + .5) if record['bound'] is not None else None
            if chosen_bin != bound_bin:
                state['reason'] = 'unproved_priority_bin'; save(); return state
            bins.append(chosen_bin)
            rows.append((vector/model['tolerance']).tolist())
            lower.append(chosen_bin-.5-constant/model['tolerance'])
            upper.append(float(np.nextafter(chosen_bin+.5, -np.inf))-constant/model['tolerance'])
        else:
            if record['bound'] is None or abs(-record['bound']-target) > 1e-7 * max(1., abs(target)):
                state['reason'] = 'unproved_primary'; save(); return state
            rows.append((vector * 1e6).tolist()); lower.append(target * 1e6); upper.append(np.inf)
        certificates.append(record.copy())
        save()  # Parent can retain this incumbent if the next stage is killed.
    return state


def main():
    payload, output = map(Path, sys.argv[1:3])
    try:
        model = json.loads(payload.read_text(encoding='utf8'))
        handle = memory_limit(model['memory_bytes'])
        model['memory_limit_kind'] = 'windows_process_commit' if os.name == 'nt' else 'posix_address_space'
        solve(model, output)
    except Exception as error:
        # Preserve an earlier completed stage, if any, but never its proof flag.
        state = json.loads(output.read_text(encoding='utf8')) if output.exists() else dict(stages=[])
        state.update(optimal=False, infeasible=False, reason='dependency_unavailable' if isinstance(error, ImportError) else 'memory_limit' if isinstance(error, MemoryError) else 'worker_error', error=type(error).__name__)
        temporary = str(output) + '.tmp'
        Path(temporary).write_text(json.dumps(state), encoding='utf8')
        os.replace(temporary, output)


if __name__ == '__main__':
    main()
