"""One native quality task per application directory, across Windows workers."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
import tempfile
import time


@contextmanager
def quality_slot():
    key = hashlib.sha256(str(Path(__file__).resolve().parent).encode()).hexdigest()[:20]
    path = Path(tempfile.gettempdir()) / ('stalzone-quality-' + key + '.lock')
    try:
        stream = open(path, 'a+b')
    except OSError:
        yield None
        return
    acquired = False
    try:
        if stream.tell() == 0:
            stream.write(b'0'); stream.flush()
        stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except OSError:
            pass
        yield acquired
    finally:
        if acquired:
            stream.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_UN)
        stream.close()


def resident_bytes(pid):
    if os.name == 'nt':
        import ctypes as c
        from ctypes import wintypes as w
        class Counters(c.Structure):
            _fields_ = [('cb', w.DWORD), ('PageFaultCount', w.DWORD)] + [(k, c.c_size_t) for k in
                ('PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage',
                 'QuotaPeakNonPagedPoolUsage', 'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
        kernel = c.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.argtypes = [w.DWORD, w.BOOL, w.DWORD]; kernel.OpenProcess.restype = w.HANDLE
        kernel.CloseHandle.argtypes = [w.HANDLE]
        api = c.WinDLL('psapi', use_last_error=True)
        api.GetProcessMemoryInfo.argtypes = [w.HANDLE, c.c_void_p, w.DWORD]
        handle = kernel.OpenProcess(0x410, False, pid)
        if not handle:
            return None
        try:
            info = Counters(); info.cb = c.sizeof(info)
            return int(info.WorkingSetSize) if api.GetProcessMemoryInfo(handle, c.byref(info), info.cb) else None
        finally:
            kernel.CloseHandle(handle)
    try:
        for line in Path(f'/proc/{pid}/status').read_text().splitlines():
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def run_model(model, deadline, cancel_event=None):
    remaining = deadline - time.monotonic()
    if remaining <= 0 or (cancel_event is not None and cancel_event.is_set()):
        return dict(stages=[], optimal=False, infeasible=False, reason='cancelled' if cancel_event and cancel_event.is_set() else 'request_time_limit')
    model = dict(model, seconds=remaining, deadline=deadline)
    encoded = json.dumps(model, allow_nan=False).encode()
    if len(encoded) > 8 * 1024**2:
        return dict(stages=[], optimal=False, infeasible=False, reason='payload_limit')
    with tempfile.TemporaryDirectory(prefix='stz-quality-') as directory:
        source = Path(directory)/'input.json'; output = Path(directory)/'output.json'
        source.write_bytes(encoded)
        root = str(Path(__file__).resolve().parent.parent)
        executable = sys.executable
        module_paths = [root]
        if os.name == 'nt' and sys.prefix != sys.base_prefix:
            # The Windows venv executable is a redirector that creates another
            # process. Monitor/terminate the actual interpreter, not that shim.
            executable = getattr(sys, '_base_executable', '')
            if not executable or not Path(executable).is_file():
                return dict(stages=[], optimal=False, infeasible=False, reason='worker_runtime_unavailable')
            module_paths.extend(dict.fromkeys(sysconfig.get_path(k) for k in ('purelib', 'platlib')))
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(module_paths), PYTHONDONTWRITEBYTECODE='1', PYTHON_DOTENV_DISABLED='1',
                   OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        start = time.monotonic(); peak = 0; stop_reason = None
        try:
            process = subprocess.Popen([executable, '-m', 'app._max_quality_worker', str(source), str(output)],
                cwd=root, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        except OSError:
            return dict(stages=[], optimal=False, infeasible=False, reason='worker_start_error')
        try:
            while process.poll() is None:
                memory = resident_bytes(process.pid)
                if memory is not None:
                    peak = max(peak, memory)
                if cancel_event is not None and cancel_event.is_set():
                    stop_reason = 'cancelled'
                elif time.monotonic() >= deadline:
                    stop_reason = 'request_time_limit'
                elif memory is not None and memory > model['memory_bytes']:
                    stop_reason = 'memory_limit'
                if stop_reason:
                    process.terminate(); break
                time.sleep(.02)
            process.wait(timeout=2)
        finally:
            if process.poll() is None:
                process.kill(); process.wait(timeout=2)
        try:
            state = json.loads(output.read_text(encoding='utf8'))
        except (OSError, ValueError):
            state = dict(stages=[], optimal=False, infeasible=False, reason='worker_failed')
        if stop_reason or process.returncode:
            state.update(optimal=False, infeasible=False, reason=stop_reason or 'worker_failed')
        state.update(worker_seconds=time.monotonic()-start, sampled_worker_peak_rss=peak,
                     worker_returncode=process.returncode, worker_pid=process.pid)
        if state.get('process_id') is not None and state['process_id'] != process.pid:
            state.update(optimal=False, infeasible=False, reason='unexpected_worker_process')
        return state
