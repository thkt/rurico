"""Own external command lifetimes for candidate verification."""
import os
import signal
import subprocess
import time


def owned_run(args, check=False, graceful=False, **kwargs):
    """Reclaim the command and all inherited-group workers before returning."""
    process = subprocess.Popen(args, start_new_session=True, **kwargs)
    try:
        returncode = process.wait()
        if check and returncode:
            raise subprocess.CalledProcessError(returncode, args)
        return subprocess.CompletedProcess(args, returncode)
    finally:
        # Do not let a second interrupt leave workers writing after incomplete.
        handlers = {sig:signal.signal(sig, signal.SIG_IGN) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            if graceful and process.poll() is None:
                # Interrupt only the runner first: its finally blocks reclaim
                # test entries that own separate sessions, before group cleanup.
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except PermissionError:
                # macOS can report EPERM for a group containing only zombies.
                pass
            process.wait()
            deadline = time.monotonic()+2
            while True:
                try:
                    os.killpg(process.pid, 0)
                except ProcessLookupError:
                    break
                except PermissionError:
                    pass
                if time.monotonic() >= deadline:
                    raise RuntimeError("owned process group could not be collected")
                time.sleep(.01)
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)


if __name__ == "__main__":
    import runpy

    def interrupted(signum, frame):
        raise KeyboardInterrupt("light tests interrupted")

    signal.signal(signal.SIGTERM, interrupted)
    runpy.run_module("unittest", run_name="__main__")


