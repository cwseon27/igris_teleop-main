import os
import subprocess
import signal
import shutil
import logging_mp


logger = logging_mp.get_logger(__name__)

NGROK_PORT = 8012
AUTO_NGROK = os.environ.get("AUTO_NGROK", "1") not in ("0", "false", "False")


def _start_ngrok(port: int = NGROK_PORT) -> subprocess.Popen | None:
    """Launch ngrok http tunnel in the background so manual terminal is not needed."""
    ngrok_bin = shutil.which("ngrok")
    if not ngrok_bin:
        logger.warning("[MAIN] ngrok executable not found; skipping tunnel.")
        return None
    if not AUTO_NGROK:
        logger.info("[MAIN] AUTO_NGROK=0 → skipping ngrok start")
        return None
    _kill_existing_ngrok(port)
    try:
        proc = subprocess.Popen(
            [ngrok_bin, "http", str(port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        logger.info("[MAIN] started ngrok http %s (pid=%s)", port, proc.pid)
        return proc
    except Exception:
        logger.exception("[MAIN] Failed to start ngrok on port %s", port)
        return None


def _stop_ngrok(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    if proc.poll() is not None:
        logger.info("[MAIN] ngrok already exited (code=%s)", proc.returncode)
        return
    # kill the whole process group to avoid lingering tunnels
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except Exception:
        proc.terminate()
    try:
        proc.wait(timeout=5)
        logger.info("[MAIN] ngrok terminated.")
    except subprocess.TimeoutExpired:
        logger.warning("[MAIN] ngrok did not exit in time; killing.")
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        else:
            logger.info("[MAIN] ngrok killed.")



def _kill_existing_ngrok(port: int = NGROK_PORT) -> None:
    """Terminate stale ngrok http processes on the same port to avoid ERR_NGROK_334."""
    try:
        subprocess.run(["pkill", "-f", f"ngrok http {port}"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        # pkill 없는 환경이면 그냥 스킵
        return
