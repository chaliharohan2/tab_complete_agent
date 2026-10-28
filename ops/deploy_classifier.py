"""
Deploy/restart the classifier microservice on the GPU box.

Usage:
    GPU_HOST=... GPU_USER=... GPU_PASSWORD=... python ops/deploy_classifier.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import remote  # noqa: E402

REMOTE_DIR = "~/tab_complete_poc_bench"
REMOTE_VENV = f"{REMOTE_DIR}/venv"
REMOTE_LOG_PATH = f"{REMOTE_DIR}/service.log"
PORT = int(os.environ.get("CLASSIFIER_PORT", "8800"))

LOCAL_SERVICE_DIR = os.path.join(os.path.dirname(__file__), "..", "classifier_service")
SERVICE_FILES = ["service.py", "router.py", "rules.py"]


def main():
    print("1. Ensuring remote deps are installed...")
    out, err, rc = remote.run(
        f"source {REMOTE_VENV}/bin/activate && pip install -q fastapi 'uvicorn[standard]' pydantic"
    )
    print(out, err)

    print("2. Uploading service code...")
    # upload() uses sftp with absolute paths on the remote side; expand ~ first.
    home_out, _, _ = remote.run("echo $HOME")
    home = home_out.strip()
    remote_dir = REMOTE_DIR.replace("~", home)
    for name in SERVICE_FILES:
        remote.upload(os.path.join(LOCAL_SERVICE_DIR, name), f"{remote_dir}/{name}")

    print("3. Stopping any previous instance...")
    remote.run("pkill -f 'uvicorn service:app' || true")
    time.sleep(1)

    print("4. Starting service in background (nohup)...")
    remote_dir_abs = REMOTE_DIR.replace("~", home)
    remote_venv_abs = REMOTE_VENV.replace("~", home)
    remote_log_abs = REMOTE_LOG_PATH.replace("~", home)
    start_cmd = (
        f"cd {remote_dir_abs} && source {remote_venv_abs}/bin/activate && "
        f"nohup python -m uvicorn service:app --host 0.0.0.0 --port {PORT} "
        f"> {remote_log_abs} 2>&1 < /dev/null & disown; exit 0"
    )
    # Fire-and-forget: the SSH channel can otherwise hang waiting for the
    # backgrounded process's inherited fds to close even though nohup detaches it.
    try:
        remote.run(start_cmd, timeout=10)
    except Exception:
        pass

    print("5. Waiting for it to come up...")
    for _ in range(90):  # loading an 8B model takes a minute or two
        time.sleep(2)
        out, err, rc = remote.run(f"curl -s http://localhost:{PORT}/health || true")
        if '"status":"ok"' in out.replace(" ", ""):
            print("Service is up:", out)
            return
    print("Service did not report healthy in time. Recent log:")
    out, _, _ = remote.run(f"tail -n 50 {remote_log_abs}")
    print(out)
    sys.exit(1)


if __name__ == "__main__":
    main()
