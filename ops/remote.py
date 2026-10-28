"""
Small SSH helper for deploying/managing the classifier microservice on the
GPU box. Credentials come from environment variables, falling back to the
gitignored .env file at the repo root -- never hardcode them here.

Required: GPU_HOST, GPU_USER, GPU_PASSWORD
"""
import os
import sys

import paramiko

_ENV_FILE = os.path.join(os.path.dirname(__file__), "..", ".env")
if os.path.exists(_ENV_FILE):
    for _line in open(_ENV_FILE):
        _k, _, _v = _line.strip().partition("=")
        if _k and not _k.startswith("#"):
            os.environ.setdefault(_k, _v)

GPU_HOST = os.environ["GPU_HOST"]
GPU_USER = os.environ["GPU_USER"]
GPU_PASSWORD = os.environ["GPU_PASSWORD"]


def _client() -> paramiko.SSHClient:
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(GPU_HOST, username=GPU_USER, password=GPU_PASSWORD, timeout=15)
    return c


def run(cmd: str, timeout: int = 600) -> tuple[str, str, int]:
    c = _client()
    try:
        stdin, stdout, stderr = c.exec_command(cmd, timeout=timeout)
        out = stdout.read().decode(errors="replace")
        err = stderr.read().decode(errors="replace")
        rc = stdout.channel.recv_exit_status()
        return out, err, rc
    finally:
        c.close()


def upload(local_path: str, remote_path: str) -> None:
    c = _client()
    try:
        sftp = c.open_sftp()
        sftp.put(local_path, remote_path)
        sftp.close()
    finally:
        c.close()


if __name__ == "__main__":
    out, err, rc = run(" ".join(sys.argv[1:]))
    print(out)
    if err.strip():
        print("STDERR:", err, file=sys.stderr)
    sys.exit(rc)
