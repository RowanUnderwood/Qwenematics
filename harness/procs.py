"""Find and stop the local processes behind the harness's servers (Windows).

ComfyUI instances are started by launcher .bat files in their own console windows:
cmd.exe (the .bat) -> python main.py. Killing only python would leave the window
sitting at the launcher's final `pause`, so the cmd.exe parent goes too when it is
the launcher's window. Only the process listening on the given port is touched,
so other ComfyUI instances (e.g. Stability Matrix on :8188) are never affected.
"""
from __future__ import annotations

import json
import logging
import socket
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def _ps(script: str) -> str:
    r = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
                       capture_output=True, text=True, timeout=30, creationflags=NO_WINDOW)
    return r.stdout.strip()


def listener(port: int) -> dict | None:
    """The process listening on 127.0.0.1:<port>, with its parent: {pid, name, parent_pid, parent_name, parent_cmd}."""
    out = _ps(
        f"$c = Get-NetTCPConnection -LocalPort {port} -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1;"
        "if ($c) { $p = Get-CimInstance Win32_Process -Filter \"ProcessId=$($c.OwningProcess)\";"
        "$q = Get-CimInstance Win32_Process -Filter \"ProcessId=$($p.ParentProcessId)\";"
        "@{pid=$p.ProcessId; name=$p.Name; parent_pid=$p.ParentProcessId; parent_name=$q.Name;"
        " parent_cmd=$q.CommandLine} | ConvertTo-Json -Compress }")
    try:
        return json.loads(out) if out else None
    except json.JSONDecodeError:
        return None


def stop_port(port: int, launcher: Path | None = None) -> str:
    """Stop whatever listens on <port>; also close its console window if that window
    is the given launcher .bat. Returns a short description of what was done."""
    info = listener(port)
    if not info:
        return f":{port} was not running"
    target = info["pid"]
    pcmd = (info.get("parent_cmd") or "").lower()
    if (info.get("parent_name") or "").lower() == "cmd.exe" and launcher and launcher.name.lower() in pcmd:
        target = info["parent_pid"]        # the launcher's window: /T takes python with it
    subprocess.run(["taskkill", "/PID", str(target), "/T", "/F"], capture_output=True, creationflags=NO_WINDOW)
    log.info("stopped :%s (pid %s, %s)", port, target, info.get("name"))
    return f"stopped :{port} ({info.get('name')} pid {info['pid']})"


def close_windows(launcher: Path) -> int:
    """Close cmd.exe windows running the given launcher .bat (e.g. one left at its
    final `pause` after its server exited). Returns how many were closed."""
    name = launcher.name.replace("'", "''")
    out = _ps(f"$n = 0; Get-CimInstance Win32_Process -Filter \"Name='cmd.exe'\" | "
              f"Where-Object {{ $_.CommandLine -like '*{name}*' }} | "
              "ForEach-Object { taskkill /PID $_.ProcessId /T /F | Out-Null; $n++ }; $n")
    return int(out) if out.isdigit() else 0


def run_bat(path: Path, timeout: float = 60) -> str:
    r = subprocess.run(["cmd.exe", "/c", str(path)], cwd=str(path.parent), capture_output=True, text=True,
                       timeout=timeout, creationflags=NO_WINDOW, stdin=subprocess.DEVNULL)
    return (r.stdout + r.stderr).strip()
