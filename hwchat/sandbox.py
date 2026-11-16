"""The only place in the program that touches the operating system.

Two capabilities are exposed, both read-only and both closed-world:

- ``run(key)``      - execute one of a handful of pre-registered commands.
- ``read_file(p)``  - read a file, but only from a short list of safe locations.

Nothing here ever accepts text that came from the language model, so there is
no way for a model (or a prompt-injection) to widen what can be executed.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass

SEARCH_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
TIMEOUT_SECONDS = 6


class SandboxError(Exception):
    """Raised when a command or file cannot be used; message is user-presentable."""


@dataclass(frozen=True)
class Command:
    argv: tuple
    needs_root: bool = False


# Every command the program may ever run. Arguments are fixed at build time.
REGISTRY: dict[str, Command] = {
    "lscpu": Command(("lscpu",)),
    "lsblk": Command(
        ("lsblk", "--json", "--bytes", "--output",
         "NAME,SIZE,TYPE,MOUNTPOINT,MODEL,ROTA,TRAN")
    ),
    "lspci": Command(("lspci", "-mm")),
    "nvidia": Command(
        ("nvidia-smi", "--query-gpu=name,memory.total,driver_version",
         "--format=csv,noheader")
    ),
    # Needs root. The sudoers rule installed by deploy/ pins these exact args.
    "dmi_memory": Command(("/usr/sbin/dmidecode", "-t", "memory"), needs_root=True),
}

# Files that may be read. Anything else is refused.
READABLE_PREFIXES = (
    "/proc/meminfo",
    "/proc/cpuinfo",
    "/etc/os-release",
    "/sys/class/dmi/id/",
    "/sys/class/power_supply/",
    "/sys/block/",
)
# DMI files that hold identifiers we deliberately never read.
BLOCKED_NAMES = ("product_serial", "board_serial", "chassis_serial",
                 "product_uuid", "chassis_asset_tag")


def _is_allowed_real_path(real: str) -> bool:
    for prefix in READABLE_PREFIXES:
        if prefix.endswith("/"):
            if real.startswith(prefix):
                return True
        else:
            if real == prefix:
                return True
    return False


def is_installed(key: str) -> bool:
    cmd = REGISTRY[key]
    binary = cmd.argv[0]
    if os.path.isabs(binary):
        return os.access(binary, os.X_OK)
    return shutil.which(binary, path=SEARCH_PATH) is not None


def run(key: str) -> str:
    """Run a registered command and return its stdout."""
    if key not in REGISTRY:
        raise SandboxError(f"'{key}' is not an allow-listed command")
    cmd = REGISTRY[key]
    argv = list(cmd.argv)
    if not os.path.isabs(argv[0]):
        found = shutil.which(argv[0], path=SEARCH_PATH)
        if not found:
            raise SandboxError(f"'{argv[0]}' is not installed on this system")
        argv[0] = found
    elif not os.access(argv[0], os.X_OK):
        raise SandboxError(f"'{argv[0]}' is not installed on this system")

    if cmd.needs_root and os.geteuid() != 0:
        sudo = shutil.which("sudo", path=SEARCH_PATH)
        if not sudo:
            raise SandboxError("this data needs root and sudo is not installed")
        argv = [sudo, "-n", *argv]  # -n: never prompt, fail immediately

    try:
        done = subprocess.run(
            argv, capture_output=True, text=True, shell=False,
            timeout=TIMEOUT_SECONDS, env={"PATH": SEARCH_PATH, "LC_ALL": "C"},
        )
    except subprocess.TimeoutExpired:
        raise SandboxError(f"'{cmd.argv[0]}' took longer than {TIMEOUT_SECONDS}s")
    except OSError as exc:
        raise SandboxError(f"could not start '{cmd.argv[0]}': {exc}")

    if done.returncode != 0:
        detail = (done.stderr or "").strip().splitlines()
        hint = detail[0][:160] if detail else f"exit code {done.returncode}"
        if cmd.needs_root and "password" in (done.stderr or "").lower():
            hint = "root privileges required (see deploy/hwchat.sudoers)"
        raise SandboxError(f"'{cmd.argv[0]}' failed: {hint}")
    return done.stdout


def read_file(path: str) -> str:
    real = os.path.realpath(path)
    if not _is_allowed_real_path(real) or os.path.basename(real) in BLOCKED_NAMES:
        raise SandboxError(f"reading '{path}' is not permitted")
    try:
        with open(real, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read(1_000_000)
    except OSError as exc:
        raise SandboxError(f"cannot read '{path}': {exc.strerror or exc}")


def list_dir(path: str) -> list[str]:
    real = os.path.realpath(path)
    real_dir = real if real.endswith("/") else real + "/"
    if not any(real_dir.startswith(p) for p in READABLE_PREFIXES if p.endswith("/")):
        raise SandboxError(f"listing '{path}' is not permitted")
    try:
        return sorted(os.listdir(real))
    except OSError:
        return []