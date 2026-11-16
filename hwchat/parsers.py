"""Pure functions: raw text in, normalised dicts out. No I/O, easy to unit-test."""
from __future__ import annotations

import json
import re
import shlex

_UNIT_TO_MB = {"KB": 1 / 1024, "MB": 1, "GB": 1024, "TB": 1024 * 1024}


def size_to_mb(text: str | None) -> int | None:
    """'16 GB' -> 16384, '8192 MB' -> 8192, 'No Module Installed' -> None."""
    if not text:
        return None
    m = re.match(r"\s*(\d+)\s*([KMGT]B)\b", text, re.IGNORECASE)
    if not m:
        return None
    return int(int(m.group(1)) * _UNIT_TO_MB[m.group(2).upper()])


def mb_to_gb(mb: int | float | None) -> float | None:
    return None if mb is None else round(mb / 1024, 1)


def _kv_lines(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        if ":" in line:
            key, _, val = line.partition(":")
            out.setdefault(key.strip(), val.strip())
    return out


# ---------------------------------------------------------------- CPU
def parse_lscpu(text: str) -> dict:
    raw = _kv_lines(text)

    def num(key):
        try:
            return int(raw[key])
        except (KeyError, ValueError):
            return None

    def mhz(key):
        try:
            return round(float(raw[key]))
        except (KeyError, ValueError):
            return None

    result = {
        "model": raw.get("Model name"),
        "architecture": raw.get("Architecture"),
        "logical_cpus": num("CPU(s)"),
        "sockets": num("Socket(s)"),
        "cores_per_socket": num("Core(s) per socket"),
        "threads_per_core": num("Thread(s) per core"),
        "max_mhz": mhz("CPU max MHz"),
        "l3_cache": raw.get("L3 cache"),
    }
    sockets, cores = result["sockets"], result["cores_per_socket"]
    if sockets and cores:
        result["physical_cores"] = sockets * cores
    return {k: v for k, v in result.items() if v is not None}


def parse_cpuinfo(text: str) -> dict:
    """Fallback when lscpu is missing."""
    model = None
    count = 0
    for line in text.splitlines():
        if line.startswith("processor"):
            count += 1
        elif line.startswith(("model name", "Model", "Hardware")) and not model:
            model = line.partition(":")[2].strip()
    result = {"model": model, "logical_cpus": count or None}
    return {k: v for k, v in result.items() if v}


# ------------------------------------------------------------- memory
def parse_meminfo(text: str) -> dict:
    kb = {}
    for line in text.splitlines():
        m = re.match(r"(\w+):\s+(\d+)\s*kB", line)
        if m:
            kb[m.group(1)] = int(m.group(2))
    gb = lambda k: round(kb[k] / 1024 / 1024, 2) if k in kb else None
    out = {
        "total_gb": gb("MemTotal"),
        "available_gb": gb("MemAvailable"),
        "swap_total_gb": gb("SwapTotal"),
    }
    if "MemTotal" in kb and "MemAvailable" in kb:
        out["used_gb"] = round((kb["MemTotal"] - kb["MemAvailable"]) / 1024 / 1024, 2)
    return {k: v for k, v in out.items() if v is not None}


def _dmi_blocks(text: str):
    for chunk in re.split(r"\n\s*\n", text):
        lines = [ln for ln in chunk.splitlines() if ln.strip()]
        lines = [ln for ln in lines if not ln.startswith(("Handle", "#"))]
        if not lines or ":" in lines[0]:
            continue
        fields = {}
        for ln in lines[1:]:
            if ":" in ln:
                k, _, v = ln.partition(":")
                fields[k.strip()] = v.strip()
        yield lines[0].strip(), fields


def parse_dmi_memory(text: str, include_empty: bool = False) -> dict:
    arrays, devices = [], []
    for title, fields in _dmi_blocks(text):
        if title == "Physical Memory Array":
            arrays.append(fields)
        elif title == "Memory Device":
            devices.append(fields)

    max_mb = 0
    for arr in arrays:
        if arr.get("Use", "System Memory") == "System Memory":
            max_mb += size_to_mb(arr.get("Maximum Capacity")) or 0

    modules, empty = [], []
    installed_mb = 0
    for dev in devices:
        mb = size_to_mb(dev.get("Size"))
        entry = {"slot": dev.get("Locator", "?")}
        if mb:
            installed_mb += mb
            entry.update(
                size_gb=mb_to_gb(mb),
                type=dev.get("Type"),
                speed=dev.get("Configured Memory Speed") or dev.get("Speed"),
                form_factor=dev.get("Form Factor"),
                manufacturer=dev.get("Manufacturer"),
            )
            modules.append({k: v for k, v in entry.items() if v not in (None, "Unknown")})
        else:
            empty.append(entry)

    total_slots = len(modules) + len(empty)
    soldered = any(
        (m.get("form_factor") or "").lower() in ("row of chips", "chip", "die")
        for m in modules
    )
    data = {
        "slots_total": total_slots,
        "slots_used": len(modules),
        "slots_free": len(empty),
        "installed_gb": mb_to_gb(installed_mb),
        "firmware_max_gb": mb_to_gb(max_mb) if max_mb else None,
        "modules": modules,
        "soldered_memory_suspected": soldered,
    }
    if max_mb:
        data["headroom_gb"] = max(0, mb_to_gb(max_mb - installed_mb))
    if include_empty:
        data["empty_slots"] = empty
    return data


# ------------------------------------------------------------ storage
def _truthy(value) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes")


def _collect_mounts(node: dict) -> list[str]:
    mounts = []
    if node.get("mountpoint"):
        mounts.append(node["mountpoint"])
    for child in node.get("children", []) or []:
        mounts.extend(_collect_mounts(child))
    return mounts


def classify_disk(name: str, transport: str | None, rotational: bool,
                  model: str | None = None) -> str:
    if "virtual" in (model or "").lower() or (transport or "").lower() == "virtio":
        return "virtual disk (the real drive type is not visible from inside a VM/WSL)"
    if name.startswith("nvme") or (transport or "").lower() == "nvme":
        return "NVMe SSD"
    if rotational:
        return "HDD (spinning disk)"
    return "SSD / non-rotational (not NVMe)"


def parse_lsblk(text: str) -> list[dict]:
    disks = []
    for dev in json.loads(text).get("blockdevices", []):
        name = dev.get("name", "")
        if dev.get("type") != "disk" or name.startswith(("zram", "ram", "loop")):
            continue
        transport = (dev.get("tran") or "") or None
        try:
            size_gb = round(int(dev.get("size", 0)) / 1e9)
        except (TypeError, ValueError):
            size_gb = None
        disks.append({
            "name": name,
            "model": (dev.get("model") or "").strip() or None,
            "size_gb": size_gb,
            "transport": transport,
            "kind": classify_disk(name, transport, _truthy(dev.get("rota")),
                                  dev.get("model")),
            "mountpoints": _collect_mounts(dev),
        })
    return disks


# ----------------------------------------------------------------- GPU
_GPU_CLASSES = ("VGA compatible", "3D controller", "Display controller")


def parse_lspci(text: str) -> list[dict]:
    gpus = []
    for line in text.splitlines():
        try:
            parts = shlex.split(line)
        except ValueError:
            continue
        if len(parts) >= 4 and any(parts[1].startswith(c) for c in _GPU_CLASSES):
            gpus.append({"pci_slot": parts[0], "class": parts[1],
                         "vendor": parts[2], "device": parts[3]})
    return gpus


def parse_nvidia_smi(text: str) -> list[dict]:
    rows = []
    for line in text.strip().splitlines():
        cells = [c.strip() for c in line.split(",")]
        if len(cells) == 3:
            rows.append({"name": cells[0], "vram": cells[1], "driver": cells[2]})
    return rows


# -------------------------------------------------------- system/battery
def parse_os_release(text: str) -> dict:
    out = {}
    for line in text.splitlines():
        if "=" in line and not line.startswith("#"):
            k, _, v = line.partition("=")
            out[k] = v.strip().strip('"')
    return {"os": out.get("PRETTY_NAME") or out.get("NAME")}


def parse_battery(values: dict[str, str]) -> dict:
    def as_int(key):
        try:
            return int(values[key])
        except (KeyError, ValueError):
            return None

    full = as_int("energy_full") or as_int("charge_full")
    design = as_int("energy_full_design") or as_int("charge_full_design")
    out = {
        "charge_percent": as_int("capacity"),
        "status": values.get("status"),
        "technology": values.get("technology"),
        "cycle_count": as_int("cycle_count"),
    }
    if full and design:
        health = round(100 * full / design)
        out["health_percent"] = health
        if health >= 80:
            verdict = "good"
        elif health >= 60:
            verdict = "fair, with noticeable wear"
        else:
            verdict = "worn; replacement may be worthwhile"
        out["health_assessment"] = f"{verdict} ({health}% of original design capacity)"
    return {k: v for k, v in out.items() if v is not None}
