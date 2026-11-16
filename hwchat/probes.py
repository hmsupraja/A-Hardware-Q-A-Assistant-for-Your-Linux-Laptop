"""The catalogue of questions the assistant can ask the machine.

Each probe has a name, a description and a parameter schema (these are shown to
the language model), plus a collector function. Collectors never raise: they
return either ``ok(...)`` or ``unavailable(...)`` so that "I don't know" is
always an explicit piece of data rather than something the model must infer.
"""
from __future__ import annotations

import json
import platform
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from . import parsers, sandbox
from .sandbox import SandboxError


def ok(data: Any, notes: list[str] | None = None, source: str | None = None) -> dict:
    out = {"status": "ok", "data": data}
    if notes:
        out["notes"] = notes
    if source:
        out["source"] = source
    return out


def unavailable(reason: str) -> dict:
    return {"status": "unavailable", "reason": reason}


# ------------------------------------------------------------ collectors
def system_overview() -> dict:
    data = {"kernel": platform.release(), "machine": platform.machine()}
    try:
        data.update(parsers.parse_os_release(sandbox.read_file("/etc/os-release")))
    except SandboxError:
        pass
    for label, fname in (("vendor", "sys_vendor"), ("model", "product_name"),
                         ("bios_version", "bios_version")):
        try:
            data[label] = sandbox.read_file(f"/sys/class/dmi/id/{fname}").strip()
        except SandboxError:
            pass
    notes = []
    if "model" not in data and "vendor" not in data:
        notes.append("The laptop vendor and model are not exposed on this system "
                     "(common inside a VM or WSL). Say you cannot tell the model.")
    return ok(data, notes=notes, source="/etc/os-release, /sys/class/dmi/id")


def processor() -> dict:
    try:
        return ok(parsers.parse_lscpu(sandbox.run("lscpu")), source="lscpu")
    except SandboxError as exc:
        try:
            info = parsers.parse_cpuinfo(sandbox.read_file("/proc/cpuinfo"))
        except SandboxError:
            return unavailable(f"lscpu failed ({exc}) and /proc/cpuinfo is unreadable")
        return ok(info, notes=[f"lscpu unavailable ({exc}); partial data from /proc/cpuinfo"],
                  source="/proc/cpuinfo")


def memory_usage() -> dict:
    try:
        return ok(parsers.parse_meminfo(sandbox.read_file("/proc/meminfo")),
                  source="/proc/meminfo")
    except SandboxError as exc:
        return unavailable(str(exc))


def memory_upgrade(include_empty_slots: bool = False) -> dict:
    try:
        total = parsers.parse_meminfo(sandbox.read_file("/proc/meminfo")).get("total_gb")
    except SandboxError:
        total = None
    try:
        slots = parsers.parse_dmi_memory(sandbox.run("dmi_memory"), include_empty_slots)
    except SandboxError as exc:
        if total is None:
            return unavailable(str(exc))
        return ok(
            {"os_visible_ram_gb": total, "slot_details_available": False},
            notes=[f"Slot count and maximum capacity could not be read: {exc}"],
            source="/proc/meminfo",
        )
    if not slots["slots_total"]:
        return ok({"os_visible_ram_gb": total, "slot_details_available": False},
                  notes=["Firmware exposed no memory-slot information."],
                  source="dmidecode")
    notes = ["The maximum capacity is reported by firmware and is sometimes "
             "inaccurate on laptops; confirm against the manufacturer's spec sheet."]
    if slots["soldered_memory_suspected"]:
        notes.append("Some memory looks soldered to the board and cannot be upgraded.")
    slots.update(os_visible_ram_gb=total, slot_details_available=True)
    return ok(slots, notes=notes, source="dmidecode -t memory")


def storage(device: str | None = None) -> dict:
    try:
        disks = parsers.parse_lsblk(sandbox.run("lsblk"))
        source = "lsblk"
    except (SandboxError, ValueError) as exc:
        return unavailable(f"could not list disks: {exc}")
    if device:
        disks = [d for d in disks if d["name"] == device]
        if not disks:
            return unavailable(f"no disk named '{device}' was found")
    if not disks:
        return unavailable("no physical disks were detected")
    notes = []
    if any(d["kind"].startswith("virtual") for d in disks):
        notes.append("Virtual disk: whether the underlying physical drive is NVMe, "
                     "SSD or HDD cannot be determined from inside this VM/WSL. "
                     "Say you cannot tell; do not conclude it is not NVMe.")
    return ok(disks, notes=notes, source=source)


def graphics() -> dict:
    gpus, notes, source = [], [], []
    try:
        gpus = parsers.parse_lspci(sandbox.run("lspci"))
        source.append("lspci")
    except SandboxError as exc:
        notes.append(f"lspci unavailable: {exc}")
    nvidia = []
    if sandbox.is_installed("nvidia"):
        try:
            nvidia = parsers.parse_nvidia_smi(sandbox.run("nvidia"))
            source.append("nvidia-smi")
        except SandboxError as exc:
            notes.append(f"nvidia-smi present but failed: {exc}")
    if not gpus and not nvidia:
        return unavailable("; ".join(notes) or "no graphics adapter was detected")
    data = {"adapters": gpus}
    if nvidia:
        data["nvidia_details"] = nvidia
    return ok(data, notes=notes, source=" + ".join(source))


def battery() -> dict:
    names = [n for n in sandbox.list_dir("/sys/class/power_supply") if n.startswith("BAT")]
    if not names:
        return unavailable("no battery was found (desktop or battery not exposed)")
    base = f"/sys/class/power_supply/{names[0]}/"
    values = {}
    for key in ("capacity", "status", "technology", "cycle_count", "energy_full",
                "energy_full_design", "charge_full", "charge_full_design"):
        try:
            values[key] = sandbox.read_file(base + key).strip()
        except SandboxError:
            pass
    return ok(parsers.parse_battery(values), source=base)


# --------------------------------------------------------------- registry
@dataclass
class Probe:
    name: str
    description: str
    func: Callable[..., dict]
    params: dict = field(default_factory=dict)  # name -> JSON-schema fragment
    ttl: float | None = None  # seconds to cache; None = whole session


PROBES: dict[str, Probe] = {p.name: p for p in [
    Probe("system_overview", "Laptop vendor/model, BIOS version, OS name and kernel.",
          system_overview),
    Probe("processor", "CPU model, core and thread counts, clock speed, cache.",
          processor),
    Probe("memory_usage", "Total, used and available RAM right now, plus swap.",
          memory_usage, ttl=5),
    Probe("memory_upgrade",
          "Physical RAM layout: modules installed, slots used/free, RAM type and "
          "speed, firmware-reported maximum capacity. Use for upgrade questions.",
          memory_upgrade,
          params={"include_empty_slots": {"type": "boolean",
                  "description": "Also list the empty slots."}}),
    Probe("storage", "Disks: size, model, and whether each is NVMe, SATA SSD or HDD.",
          storage, ttl=60,
          params={"device": {"type": "string", "pattern": r"^[a-z0-9]{1,16}$",
                  "description": "Optional disk name such as nvme0n1 or sda."}}),
    Probe("graphics", "Graphics adapters (integrated and discrete) and NVIDIA VRAM.",
          graphics),
    Probe("battery", "Battery charge, status and wear (health) if present.",
          battery, ttl=10),
]}


def tool_schemas() -> list[dict]:
    """Function-calling schemas in the format local LLM servers expect."""
    return [{
        "type": "function",
        "function": {
            "name": p.name,
            "description": p.description,
            "parameters": {"type": "object", "properties": p.params, "required": []},
        },
    } for p in PROBES.values()]


def validate_args(probe: Probe, args: Any) -> dict:
    if args in (None, ""):
        return {}
    if not isinstance(args, dict):
        raise ValueError("arguments must be an object")
    clean = {}
    for key, value in args.items():
        spec = probe.params.get(key)
        if spec is None:
            raise ValueError(f"unknown argument '{key}'")
        kind = spec["type"]
        if kind == "boolean":
            # Small local models often send "true"/"false" as text; accept that.
            if isinstance(value, str) and value.strip().lower() in ("true", "false"):
                value = value.strip().lower() == "true"
            if not isinstance(value, bool):
                raise ValueError(f"'{key}' must be true or false")
        if kind == "string":
            if not isinstance(value, str) or not re.fullmatch(spec.get("pattern", ".*"), value):
                raise ValueError(f"'{key}' has an invalid value")
        clean[key] = value
    return clean


class Runner:
    """Validates a request, executes the probe, caches results for the session."""

    def __init__(self) -> None:
        self._cache: dict[tuple, tuple[float, dict]] = {}

    def call(self, name: str, args: Any = None) -> dict:
        probe = PROBES.get(name)
        if probe is None:
            return unavailable(f"'{name}' is not an available tool")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                return unavailable("tool arguments were not valid JSON")
        try:
            clean = validate_args(probe, args)
        except ValueError as exc:
            return unavailable(
                f"invalid arguments ({exc}). Call the tool again with no arguments.")

        key = (name, json.dumps(clean, sort_keys=True))
        hit = self._cache.get(key)
        if hit and (probe.ttl is None or time.monotonic() - hit[0] < probe.ttl):
            return hit[1]
        try:
            result = probe.func(**clean)
        except Exception as exc:  # a collector bug must never crash the chat
            return unavailable(f"internal error while reading hardware: {exc}")
        if result.get("status") == "ok":  # don't cache failures; permissions may change
            self._cache[key] = (time.monotonic(), result)
        return result

    def reset(self) -> None:
        self._cache.clear()
