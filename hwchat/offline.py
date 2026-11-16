"""Rule-based mode: no language model needed.

Maps keywords in the question to probes and renders fixed sentences from the
structured results. Less flexible than the LLM mode, but fully deterministic -
useful for demos, tests, and machines that cannot run a local model.
"""
from __future__ import annotations

import re

from . import probes

TOPICS: list[tuple[str, tuple[str, ...]]] = [
    ("memory_upgrade", ("upgrade", "add more", "more ram", "slot", "dimm", "expand",
                        "maximum", "max ram", "ddr")),
    ("memory_usage", ("free", "used", "usage", "available", "using", "swap")),
    ("memory_upgrade", ("ram", "memory")),
    ("processor", ("cpu", "processor", "core", "thread", "clock", "ghz")),
    ("storage", ("ssd", "nvme", "hdd", "disk", "drive", "storage", "sata")),
    ("graphics", ("gpu", "graphic", "video card", "nvidia", "amd", "radeon")),
    ("battery", ("battery", "charge", "charging")),
    ("system_overview", ("model", "bios", "kernel", "distro", "os", "operating",
                         "vendor", "manufacturer", "laptop")),
]
_TOPIC_PATTERNS: list[tuple[str, tuple[str, ...], re.Pattern]] = [
    (
        name,
        words,
        re.compile(r"\b(?:" + "|".join(re.escape(w) for w in words) + r")\b", re.I),
    )
    for name, words in TOPICS
]
FOLLOW_UP = re.compile(r"\b(and|also|more|details?|again|that|it)\b", re.I)


def pick_topics(question: str) -> list[str]:
    q = question.lower()
    found: list[str] = []
    has_usage_word = bool(_TOPIC_PATTERNS[1][2].search(q))
    for name, words, pattern in _TOPIC_PATTERNS:
        if pattern.search(q) and name not in found:
            if name == "memory_upgrade" and words[0] == "ram" and has_usage_word \
                    and "memory_usage" in found:
                continue
            found.append(name)
    asked_upgrade = bool(_TOPIC_PATTERNS[0][2].search(q))
    if re.search(r"\bhow much\b", q) and "memory_upgrade" in found and not asked_upgrade:
        found[found.index("memory_upgrade")] = "memory_usage"
    return found


def _gb(v):
    return f"{v:g} GB"


def _unavail(res):
    return f"I can't determine that: {res.get('reason', 'unknown reason')}."


def say_system(res):
    if res["status"] != "ok":
        return _unavail(res)
    d = res["data"]
    parts = []
    if d.get("vendor") or d.get("model"):
        parts.append(f"Machine: {' '.join(x for x in (d.get('vendor'), d.get('model')) if x)}.")
    if d.get("os"):
        parts.append(f"OS: {d['os']} (kernel {d.get('kernel')}).")
    if d.get("bios_version"):
        parts.append(f"BIOS: {d['bios_version']}.")
    return " ".join(parts) or "No system details were readable."


def say_cpu(res):
    if res["status"] != "ok":
        return _unavail(res)
    d = res["data"]
    s = f"CPU: {d.get('model', 'unknown model')}."
    if "physical_cores" in d:
        s += f" {d['physical_cores']} physical cores"
        if "logical_cpus" in d:
            s += f", {d['logical_cpus']} logical CPUs"
        s += "."
    elif "logical_cpus" in d:
        s += f" {d['logical_cpus']} logical CPUs."
    if "max_mhz" in d:
        s += f" Max clock about {d['max_mhz'] / 1000:.1f} GHz."
    return s


def say_mem_usage(res):
    if res["status"] != "ok":
        return _unavail(res)
    d = res["data"]
    s = f"Total RAM {_gb(d.get('total_gb'))}"
    if "used_gb" in d:
        s += f", about {_gb(d['used_gb'])} in use, {_gb(d['available_gb'])} available"
    return s + "."


def say_mem_upgrade(res):
    if res["status"] != "ok":
        return _unavail(res)
    d = res["data"]
    if not d.get("slot_details_available"):
        s = f"The OS sees {_gb(d.get('os_visible_ram_gb'))} of RAM, "
        s += "but I can't see the slot layout or maximum capacity"
        note = (res.get("notes") or [""])[0]
        return s + (f". ({note})" if note else ".")
    s = (f"You have {_gb(d['installed_gb'])} installed across {d['slots_used']} of "
         f"{d['slots_total']} slots")
    kinds = {m.get("type") for m in d["modules"] if m.get("type")}
    if kinds:
        s += f" ({', '.join(sorted(kinds))})"
    s += "."
    if d.get("soldered_memory_suspected"):
        s += " Some memory appears to be soldered, so it may not be upgradable."
    if d["slots_free"] and d.get("firmware_max_gb"):
        s += (f" The firmware reports a {_gb(d['firmware_max_gb'])} maximum, so up to "
              f"{_gb(d['headroom_gb'])} more could be added in {d['slots_free']} free slot(s).")
    elif not d["slots_free"]:
        s += " There are no free slots; upgrading means replacing existing modules."
    return s + " Check the maker's spec sheet; firmware limits are not always accurate."


def say_storage(res):
    if res["status"] != "ok":
        return _unavail(res)
    return " ".join(
        f"{d['name']}: {d.get('model') or 'unknown model'}, {d.get('size_gb')} GB, {d['kind']}."
        for d in res["data"])


def say_gpu(res):
    if res["status"] != "ok":
        return _unavail(res)
    d = res["data"]
    lines = [f"{a['vendor']} {a['device']}" for a in d["adapters"]]
    s = "Graphics: " + "; ".join(lines) + "." if lines else ""
    for n in d.get("nvidia_details", []):
        s += f" NVIDIA {n['name']} with {n['vram']} VRAM (driver {n['driver']})."
    return s.strip()


def say_battery(res):
    if res["status"] != "ok":
        return _unavail(res)
    d = res["data"]
    s = f"Battery at {d.get('charge_percent', '?')}% ({d.get('status', 'unknown status')})."
    if "health_percent" in d:
        s += f" Capacity is {d['health_percent']}% of its original design."
    return s


RENDER = {"system_overview": say_system, "processor": say_cpu,
          "memory_usage": say_mem_usage, "memory_upgrade": say_mem_upgrade,
          "storage": say_storage, "graphics": say_gpu, "battery": say_battery}


class OfflineAssistant:
    def __init__(self, runner: probes.Runner | None = None):
        self.runner = runner or probes.Runner()
        self.last_topics: list[str] = []
        self.trace: list[str] = []

    def reset(self):
        self.runner.reset()
        self.last_topics = []

    def ask(self, question: str) -> str:
        self.trace = []
        topics = pick_topics(question)
        if not topics and self.last_topics and FOLLOW_UP.search(question):
            topics = self.last_topics
        if not topics:
            return ("I can't tell which hardware you mean. Ask about RAM, CPU, "
                    "storage, GPU, battery or the machine model.")
        self.last_topics = topics
        self.trace = list(topics)
        return "\n".join(RENDER[t](self.runner.call(t)) for t in topics)