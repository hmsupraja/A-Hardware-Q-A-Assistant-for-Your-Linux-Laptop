import json
import unittest

from hwchat import parsers, probes, sandbox
from hwchat.offline import OfflineAssistant, pick_topics

DMI = """# dmidecode 3.5
Getting SMBIOS data from sysfs.

Handle 0x0031, DMI type 16, 23 bytes
Physical Memory Array
\tLocation: System Board Or Motherboard
\tUse: System Memory
\tMaximum Capacity: 64 GB
\tNumber Of Devices: 2

Handle 0x0032, DMI type 17, 92 bytes
Memory Device
\tSize: 16 GB
\tForm Factor: SODIMM
\tLocator: DIMM A
\tType: DDR4
\tSpeed: 3200 MT/s
\tManufacturer: Acme

Handle 0x0033, DMI type 17, 92 bytes
Memory Device
\tSize: No Module Installed
\tForm Factor: SODIMM
\tLocator: DIMM B
"""

LSBLK = json.dumps({"blockdevices": [
    {"name": "nvme0n1", "size": "512110190592", "type": "disk", "mountpoint": None,
     "model": "Samsung SSD 970", "rota": False, "tran": "nvme",
     "children": [{"name": "nvme0n1p2", "type": "part", "mountpoint": "/"}]},
    {"name": "sda", "size": 1000204886016, "type": "disk", "model": "WD10", "rota": "1",
     "tran": "sata", "mountpoint": None},
    {"name": "loop0", "size": 1, "type": "loop"},
    {"name": "zram0", "size": 1, "type": "disk", "rota": "0"},
]})

LSPCI = ('00:02.0 "VGA compatible controller" "Intel Corporation" "UHD Graphics 620" -r07\n'
         '01:00.0 "3D controller" "NVIDIA Corporation" "GP107M [GeForce GTX 1050]" -ra1\n'
         '00:1f.3 "Audio device" "Intel Corporation" "Sunrise Point-LP HD Audio"\n')


class ParserTests(unittest.TestCase):
    def test_size_conversion(self):
        self.assertEqual(parsers.size_to_mb("16 GB"), 16384)
        self.assertEqual(parsers.size_to_mb("8192 MB"), 8192)
        self.assertIsNone(parsers.size_to_mb("No Module Installed"))

    def test_dmi_memory(self):
        d = parsers.parse_dmi_memory(DMI, include_empty=True)
        self.assertEqual((d["slots_total"], d["slots_used"], d["slots_free"]), (2, 1, 1))
        self.assertEqual(d["installed_gb"], 16.0)
        self.assertEqual(d["firmware_max_gb"], 64.0)
        self.assertEqual(d["headroom_gb"], 48.0)
        self.assertEqual(d["empty_slots"], [{"slot": "DIMM B"}])
        self.assertFalse(d["soldered_memory_suspected"])

    def test_dmi_hides_empty_by_default(self):
        self.assertNotIn("empty_slots", parsers.parse_dmi_memory(DMI))

    def test_lsblk_nvme_detection(self):
        disks = {d["name"]: d for d in parsers.parse_lsblk(LSBLK)}
        self.assertEqual(set(disks), {"nvme0n1", "sda"})  # loop/zram skipped
        self.assertEqual(disks["nvme0n1"]["kind"], "NVMe SSD")
        self.assertEqual(disks["nvme0n1"]["mountpoints"], ["/"])
        self.assertEqual(disks["sda"]["kind"], "HDD (spinning disk)")

    def test_virtual_disk_not_called_hdd(self):
        t = json.dumps({"blockdevices": [{"name": "sda", "size": "1000000000", "type": "disk",
                                          "model": "Virtual Disk", "rota": "1"}]})
        self.assertIn("virtual", parsers.parse_lsblk(t)[0]["kind"])

    def test_lspci_only_gpus(self):
        gpus = parsers.parse_lspci(LSPCI)
        self.assertEqual(len(gpus), 2)
        self.assertEqual(gpus[1]["vendor"], "NVIDIA Corporation")

    def test_meminfo(self):
        d = parsers.parse_meminfo("MemTotal: 16777216 kB\nMemAvailable: 8388608 kB\n")
        self.assertEqual((d["total_gb"], d["used_gb"]), (16.0, 8.0))

    def test_battery_health(self):
        d = parsers.parse_battery({"capacity": "80", "energy_full": "40", "energy_full_design": "50"})
        self.assertEqual(d["health_percent"], 80)


class SafetyTests(unittest.TestCase):
    def test_unknown_command_refused(self):
        with self.assertRaises(sandbox.SandboxError):
            sandbox.run("rm -rf /")

    def test_file_allowlist(self):
        for bad in ("/etc/shadow", "/home/user/.ssh/id_rsa", "/sys/class/dmi/id/product_serial"):
            with self.assertRaises(sandbox.SandboxError):
                sandbox.read_file(bad)

    def test_no_shell_metacharacters_in_registry(self):
        for cmd in sandbox.REGISTRY.values():
            for arg in cmd.argv:
                self.assertFalse(any(c in arg for c in ";|&$`"), arg)

    def test_unknown_tool_and_bad_args(self):
        r = probes.Runner()
        self.assertEqual(r.call("shell", {"cmd": "ls"})["status"], "unavailable")
        self.assertEqual(r.call("storage", {"device": "sda; rm -rf /"})["status"], "unavailable")
        self.assertEqual(r.call("storage", {"bogus": 1})["status"], "unavailable")
        self.assertEqual(r.call("memory_upgrade", {"include_empty_slots": "yes"})["status"], "unavailable")

    def test_boolean_sent_as_text_is_accepted(self):
        r = probes.Runner()
        for v in ("false", "True", False):
            self.assertEqual(r.call("memory_upgrade", {"include_empty_slots": v})["status"], "ok")

    def test_virtual_disk_result_carries_caveat(self):
        from unittest import mock
        t = json.dumps({"blockdevices": [{"name": "sda", "size": "1000000000", "type": "disk",
                                          "model": "Virtual Disk", "rota": "1"}]})
        with mock.patch("hwchat.sandbox.run", return_value=t):
            res = probes.storage()
        self.assertTrue(any("cannot be determined" in n for n in res["notes"]))

    def test_tool_arguments_may_arrive_as_json_string(self):
        r = probes.Runner()
        self.assertEqual(r.call("memory_usage", "{}")["status"], "ok")


class OfflineTests(unittest.TestCase):
    def test_routing(self):
        self.assertEqual(pick_topics("Can I add more RAM?"), ["memory_upgrade"])
        self.assertEqual(pick_topics("how much RAM is free"), ["memory_usage"])
        self.assertEqual(pick_topics("how much ram do I have"), ["memory_usage"])
        self.assertEqual(pick_topics("Is my SSD NVMe?"), ["storage"])
        self.assertEqual(pick_topics("and what about the GPU?"), ["graphics"])
        self.assertEqual(pick_topics("tell me a joke"), [])

    def test_unknown_question_is_honest(self):
        self.assertIn("can't tell", OfflineAssistant().ask("what's the weather"))

    def test_follow_up_reuses_topic(self):
        bot = OfflineAssistant()
        bot.ask("how much ram do I have")
        self.assertEqual(bot.last_topics, ["memory_usage"])
        self.assertTrue(bot.ask("and more details"))


class FakeClient:
    """Stands in for the LLM: returns scripted replies and records nothing else."""
    def __init__(self, *replies):
        self.replies = list(replies)

    def chat(self, messages, tools):
        return self.replies.pop(0)


def _call(name, **args):
    return {"role": "assistant", "content": "",
            "tool_calls": [{"function": {"name": name, "arguments": args}}]}


class AgentGuardTests(unittest.TestCase):
    def test_guard_runs_tool_when_model_skips_it(self):
        from hwchat.agent import Assistant
        bot = Assistant(FakeClient({"role": "assistant", "content": "blah about RAM"},
                                   {"role": "assistant", "content": "final"}))
        self.assertEqual(bot.ask("and what about the GPU?"), "final")
        self.assertEqual(bot.trace, ["graphics() [auto]"])

    def test_no_guard_for_non_hardware_question(self):
        from hwchat.agent import Assistant
        bot = Assistant(FakeClient({"role": "assistant", "content": "out of scope"}))
        self.assertEqual(bot.ask("what is the weather today"), "out of scope")
        self.assertEqual(bot.trace, [])

    def test_text_json_tool_call_is_executed(self):
        from hwchat.agent import Assistant, tool_call_from_text
        self.assertEqual(tool_call_from_text('Sure: {"name": "battery", "parameters": {}}'), "battery")
        self.assertIsNone(tool_call_from_text('{"name": "rm", "parameters": {}}'))

    def test_invented_text_argument_is_dropped(self):
        from hwchat.agent import Assistant
        bot = Assistant(FakeClient(_call("storage", device="sda"),
                                   {"role": "assistant", "content": "ok"}))
        bot.ask("is my ssd nvme")
        self.assertEqual(bot.trace, ["storage()"])

    def test_leaked_tool_json_becomes_refusal(self):
        from hwchat.agent import Assistant, OUT_OF_SCOPE, looks_like_tool_json
        self.assertTrue(looks_like_tool_json('{"name": null, "parameters": {}}'))
        self.assertFalse(looks_like_tool_json("You have 8 GB of RAM."))
        bot = Assistant(FakeClient({"role": "assistant",
                                    "content": '{"name": null, "parameters": {}}'}))
        self.assertEqual(bot.ask("what is the weather today"), OUT_OF_SCOPE)

    def test_named_device_argument_is_kept(self):
        from hwchat.agent import ground_args
        self.assertEqual(ground_args("storage", {"device": "sda"}, "tell me about sda"),
                         {"device": "sda"})


class VerdictTests(unittest.TestCase):
    def test_battery_assessment_text(self):
        d = parsers.parse_battery({"capacity": "28", "energy_full": "50", "energy_full_design": "50"})
        self.assertIn("good", d["health_assessment"])
        w = parsers.parse_battery({"energy_full": "20", "energy_full_design": "50"})
        self.assertIn("worn", w["health_assessment"])


if __name__ == "__main__":
    unittest.main()
