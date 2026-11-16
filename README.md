# hwchat - local Linux hardware Q&A assistant

Ask plain-English questions about the laptop you are sitting at
("Can I add more RAM?", "Is my SSD NVMe?", "and what about the GPU?").
The assistant never guesses: a language model *chooses* from a fixed set of
read-only tools, the program runs them, and the answer is built only from
the structured results.

## Quick start
```bash
python3 -m hwchat                    # auto: LLM if Ollama is running, else offline
python3 -m hwchat --mode offline     # deterministic, no model needed
python3 -m hwchat --list-tools
python3 -m hwchat --probe storage    # raw JSON from one tool (debugging)
python3 -m unittest discover -s tests
```
Local LLM mode (nothing leaves the machine):
```bash
ollama pull qwen2.5:7b && python3 -m hwchat --mode llm --model qwen2.5:7b
```
Optional, for slot count and max RAM (needs root):
```bash
sudo visudo -f /etc/sudoers.d/hwchat   # paste the line from deploy/hwchat.sudoers
sudo apt install dmidecode pciutils    # or your distro's equivalent
```

## Architecture
```
question -> Assistant (LLM picks a tool) -> Runner (validate args, cache)
         -> probes (collectors) -> sandbox (the ONLY OS access) -> parsers (pure)
         -> structured JSON {status, data, notes} -> LLM phrases the answer
```
| Module | Responsibility |
|---|---|
| `sandbox.py` | Registry of fixed commands, `shell=False`, scrubbed env, timeouts; allow-listed file reads |
| `parsers.py` | Pure text-to-dict functions, unit-tested with sample output |
| `probes.py` | Tool catalogue, argument validation, session cache, JSON schemas for the LLM |
| `agent.py` | Tool-calling loop against Ollama (stdlib HTTP), conversation memory |
| `offline.py` | Keyword router + templated answers (no LLM) |

## Design decisions
**Safety.** The model only supplies a tool *name* and validated arguments, never a
command string. Commands live in a closed registry with fixed argv, run without a
shell. File reads are limited to a short prefix list and serial numbers/UUIDs are
blocked. Unknown tools or arguments return an error result instead of executing.
Tool output is treated as data in the system prompt to blunt prompt injection.

**Permissions.** Only `dmidecode -t memory` needs root. The program runs as the normal
user and calls `sudo -n` (never prompts). The sudoers rule pins the exact arguments,
so it cannot be abused for other dmidecode options. If the rule is missing, the tool
degrades to `/proc/meminfo` and reports that slot data is unavailable.

**Accuracy.** Every result has an explicit `status`. Failure is `"unavailable"` plus a
reason, so "I don't know" is a data value the model repeats. Firmware-reported maximum
RAM is flagged as possibly inaccurate; soldered memory is detected from the form factor.
NVMe is decided from the device name / transport, not the marketing model string.

**Portability.** Missing binaries are detected at call time (`lscpu` falls back to
`/proc/cpuinfo`; GPU info works with `lspci` and/or `nvidia-smi`). `LC_ALL=C` keeps
output parseable on any locale. Installed utilities are listed in the system prompt.

**Follow-ups.** Conversation history is kept (trimmed), and tool results are cached
for the session (volatile ones like free memory expire in seconds), so
"and the GPU?" reuses context and fetches only what is missing. `/reset` clears both.

**Privacy.** All inspection is local. The LLM backend is a local Ollama server, so no
hardware data leaves the laptop. There is no cloud code path.

## Limitations / future work
Virtual machines report odd disk and DMI data; AMD GPU VRAM is not read; only the first
battery is reported; a cloud-LLM backend could be added behind an explicit opt-in flag.
