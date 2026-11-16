from __future__ import annotations

import argparse
import json
import sys

from . import __version__, probes
from .agent import Assistant, LLMUnavailable, OllamaClient
from .offline import OfflineAssistant

HELP = "Commands: /help  /tools  /reset  /exit     Ask e.g. 'Is my SSD NVMe?'"


def build_assistant(args):
    if args.mode != "offline":
        client = OllamaClient(args.model, args.host)
        if client.ping():
            return Assistant(client), f"LLM mode (local model {args.model})"
        if args.mode == "llm":
            sys.exit(f"Ollama model '{args.model}' is not reachable at {args.host}. "
                     f"Start Ollama and run: ollama pull {args.model}")
    return OfflineAssistant(), "offline rule-based mode (no LLM reachable)"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="hwchat", description=__doc__ or "Hardware Q&A")
    ap.add_argument("--mode", choices=["auto", "llm", "offline"], default="auto")
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--list-tools", action="store_true")
    ap.add_argument("--probe", metavar="NAME", help="run one tool and print raw JSON")
    ap.add_argument("--version", action="version", version=__version__)
    args = ap.parse_args(argv)

    if args.list_tools:
        for p in probes.PROBES.values():
            print(f"{p.name:16} {p.description}")
        return 0
    if args.probe:
        print(json.dumps(probes.Runner().call(args.probe), indent=2))
        return 0

    bot, banner = build_assistant(args)
    print(f"hwchat {__version__} — {banner}\n{HELP}\n")
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        if line in ("/exit", "/quit"):
            return 0
        if line == "/help":
            print(HELP)
        elif line == "/tools":
            for p in probes.PROBES.values():
                print(f"  {p.name}: {p.description}")
        elif line == "/reset":
            bot.reset()
            print("Conversation and cached hardware data cleared.")
        else:
            try:
                answer = bot.ask(line)
            except LLMUnavailable as exc:
                answer = f"The language model is unreachable ({exc})."
            used = ", ".join(bot.trace)
            print(f"\nhwchat> {answer}" + (f"\n        [tools: {used}]" if used else "") + "\n")


if __name__ == "__main__":
    raise SystemExit(main())