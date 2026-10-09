"""Chunker self-check for unlimited Translate-tab text (no models needed)."""
import ast
import re
from pathlib import Path

src = Path(__file__).resolve().parent.parent.joinpath("app", "main.py").read_text(encoding="utf-8")
ns = {"re": re}
for node in ast.parse(src).body:  # pull just _CJK/_cost/_split_chunks: importing main loads models
    if getattr(node, "name", None) in ("_cost", "_split_chunks") or (
            isinstance(node, ast.Assign) and node.targets[0].id == "_CJK"):
        exec(compile(ast.Module([node], []), "main.py", "exec"), ns)
split, cost = ns["_split_chunks"], ns["_cost"]

en = " ".join(f"Sentence number {i} talks about pi = 3.14 and more." for i in range(200))
zh = "大家早上好，今天我们讨论语音识别。" * 120
runon = "word " * 2000
for text in (en, zh, runon, en + " " + zh):
    ch = split(text, 240)
    assert all(cost(c) <= 240 for c in ch), max(map(cost, ch))
    assert re.sub(r"\s", "", "".join(ch)) == re.sub(r"\s", "", text)  # nothing lost
assert "3.14" in split(en, 240)[0]
assert split("", 240) == [] and split("hi.", 240) == ["hi."]
print("ok")
