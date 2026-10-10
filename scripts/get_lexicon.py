"""ASR spelling lexicon from Tatoeba sentences (CC BY 2.0 FR): English + Vietnamese word counts.

  python scripts/get_lexicon.py      # -> models/lexicon/words.txt  (config.yaml › asr_lexicon)

Streams each language's sentence export once (no archive kept on disk), counts lowercase
words, keeps those seen >= MIN_COUNT times (drops Tatoeba typos). Chinese is skipped: every
ASR output there is a real character, so a spelling lexicon has nothing to snap.
"""
import bz2
import io
import re
import urllib.request
from collections import Counter
from pathlib import Path

DST = Path(__file__).resolve().parent.parent / "models" / "lexicon" / "words.txt"
URL = "https://downloads.tatoeba.org/exports/per_language/{0}/{0}_sentences.tsv.bz2"
LANGS = ["eng", "vie"]
MIN_COUNT = 3
WORD = re.compile(r"[^\W\d_]+")

counts = Counter()
for lang in LANGS:
    print(f"{lang}: streaming {URL.format(lang)} ...")
    with urllib.request.urlopen(URL.format(lang)) as r, \
            io.TextIOWrapper(bz2.open(r), encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            counts.update(WORD.findall(line.rsplit("\t", 1)[-1].lower()))
    print(f"{lang}: {n} sentences")
DST.parent.mkdir(parents=True, exist_ok=True)
words = [(w, c) for w, c in counts.most_common() if c >= MIN_COUNT]
DST.write_text("".join(f"{w} {c}\n" for w, c in words), encoding="utf-8")
print(f"{len(words)} words -> {DST} ({DST.stat().st_size / 1e6:.1f} MB)")
