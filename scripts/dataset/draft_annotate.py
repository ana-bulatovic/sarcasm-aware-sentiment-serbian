#!/usr/bin/env python3
"""Draft anotacija sentiment + sarkazam (za ručnu korekciju).

Heuristika na leksikonima (``POSITIVE``, ``NEGATIVE``, ``SARCASM_CUES``),
opciono Ollama batch. Upisuje labele u annotation CSV i dataset CSV —
nije konačna anotacija.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts._bootstrap import ensure_project_root

ensure_project_root()

import pandas as pd

from src.common.config import load_config, resolve_path
from src.common.io_utils import save_csv
from src.common.schema import FINAL_COLUMNS
from src.common.stdio_utf8 import configure_utf8_stdio
from src.preprocessing.clean import normalize_whitespace

# Leksikon pozitivnih fraza (substring match nakon casefold).
POSITIVE = {
    "odličan", "odlican", "odlična", "odlicna", "odlično", "odlicno",
    "super", "bravo", "svaka čast", "svaka cast", "prelep", "prelepa",
    "prelepo", "divan", "divna", "divno", "sjajan", "sjajna", "sjajno",
    "fantastičan", "fantasticno", "fantastično", "odlični", "volim",
    "sviđa", "svidja", "preporučujem", "preporucujem", "majstorski",
    "genijalno", "genijalan", "top", "legenda", "legendaran", "najbolji",
    "najbolja", "najbolje", "lep", "lepa", "lepo", "kul", "hvala",
    "respect", "poštovanje", "postovanje", "remek delo", "remek-delo",
    "dopada", "zadovoljan", "zadovoljna", "obožavam", "obozavam",
}

# Leksikon negativnih fraza (substring match nakon casefold).
NEGATIVE = {
    "loš", "los", "loša", "losa", "loše", "lose", "sranje", "govno",
    "katastrofa", "užas", "uzas", "odvratno", "odvratan", "glup", "glupa",
    "glupo", "idiot", "kreten", "mrš", "mrs", "mrzi", "ne sviđa", "ne svidja",
    "dosadan", "dosadna", "dosadno", "slab", "slaba", "slabo", "propast",
    "bezveze", "bez veze", "najgori", "najgora", "najgore", "sramota",
    "patetika", "patetično", "pateticno", "jadno", "jadan", "grozno",
    "grozan", "prevara", "laž", "laz", "ne valja", "nevalja", "propao",
    "propala", "đubre", "djubre", "k*rac", "kurac", "jebem", "jebiga",
    "bolan", "boli me", "krade", "lopov", "blamaža", "blamaza",
}

# Regex signali sarkazma / ironije (navodnici, „baš“, „naravno“, …).
SARCASM_CUES = [
    r"\bbaš\b", r"\bbas\b", r"\bnaravno\b", r"\bkao da\b", r"\bkako da ne\b",
    r"\bvaljda\b", r"\bzeza", r"\bsarkaz", r"\bironi", r"„[^”]{2,}”",
    r"\"[^\"]{2,}\"", r"\bah da\b", r"\bjoš nam treba\b", r"\bjos nam treba\b",
]

# Kratki timestamp redovi (npr. video markeri) → neutralno.
TIMESTAMP_RE = re.compile(r"^\s*\d{1,2}:\d{2}(?::\d{2})?\b")


def _fold(text: str) -> str:
    """Normalizuj beline i casefold radi poređenja sa leksikonima."""
    return normalize_whitespace(text or "").casefold()


def _has_any(text: str, phrases: set[str]) -> bool:
    """True ako bilo koja fraza iz skupa postoji kao substring u tekstu."""
    return any(p in text for p in phrases)


def draft_annotate(text: str) -> tuple[str, str]:
    """Heuristička draft labela (sentiment, sarcasm) za jedan komentar.

    Koristi ``POSITIVE`` / ``NEGATIVE`` i ``SARCASM_CUES``. Prazan/kratak
    tekst i timestamp → ``(\"0\", \"0\")``. Sarkazam ima prioritet nad
    čistim sentimentom; pitanja bez stava ostaju neutralna.

    Returns:
        Par stringova: sentiment u {1, 0, -1}, sarcasm u {1, 0}.
    """
    raw = normalize_whitespace(text or "")
    folded = _fold(raw)
    if not raw or len(raw) < 8:
        return "0", "0"
    if TIMESTAMP_RE.match(raw) and len(raw) < 40:
        return "0", "0"

    pos = _has_any(folded, POSITIVE)
    neg = _has_any(folded, NEGATIVE)
    sarc = sum(1 for pat in SARCASM_CUES if re.search(pat, folded, flags=re.I))
    # Pohvala + negacija / elipsa → jači signal sarkazma
    if re.search(r"\b(bravo|genijalno|odlično|odlicno|super|svaka čast|svaka cast)\b", folded):
        if neg or "..." in raw or "…" in raw:
            sarc += 2

    is_sarcastic = False
    if sarc >= 2:
        is_sarcastic = True
    elif sarc >= 1 and pos and neg:
        is_sarcastic = True
    elif re.search(
        r"\b(bravo|genijalno|odlično|odlicno|super)\b.{0,40}\b("
        r"sranje|katastrofa|dosad|loš|los|uzas|užas|glup|bezveze|propast)\b",
        folded,
    ):
        is_sarcastic = True

    if is_sarcastic:
        if neg or pos:
            return "-1", "1"
        return "0", "1"

    if pos and neg:
        p_hits = sum(1 for p in POSITIVE if p in folded)
        n_hits = sum(1 for n in NEGATIVE if n in folded)
        if n_hits > p_hits:
            return "-1", "0"
        if p_hits > n_hits:
            return "1", "0"
        return "0", "0"
    if pos:
        return "1", "0"
    if neg:
        return "-1", "0"

    questionish = raw.strip().endswith("?") or folded.startswith(
        ("zašto", "zasto", "kako", "da li", "jel ", "ko ", "šta ", "sta ")
    )
    if questionish:
        return "0", "0"
    return "0", "0"


def _ollama_annotate_batch(items: list[dict[str, str]], model: str) -> dict[str, tuple[str, str]]:
    """Pošalji batch komentara Ollama modelu; parsira JSONL odgovor.

    Args:
        items: Lista ``{\"id\", \"text\"}`` (tekst se seče na 400 znakova).
        model: Ime lokalnog Ollama modela (npr. ``llama2``).

    Returns:
        Mapiranje id → (sentiment, sarcasm). Prazan dict pri grešci / timeoutu.
    """
    payload = "\n".join(f"{it['id']}: {it['text'][:400]}" for it in items)
    prompt = f"""Ti anotiras srpske komentare za master rad.
Za SVAKI red vrati TACNO jedan JSON objekat po liniji (JSONL), polja:
id, sentiment, sarcasm
sentiment: 1|0|-1
sarcasm: 1|0

Pravila:
- sentiment = preneseni/namerni stav, ne samo povrsinske reci
- sarkasticni kompliment (npr. "bravo" uz kritiku) => sentiment=-1, sarcasm=1
- sarcasm=yes SAMO ako je jasan podsmeh; ako nisi siguran => 0
- neutral za cinjenice, timestamp, pitanja bez stava
- bez objasnjenja, samo JSONL

Komentari:
{payload}
"""
    try:
        proc = subprocess.run(
            ["ollama", "run", model],
            input=prompt,
            text=True,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=180,
        )
    except Exception:
        return {}
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    result: dict[str, tuple[str, str]] = {}
    for line in out.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            m = re.search(r"\{.*\}", line)
            if not m:
                continue
            line = m.group(0)
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        rid = str(obj.get("id", "")).strip()
        sent = str(obj.get("sentiment", "")).strip().lower()
        sarc = str(obj.get("sarcasm", "")).strip().lower()
        if rid and sent in {"1", "0", "-1"} and sarc in {"1", "0"}:
            result[rid] = (sent, sarc)
    return result


def main() -> None:
    """CLI: draft anotacija annotation CSV-a (leksikon + opciono Ollama).

    Pravi backup, popunjava sentiment/sarcasm preko ``draft_annotate``,
    opciono prepisuje labele iz Ollama batcheva, čuva annotation i dataset CSV.
    """
    configure_utf8_stdio()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--ollama-model",
        default="",
        help="Ako se navede (npr. llama2), pokusaj Ollama draft preko batcheva",
    )
    parser.add_argument("--batch-size", type=int, default=12)
    args = parser.parse_args()

    config = load_config(args.config)
    path = resolve_path(config["paths"]["annotation_csv"])
    dataset_path = resolve_path(config["paths"]["dataset_csv"])

    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str).fillna("")
    backup = path.with_name("annotation_template_before_draft.csv")
    df.to_csv(backup, index=False, encoding="utf-8-sig")
    print(f"[draft] Backup: {backup}")

    labels: dict[str, tuple[str, str]] = {}
    for _, row in df.iterrows():
        labels[str(row["id"])] = draft_annotate(str(row["text"]))

    if args.ollama_model:
        print(f"[draft] Ollama model: {args.ollama_model}")
        rows = df.to_dict(orient="records")
        for i in range(0, len(rows), args.batch_size):
            batch = [
                {"id": str(r["id"]), "text": str(r["text"])}
                for r in rows[i : i + args.batch_size]
            ]
            print(f"[draft] Ollama batch {i // args.batch_size + 1} ({len(batch)})")
            got = _ollama_annotate_batch(batch, args.ollama_model)
            labels.update(got)
            print(f"  parsed {len(got)}/{len(batch)}")

    sentiments = []
    sarcasms = []
    for _, row in df.iterrows():
        s, c = labels[str(row["id"])]
        sentiments.append(s)
        sarcasms.append(c)
    df["sentiment"] = sentiments
    df["sarcasm"] = sarcasms

    out_rows = df[FINAL_COLUMNS].to_dict(orient="records")
    save_csv(out_rows, path, columns=FINAL_COLUMNS)
    save_csv(out_rows, dataset_path, columns=FINAL_COLUMNS)

    print("[draft] sentiment:", df["sentiment"].value_counts().to_dict())
    print("[draft] sarcasm:", df["sarcasm"].value_counts().to_dict())
    print(f"[draft] Saved: {path}")
    print("DRAFT za tvoju korekciju — posebno proveri sarcasm=yes.")


if __name__ == "__main__":
    main()
