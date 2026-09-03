"""Radformatet i predictions.csv. Delas av analyze.py och evaluate.py."""

from __future__ import annotations

import csv
from pathlib import Path

#: Falt som analyze.py fyller i vid prediktionstillfallet.
PREDIKTIONSFALT = [
    "prediktion_id",
    "skapad_vid_utc",
    "ticker",
    "horisont",
    "kalla",
    "intervall",
    "senaste_bar_tid_utc",
    "latens_minuter",
    "pris_vid_prediktion",
    "riktning",
    "konfidens",
    "basrat",
    "basrat_for_riktning",
    "marknadslage",
    "indikatorer_json",
    "mognar_tidigast_utc",
]

#: Falt som evaluate.py fyller i nar prediktionen mognat. Tomma fram till dess.
UTFALLSFALT = [
    "utvarderad_vid_utc",
    "utfall_bar_tid_utc",
    "utfall_pris",
    "utfall_avkastning",
    "utfall_riktning",
    "traff",
    "index_avkastning",
]

FALT = PREDIKTIONSFALT + UTFALLSFALT


def skriv_rad(sokvag: Path, rad: dict) -> None:
    """Lagger till en rad. Skapar filen med rubrikrad om den inte finns."""
    sokvag = Path(sokvag)
    ny = not sokvag.exists() or sokvag.stat().st_size == 0
    sokvag.parent.mkdir(parents=True, exist_ok=True)
    with sokvag.open("a", newline="", encoding="utf-8") as fh:
        skrivare = csv.DictWriter(fh, fieldnames=FALT, extrasaction="raise")
        if ny:
            skrivare.writeheader()
        skrivare.writerow({f: rad.get(f, "") for f in FALT})
