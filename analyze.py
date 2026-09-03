"""Kor en analys for en ticker och en horisont, och loggar prediktionen.

Ordningen ar inte forhandlingsbar:
    1. hamta data
    2. farskhetskontroll -- misslyckas den avbryts allt, ingen analys skrivs
    3. berakna indikatorer
    4. berakna basrat
    5. skriv outputblocket
    6. lagg till en rad i predictions.csv

Anvandning:
    python analyze.py --ticker NVDA --horisont 5
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import uuid
from datetime import datetime, timedelta, timezone

import pandas as pd

import indicators as ind
from config import CONFIG, PREDIKTIONSFIL
from data import DataFel, farskhetskontroll, hamta, hamta_dagsdata
from predictions_schema import skriv_rad

#: Konfidensen far aldrig hamna utanfor det spann evaluate.py rapporterar pa.
KONFIDENS_MIN = min(lo for lo, _ in CONFIG["KONFIDENSINTERVALL"]) / 100.0
KONFIDENS_MAX = max(hi for _, hi in CONFIG["KONFIDENSINTERVALL"]) / 100.0


class AnalysAvbruten(Exception):
    """Kastas nar analysen inte far fortsatta. Avslutar med felkod, skriver ingen rad."""


# --------------------------------------------------------------------------
# Bedomning
# --------------------------------------------------------------------------
#
# PROVISORISK -- ska ersattas med reglerna i CLAUDE.md.
#
# CLAUDE.md fanns inte i repot nar filen skrevs, sa hur indikatorer ska vagas
# ihop till en riktning och en konfidens ar ospecificerat. Funktionen nedan ar
# medvetet enkel och genomskinlig sa att den gar att byta ut rakt av:
#
#   - Utgangspunkten ar basraten, inte 50 procent. En modell som inte vet nagot
#     ska aterge driften, inte ett myntkast.
#   - Varje indikator bidrar med en liten additiv tilt i log-odds.
#   - Konfidensen klipps till [50, 85] procent. Allt over 85 ar sjalvbedrageri
#     pa den har typen av underlag.
#
# Vikterna ar inte anpassade mot nagon data. De ar gissningar, och prediktioner
# fran den har versionen bor betraktas som en nollhypotes tills evaluate.py
# sagt nagot annat pa minst 100 utfall.
# --------------------------------------------------------------------------

TILTVIKTER = {
    "rsi_14": 0.30,
    "macd_histogram": 0.25,
    "pris_mot_sma_50": 0.30,
    "pris_mot_sma_200": 0.20,
    "bollinger_procent_b": 0.20,
    "adx_14": 0.15,
    "volymratio_20": 0.10,
}


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def bedomning(indikatorer: dict[str, float], basrat_upp: float) -> tuple[str, float, dict]:
    """Returnerar (riktning, konfidens, bidrag). PROVISORISK, se noten ovan."""
    bidrag: dict[str, float] = {}

    def tilt(namn: str, normaliserad: float) -> None:
        """`normaliserad` ligger ungefar i [-1, 1] dar positivt betyder uppgang."""
        if namn in TILTVIKTER:
            bidrag[namn] = TILTVIKTER[namn] * max(-1.0, min(1.0, normaliserad))

    if "rsi_14" in indikatorer:
        tilt("rsi_14", (indikatorer["rsi_14"] - 50.0) / 50.0)
    if "macd_histogram" in indikatorer and indikatorer.get("pris"):
        tilt("macd_histogram", indikatorer["macd_histogram"] / (0.01 * indikatorer["pris"]))
    if "pris_mot_sma_50" in indikatorer:
        tilt("pris_mot_sma_50", indikatorer["pris_mot_sma_50"] / 0.05)
    if "pris_mot_sma_200" in indikatorer:
        tilt("pris_mot_sma_200", indikatorer["pris_mot_sma_200"] / 0.10)
    if "bollinger_procent_b" in indikatorer:
        tilt("bollinger_procent_b", 2.0 * (indikatorer["bollinger_procent_b"] - 0.5))
    if {"plus_di_14", "minus_di_14", "adx_14"} <= indikatorer.keys():
        riktad = 1.0 if indikatorer["plus_di_14"] >= indikatorer["minus_di_14"] else -1.0
        tilt("adx_14", riktad * indikatorer["adx_14"] / 50.0)
    if "volymratio_20" in indikatorer and "momentum_1" in indikatorer:
        styrka = min(2.0, max(0.0, indikatorer["volymratio_20"] - 1.0))
        tilt("volymratio_20", styrka * (1.0 if indikatorer["momentum_1"] >= 0 else -1.0))

    p_upp = _sigmoid(_logit(basrat_upp) + sum(bidrag.values()))

    riktning = "UPP" if p_upp >= 0.5 else "NER"
    konfidens = p_upp if riktning == "UPP" else 1.0 - p_upp
    konfidens = min(KONFIDENS_MAX, max(KONFIDENS_MIN, konfidens))
    return riktning, konfidens, bidrag


# --------------------------------------------------------------------------
# Korning
# --------------------------------------------------------------------------


def _mognadstid(senaste_bar_tid: datetime, horisont: int) -> datetime:
    """Tidigaste tidpunkt da `horisont` handelsdagar sakert har passerat.

    Kalenderdagar = handelsdagar * 7/5, plus tva dagar marginal for helger och
    rodadagar. evaluate.py rustar sedan pa faktiska barer, inte pa den har
    uppskattningen -- den styr bara nar det ar mening att titta efter.
    """
    return senaste_bar_tid + timedelta(days=math.ceil(horisont * 7 / 5) + 2)


def analysera(ticker: str, horisont: int, *, skriv: bool = True) -> dict:
    """Kor hela kedjan. Kastar AnalysAvbruten om farskhetskontrollen misslyckas."""
    if horisont not in CONFIG["HORISONTER"]:
        raise AnalysAvbruten(
            f"Horisont {horisont} finns inte i config.yaml: {CONFIG['HORISONTER']}"
        )

    try:
        df, metadata = hamta(ticker)
    except DataFel as fel:
        raise AnalysAvbruten(f"AVBRUTEN: {fel}") from fel

    farskhet = farskhetskontroll(metadata)
    if not farskhet.passerad:
        raise AnalysAvbruten(farskhet.status)

    indikatorer = ind.berakna_alla(df)

    try:
        dagsdf, _ = hamta_dagsdata(ticker, nu=metadata.hamtad_vid)
    except DataFel as fel:
        raise AnalysAvbruten(f"AVBRUTEN: basrat kan inte beraknas: {fel}") from fel

    basrat_upp = ind.basrat(ticker, horisont, df=dagsdf)
    lage = ind.marknadslage(dagsdf)

    riktning, konfidens, bidrag = bedomning(indikatorer, basrat_upp)
    basrat_for_riktning = basrat_upp if riktning == "UPP" else 1.0 - basrat_upp

    rad = {
        "prediktion_id": uuid.uuid4().hex[:12],
        "skapad_vid_utc": metadata.hamtad_vid.isoformat(),
        "ticker": ticker,
        "horisont": horisont,
        "kalla": metadata.kalla,
        "intervall": metadata.intervall,
        "senaste_bar_tid_utc": metadata.senaste_bar_tid.isoformat(),
        "latens_minuter": round(farskhet.latens_minuter, 2),
        "pris_vid_prediktion": round(indikatorer["pris"], 6),
        "riktning": riktning,
        "konfidens": round(konfidens, 4),
        "basrat": round(basrat_upp, 4),
        "basrat_for_riktning": round(basrat_for_riktning, 4),
        "marknadslage": lage,
        "indikatorer_json": json.dumps(
            {k: round(v, 6) for k, v in indikatorer.items()}, sort_keys=True
        ),
        "mognar_tidigast_utc": _mognadstid(metadata.senaste_bar_tid, horisont).isoformat(),
    }

    if skriv:
        skriv_rad(PREDIKTIONSFIL, rad)

    return {
        "rad": rad,
        "indikatorer": indikatorer,
        "bidrag": bidrag,
        "farskhet": farskhet,
        "metadata": metadata,
    }


# --------------------------------------------------------------------------
# Outputblock
# --------------------------------------------------------------------------
#
# PROVISORISKT FORMAT -- CLAUDE.md specificerar hur outputblocket ska se ut och
# fanns inte i repot. Innehallet nedan tacker det som prediktionsraden bar pa.
# --------------------------------------------------------------------------

_BREDD = 66


def formatera_block(resultat: dict) -> str:
    rad = resultat["rad"]
    ind_varden = resultat["indikatorer"]
    edge = rad["konfidens"] - rad["basrat_for_riktning"]

    rader = [
        "=" * _BREDD,
        f"ANALYS  {rad['ticker']}  |  horisont {rad['horisont']} handelsdagar",
        "=" * _BREDD,
        "",
        "DATA",
        f"  kalla                 {rad['kalla']} ({rad['intervall']})",
        f"  senaste bar           {rad['senaste_bar_tid_utc']}",
        f"  hamtad                {rad['skapad_vid_utc']}",
        f"  latens                {rad['latens_minuter']} min "
        f"(grans {CONFIG['MAX_LATENS_MINUTER']} min)",
        f"  farskhetskontroll     {resultat['farskhet'].status}",
        f"  marknadslage          {rad['marknadslage']}",
        "",
        "INDIKATORER",
    ]

    for namn in sorted(ind_varden):
        rader.append(f"  {namn:<21} {ind_varden[namn]:>14.4f}")

    rader += [
        "",
        "BEDOMNING",
        f"  riktning              {rad['riktning']}",
        f"  konfidens             {rad['konfidens'] * 100:.1f} %",
        f"  basrat ({rad['riktning']})          {rad['basrat_for_riktning'] * 100:.1f} % "
        f"(2 ar dagsdata)",
        f"  pastadd edge          {edge * 100:+.1f} procentenheter (pastaende, inte resultat)",
        "",
        "  Edge, inte konfidens, ar det som betyder nagot. En konfidens pa",
        "  62 procent mot en basrat pa 62 procent ar noll information.",
        "",
        "BIDRAG (log-odds, provisoriska vikter)",
    ]

    if resultat["bidrag"]:
        for namn, varde in sorted(
            resultat["bidrag"].items(), key=lambda p: -abs(p[1])
        ):
            rader.append(f"  {namn:<21} {varde:>+14.4f}")
    else:
        rader.append("  (inga -- underlaget racker inte till nagon indikator)")

    rader += [
        "",
        "LOGGNING",
        f"  prediktion_id         {rad['prediktion_id']}",
        f"  mognar tidigast       {rad['mognar_tidigast_utc']}",
        f"  fil                   {PREDIKTIONSFIL.name}",
        "=" * _BREDD,
    ]
    return "\n".join(rader)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analysera en ticker pa en horisont.")
    parser.add_argument("--ticker", required=True, help="t.ex. NVDA eller ERIC-B.ST")
    parser.add_argument(
        "--horisont",
        required=True,
        type=int,
        choices=CONFIG["HORISONTER"],
        help="antal handelsdagar",
    )
    parser.add_argument(
        "--torrkorning",
        action="store_true",
        help="skriv inte nagon rad till predictions.csv",
    )
    args = parser.parse_args(argv)

    try:
        resultat = analysera(args.ticker, args.horisont, skriv=not args.torrkorning)
    except (AnalysAvbruten, DataFel, ind.IndikatorFel) as fel:
        print(str(fel), file=sys.stderr)
        print("Ingen prediktion loggad.", file=sys.stderr)
        return 2

    print(formatera_block(resultat))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
