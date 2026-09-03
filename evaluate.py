"""Utvarderar mogna prediktioner och rapporterar om botten faktiskt kan nagot.

Rapporten ar byggd for att kunna saga nej. Traffsakerhet i sig sager ingenting
-- en aktie som gar upp 58 procent av tiden ger 58 procents traffsakerhet at en
modell som alltid gissar UPP. Darfor ar huvudmattet edge: traffsakerhet minus
basrat. Ar den noll finns ingen signal, hur hog traffsakerheten an ser ut.

Anvandning:
    python evaluate.py                # fyll i utfall och rapportera
    python evaluate.py --bara-rapport # rapportera pa befintliga utfall
"""

from __future__ import annotations

import argparse
import csv
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from config import CONFIG, PREDIKTIONSFIL
from data import DataFel, hamta
from predictions_schema import FALT

BRUSVARNING = "OTILLRACKLIGT UNDERLAG - RESULTATET AR BRUS"
_BREDD = 72


# --------------------------------------------------------------------------
# Lasning och skrivning
# --------------------------------------------------------------------------


def las_prediktioner(sokvag: Path) -> list[dict]:
    sokvag = Path(sokvag)
    if not sokvag.exists():
        return []
    with sokvag.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def skriv_prediktioner(sokvag: Path, rader: list[dict]) -> None:
    with Path(sokvag).open("w", newline="", encoding="utf-8") as fh:
        skrivare = csv.DictWriter(fh, fieldnames=FALT, extrasaction="ignore")
        skrivare.writeheader()
        for rad in rader:
            skrivare.writerow({f: rad.get(f, "") for f in FALT})


def _tid(text: str) -> datetime:
    return pd.Timestamp(text).tz_convert("UTC").to_pydatetime()


# --------------------------------------------------------------------------
# Steg 1: fyll i utfall
# --------------------------------------------------------------------------


class Dagscache:
    """Hamtar dagsbarer en gang per ticker och aterger dem till alla rader."""

    def __init__(self, nu: datetime) -> None:
        self._nu = nu
        self._cache: dict[str, pd.DataFrame | None] = {}

    def hamta(self, ticker: str) -> pd.DataFrame | None:
        if ticker not in self._cache:
            try:
                df, _ = hamta(ticker, intervall="1d", period="10y", nu=self._nu)
                self._cache[ticker] = df
            except DataFel as fel:
                print(f"  varning: {ticker} kunde inte hamtas ({fel})")
                self._cache[ticker] = None
        return self._cache[ticker]


def _utfall_ur_dagsdata(
    df: pd.DataFrame, fran: datetime, horisont: int
) -> tuple[pd.Timestamp, float] | None:
    """Baren `horisont` handelsdagar efter `fran`, eller None om den inte finns an."""
    tidigare = df.index[df.index <= pd.Timestamp(fran)]
    if len(tidigare) == 0:
        return None
    startpos = int(df.index.get_loc(tidigare[-1]))
    malpos = startpos + horisont
    if malpos >= len(df):
        return None
    return df.index[malpos], float(df["Close"].iloc[malpos])


def fyll_utfall(rader: list[dict], *, nu: datetime | None = None) -> int:
    """Fyller i utfallsfalten for prediktioner som mognat. Returnerar antal fyllda."""
    nu = nu or datetime.now(timezone.utc)
    cache = Dagscache(nu)
    index_df = cache.hamta(CONFIG["BENCHMARK_INDEX"])
    fyllda = 0

    for rad in rader:
        if rad.get("traff") not in ("", None):
            continue
        if not rad.get("mognar_tidigast_utc") or _tid(rad["mognar_tidigast_utc"]) > nu:
            continue

        df = cache.hamta(rad["ticker"])
        if df is None:
            continue

        horisont = int(rad["horisont"])
        start = _tid(rad["senaste_bar_tid_utc"])
        utfall = _utfall_ur_dagsdata(df, start, horisont)
        if utfall is None:
            continue

        bartid, pris = utfall
        utgangspris = float(rad["pris_vid_prediktion"])
        avkastning = pris / utgangspris - 1.0
        faktisk = "UPP" if pris > utgangspris else "NER"

        rad["utvarderad_vid_utc"] = nu.isoformat()
        rad["utfall_bar_tid_utc"] = bartid.isoformat()
        rad["utfall_pris"] = f"{pris:.6f}"
        rad["utfall_avkastning"] = f"{avkastning:.6f}"
        rad["utfall_riktning"] = faktisk
        rad["traff"] = "1" if faktisk == rad["riktning"] else "0"

        if index_df is not None:
            index_utfall = _utfall_ur_dagsdata(index_df, start, horisont)
            index_start = index_df.index[index_df.index <= pd.Timestamp(start)]
            if index_utfall is not None and len(index_start) > 0:
                fran_pris = float(index_df["Close"].loc[index_start[-1]])
                rad["index_avkastning"] = f"{index_utfall[1] / fran_pris - 1.0:.6f}"

        fyllda += 1

    return fyllda


# --------------------------------------------------------------------------
# Steg 2: rapport
# --------------------------------------------------------------------------


def _mogna(rader: list[dict]) -> pd.DataFrame:
    klara = [r for r in rader if r.get("traff") in ("0", "1")]
    if not klara:
        return pd.DataFrame()
    df = pd.DataFrame(klara)
    for kolumn in (
        "traff", "konfidens", "basrat", "basrat_for_riktning",
        "latens_minuter", "utfall_avkastning", "index_avkastning", "horisont",
    ):
        df[kolumn] = pd.to_numeric(df.get(kolumn), errors="coerce")
    return df


def _andel(traffar: pd.Series) -> tuple[int, float | None]:
    n = int(traffar.notna().sum())
    return n, (float(traffar.mean()) if n else None)


def _rubrik(text: str) -> list[str]:
    return ["", text, "-" * len(text)]


def _procent(varde: float | None) -> str:
    return "-" if varde is None else f"{varde * 100:.1f} %"


def rapport(rader: list[dict], sokvag: Path | None = None) -> str:
    sokvag = Path(sokvag) if sokvag is not None else PREDIKTIONSFIL
    df = _mogna(rader)
    n = len(df)
    ut = ["=" * _BREDD, "UTVARDERING AV PREDIKTIONER", "=" * _BREDD]

    total = len(rader)
    ut.append(f"Rader i {sokvag.name}: {total}   varav mogna och utvarderade: {n}")

    if n == 0:
        ut += ["", BRUSVARNING, "Inga mogna prediktioner an. Ingenting att rapportera."]
        return "\n".join(ut)

    traffsakerhet = float(df["traff"].mean())
    medelbasrat = float(df["basrat_for_riktning"].mean())
    edge = traffsakerhet - medelbasrat

    # ---- 6. Underlagsvarning forst, sa att inget nedanfor lases som ett resultat.
    minsta = int(CONFIG["MIN_ANTAL_FOR_SLUTSATS"])
    if n < minsta:
        ut += [
            "",
            BRUSVARNING,
            f"n = {n}, kravet ar {minsta}. Siffrorna nedan star kvar for att kunna "
            "folja utvecklingen,",
            "men de far inte tolkas som bevis for att modellen fungerar eller inte.",
        ]

    # ---- 1. Traffsakerhet
    ut += _rubrik("1. TRAFFSAKERHET")
    ut.append(f"  totalt                    {_procent(traffsakerhet)}  (n = {n})")
    ut.append("")
    ut.append("  per konfidensintervall (kalibrering: traffsakerhet bor likna konfidens)")
    for lo, hi in CONFIG["KONFIDENSINTERVALL"]:
        grupp = df[(df["konfidens"] * 100 >= lo) & (df["konfidens"] * 100 < hi)]
        gn, ga = _andel(grupp["traff"])
        snitt = f"{grupp['konfidens'].mean() * 100:.1f} %" if gn else "-"
        ut.append(
            f"    {f'{lo}-{hi} %':<15} traff {_procent(ga):<9} "
            f"snittkonfidens {snitt:<9} n = {gn}"
        )

    # ---- 2. Brier score
    p = df["konfidens"].to_numpy(dtype=float)
    y = df["traff"].to_numpy(dtype=float)
    p_bas = df["basrat_for_riktning"].to_numpy(dtype=float)
    brier = float(np.mean((p - y) ** 2))
    brier_bas = float(np.mean((p_bas - y) ** 2))

    ut += _rubrik("2. BRIER SCORE (lagre ar battre, 0 = perfekt, 0.25 = myntkast)")
    ut.append(f"  modellen                  {brier:.4f}")
    ut.append(f"  alltid basraten           {brier_bas:.4f}")
    ut.append(f"  skillnad                  {brier - brier_bas:+.4f}")
    ut.append(
        "  " + (
            "Modellen slar basratsmodellen."
            if brier < brier_bas
            else "Modellen ar SAMRE an att bara gissa basraten."
        )
    )

    # ---- 3. Edge, huvudmattet
    edge_serie = df["traff"] - df["basrat_for_riktning"]
    stdfel = float(edge_serie.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")

    ut += _rubrik("3. GENOMSNITTLIG EDGE (HUVUDMATTET)")
    ut.append(f"  traffsakerhet             {_procent(traffsakerhet)}")
    ut.append(f"  genomsnittlig basrat      {_procent(medelbasrat)}")
    ut.append(f"  edge                      {edge * 100:+.2f} procentenheter")
    if n > 1:
        ut.append(f"  standardfel               {stdfel * 100:.2f} procentenheter")
        ut.append(
            f"  95 % konfidensintervall   [{(edge - 1.96 * stdfel) * 100:+.2f}, "
            f"{(edge + 1.96 * stdfel) * 100:+.2f}] procentenheter"
        )
        if edge - 1.96 * stdfel <= 0 <= edge + 1.96 * stdfel:
            ut.append("  Intervallet innehaller noll. Ingen patvisad edge.")

    # ---- 4. Benchmarks
    slump_forvantad = float(np.mean(p_bas**2 + (1 - p_bas) ** 2))
    index_avk = df["index_avkastning"].dropna()
    egen_avk = df["utfall_avkastning"].dropna()

    ut += _rubrik("4. BENCHMARK")
    ut.append(
        f"  slumpprediktion viktad efter basrat   {_procent(slump_forvantad)} "
        "forvantad traff"
    )
    ut.append(f"  modellen                              {_procent(traffsakerhet)}")
    ut.append(
        f"  skillnad                              "
        f"{(traffsakerhet - slump_forvantad) * 100:+.2f} procentenheter"
    )
    ut.append("")
    if len(index_avk):
        ut.append(
            f"  {'buy-and-hold ' + CONFIG['BENCHMARK_INDEX']:<37} "
            f"snittavkastning {index_avk.mean() * 100:+.2f} % over samma fonster"
        )
        ut.append(
            f"  {'andel positiva index-fonster':<37} "
            f"{_procent(float((index_avk > 0).mean()))}  (n = {len(index_avk)})"
        )
    else:
        ut.append(f"  buy-and-hold {CONFIG['BENCHMARK_INDEX']}: ingen indexdata sparad.")
    if len(egen_avk):
        ut.append(
            f"  {'aktiens snittavkastning':<37} {egen_avk.mean() * 100:+.2f} % "
            "over samma fonster"
        )

    # ---- 5. Binomialtest
    traffar = int(df["traff"].sum())
    p0 = min(max(medelbasrat, 1e-9), 1 - 1e-9)
    test = stats.binomtest(traffar, n, p0, alternative="two-sided")
    ki = test.proportion_ci(confidence_level=0.95, method="wilson")

    ut += _rubrik("5. BINOMIALTEST MOT BASRATEN")
    ut.append(f"  nollhypotes               traffsannolikhet = basrat = {_procent(p0)}")
    ut.append(f"  utfall                    {traffar} traffar av {n}")
    ut.append(f"  p-varde                   {test.pvalue:.4g}")
    ut.append(
        f"  95 % KI (Wilson)          [{ki.low * 100:.1f} %, {ki.high * 100:.1f} %]"
    )
    if test.pvalue < 0.05 and n >= minsta:
        riktning = "battre" if traffsakerhet > p0 else "samre"
        ut.append(f"  Signifikant {riktning} an basraten pa 5 procents niva.")
    elif n < minsta:
        ut.append("  For litet underlag for att tolka p-vardet. Se varningen ovan.")
    else:
        ut.append("  Kan inte skiljas fran basraten. Ingen patvisad skicklighet.")

    # ---- 7. Uppdelningar
    ut += _rubrik("7. UPPDELAD TRAFFSAKERHET")
    ut.append("  per latens")
    for lo, hi in CONFIG["LATENSINTERVALL"]:
        grupp = df[(df["latens_minuter"] >= lo) & (df["latens_minuter"] < hi)]
        gn, ga = _andel(grupp["traff"])
        gbas = float(grupp["basrat_for_riktning"].mean()) if gn else None
        gedge = (ga - gbas) if (ga is not None and gbas is not None) else None
        gedge_text = "-" if gedge is None else f"{gedge * 100:+.1f} pe"
        ut.append(
            f"    {f'{lo}-{hi} min':<15} traff {_procent(ga):<9} "
            f"edge {gedge_text:<10} n = {gn}"
        )

    ut.append("")
    ut.append("  per marknadslage")
    for lage in sorted(df["marknadslage"].fillna("OKANT").unique()):
        grupp = df[df["marknadslage"].fillna("OKANT") == lage]
        gn, ga = _andel(grupp["traff"])
        gbas = float(grupp["basrat_for_riktning"].mean()) if gn else None
        gedge = (ga - gbas) if (ga is not None and gbas is not None) else None
        gedge_text = "-" if gedge is None else f"{gedge * 100:+.1f} pe"
        ut.append(
            f"    {lage:<15} traff {_procent(ga):<9} edge {gedge_text:<10} n = {gn}"
        )

    ut.append("")
    ut.append(
        "  Uppdelningar tar slut pa data snabbt. En grupp med farre an "
        f"{minsta} rader sager ingenting."
    )
    ut.append("=" * _BREDD)
    return "\n".join(ut)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Utvardera loggade prediktioner.")
    parser.add_argument(
        "--fil", default=str(PREDIKTIONSFIL), help="sokvag till predictions.csv"
    )
    parser.add_argument(
        "--bara-rapport",
        action="store_true",
        help="hamta ingen ny data, rapportera pa redan ifyllda utfall",
    )
    args = parser.parse_args(argv)

    sokvag = Path(args.fil)
    rader = las_prediktioner(sokvag)
    if not rader:
        print(f"{sokvag} saknas eller ar tom. Inget att utvardera.")
        print(BRUSVARNING)
        return 0

    if not args.bara_rapport:
        fyllda = fyll_utfall(rader)
        print(f"Fyllde i utfall for {fyllda} prediktioner.")
        if fyllda:
            skriv_prediktioner(sokvag, rader)

    print(rapport(rader, sokvag))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
