"""Datahamtning och farskhetskontroll.

Regel som gar fore allt annat i den har modulen: om nagot ar okant sa kastas ett
fel. Inga defaultvarden, ingen tyst fallback, ingen tidsstampel som hittas pa.
En prediktion pa data av okand alder ar varre an ingen prediktion alls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

from config import CONFIG

PASSERAD = "PASSERAD"

OBLIGATORISKA_KOLUMNER = ("Open", "High", "Low", "Close", "Volume")


class DataFel(Exception):
    """Kastas nar hamtad data ar oanvandbar eller saknar tidsstampel."""


@dataclass(frozen=True)
class Metadata:
    """Harkomsten for ett datauttag. Bada tidsstamplarna ar obligatoriska."""

    ticker: str
    kalla: str
    intervall: str
    antal_barer: int
    #: Nar hamtningen gjordes (UTC).
    hamtad_vid: datetime
    #: Tidsstampeln pa den senaste baren i datamangden (UTC).
    senaste_bar_tid: datetime
    #: Tidsstampeln pa den forsta baren i datamangden (UTC).
    forsta_bar_tid: datetime
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def latens_minuter(self) -> float:
        """Alder pa senaste baren, raknat fran hamtningstillfallet."""
        return (self.hamtad_vid - self.senaste_bar_tid).total_seconds() / 60.0


@dataclass(frozen=True)
class Farskhetsresultat:
    """Utfallet av farskhetskontroll. `status` ar PASSERAD eller en avbrottsorsak."""

    status: str
    latens_minuter: float | None
    granslatens_minuter: float

    @property
    def passerad(self) -> bool:
        return self.status == PASSERAD

    def __str__(self) -> str:
        if self.passerad:
            return f"{PASSERAD} (latens {self.latens_minuter:.1f} min)"
        return self.status


def _till_utc(varde: Any, vad: str) -> datetime:
    """Konverterar till tidszonsmedveten UTC. Kastar hellre an att anta UTC."""
    if varde is None:
        raise DataFel(f"{vad} saknas.")
    tid = pd.Timestamp(varde)
    if tid is pd.NaT or pd.isna(tid):
        raise DataFel(f"{vad} ar NaT.")
    if tid.tzinfo is None:
        raise DataFel(
            f"{vad} saknar tidszon. En naiv tidsstampel kan inte jamforas mot "
            "nutid utan att gissa, och gissningar ar forbjudna har."
        )
    return tid.tz_convert("UTC").to_pydatetime()


def _platta_kolumner(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance returnerar MultiIndex-kolumner for enskilda tickers i vissa versioner."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [str(niva[0]) for niva in df.columns]
    return df


def hamta(
    ticker: str,
    intervall: str | None = None,
    period: str | None = None,
    *,
    nu: datetime | None = None,
) -> tuple[pd.DataFrame, Metadata]:
    """Hamtar prisdata for `ticker`.

    Returnerar (DataFrame, Metadata). DataFrame har ett tidszonsmedvetet
    DatetimeIndex i UTC och kolumnerna Open/High/Low/Close/Volume.

    Kastar DataFel om datamangden ar tom, saknar kolumner, eller om index
    saknar tidsstampel eller tidszon. Ingen tyst fallback.
    """
    import yfinance as yf  # importeras lokalt sa att tester slipper natverksberoendet

    kalla = CONFIG["DATAKALLA"]
    if kalla != "yfinance":
        raise DataFel(f"Okand DATAKALLA i config.yaml: {kalla!r}")

    intervall = intervall or CONFIG["INTERVALL"]
    period = period or _standardperiod(intervall)

    hamtad_vid = nu or datetime.now(timezone.utc)

    df = yf.download(
        ticker,
        interval=intervall,
        period=period,
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    if df is None or len(df) == 0:
        raise DataFel(f"Tom datamangd fran {kalla} for {ticker} ({intervall}/{period}).")

    df = _platta_kolumner(df)

    saknade = [k for k in OBLIGATORISKA_KOLUMNER if k not in df.columns]
    if saknade:
        raise DataFel(f"{ticker}: datamangden saknar kolumner {saknade}.")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise DataFel(f"{ticker}: index ar inte ett DatetimeIndex, tidsstampel saknas.")

    if df.index.tz is None:
        raise DataFel(
            f"{ticker}: index saknar tidszon. Utan tidszon gar latensen inte att "
            "berakna utan att gissa borsens tidszon."
        )

    df = df.tz_convert("UTC").sort_index()
    df = df[list(OBLIGATORISKA_KOLUMNER)].dropna(how="all")

    if len(df) == 0:
        raise DataFel(f"{ticker}: alla barer var tomma efter rensning.")

    metadata = Metadata(
        ticker=ticker,
        kalla=kalla,
        intervall=intervall,
        antal_barer=int(len(df)),
        hamtad_vid=_till_utc(hamtad_vid, "hamtad_vid"),
        senaste_bar_tid=_till_utc(df.index[-1], f"{ticker}: senaste bartidsstampel"),
        forsta_bar_tid=_till_utc(df.index[0], f"{ticker}: forsta bartidsstampel"),
        extra={"period": period},
    )
    return df, metadata


def _standardperiod(intervall: str) -> str:
    """yfinance begransar hur langt bakat varje upplosning far hamtas."""
    return {
        "1m": "7d",
        "2m": "60d",
        "5m": "60d",
        "15m": "60d",
        "30m": "60d",
        "60m": "730d",
        "1h": "730d",
        "1d": "10y",
    }.get(intervall, "60d")


def farskhetskontroll(
    metadata: Metadata,
    max_latens_minuter: float | None = None,
) -> Farskhetsresultat:
    """Avgor om datan ar fardsk nog att analysera.

    Returnerar Farskhetsresultat med status PASSERAD eller en avbrottsorsak i
    klartext. Kastar DataFel om tidsstampel saknas helt -- det ar inte ett
    avbrott utan ett trasigt datauttag.
    """
    if metadata is None:
        raise DataFel("Metadata saknas. Farskhetskontroll kan inte utforas.")

    grans = float(
        max_latens_minuter
        if max_latens_minuter is not None
        else CONFIG["MAX_LATENS_MINUTER"]
    )

    if getattr(metadata, "senaste_bar_tid", None) is None:
        raise DataFel(f"{metadata.ticker}: senaste_bar_tid saknas i metadata.")
    if getattr(metadata, "hamtad_vid", None) is None:
        raise DataFel(f"{metadata.ticker}: hamtad_vid saknas i metadata.")
    if metadata.senaste_bar_tid.tzinfo is None or metadata.hamtad_vid.tzinfo is None:
        raise DataFel(f"{metadata.ticker}: tidsstamplar i metadata saknar tidszon.")

    latens = metadata.latens_minuter

    if latens < 0:
        return Farskhetsresultat(
            status=(
                f"AVBRUTEN: {metadata.ticker} har en bar {abs(latens):.1f} min in i "
                "framtiden. Klockan eller datakallan ar fel."
            ),
            latens_minuter=latens,
            granslatens_minuter=grans,
        )

    if metadata.antal_barer <= 0:
        return Farskhetsresultat(
            status=f"AVBRUTEN: {metadata.ticker} har noll barer.",
            latens_minuter=latens,
            granslatens_minuter=grans,
        )

    if latens > grans:
        return Farskhetsresultat(
            status=(
                f"AVBRUTEN: {metadata.ticker} har {latens:.1f} min gammal data, "
                f"gransen ar {grans:.0f} min. Borsen ar sannolikt stangd eller "
                f"{metadata.kalla} slapar efter."
            ),
            latens_minuter=latens,
            granslatens_minuter=grans,
        )

    return Farskhetsresultat(
        status=PASSERAD,
        latens_minuter=latens,
        granslatens_minuter=grans,
    )


def krav_farsk(metadata: Metadata, max_latens_minuter: float | None = None) -> Farskhetsresultat:
    """Som farskhetskontroll men kastar DataFel vid avbrott. Anvands av analyze.py."""
    resultat = farskhetskontroll(metadata, max_latens_minuter)
    if not resultat.passerad:
        raise DataFel(resultat.status)
    return resultat


def hamta_dagsdata(
    ticker: str,
    ar: float | None = None,
    *,
    nu: datetime | None = None,
) -> tuple[pd.DataFrame, Metadata]:
    """Dagsbarer for basratsberakning. Ingen farskhetskontroll -- historik, inte nutid."""
    ar = float(ar if ar is not None else CONFIG["BASRAT_AR"])
    df, metadata = hamta(ticker, intervall="1d", period="10y", nu=nu)
    slut = metadata.hamtad_vid
    start = slut - timedelta(days=int(round(ar * 365.25)))
    fonster = df.loc[df.index >= pd.Timestamp(start)]
    if len(fonster) == 0:
        raise DataFel(f"{ticker}: ingen dagsdata inom de senaste {ar} aren.")
    return fonster, metadata
