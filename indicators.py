"""Indikatorer, beraknade fran grunden.

Inga tredjepartsbibliotek for teknisk analys. Varje formel star skriven har sa
att den gar att granska och ifragasatta. pandas anvands bara som datastruktur.

VARNING -- OKLAR SPECIFIKATION
Uppdraget sager "alla indikatorer fran CLAUDE.md". CLAUDE.md fanns inte i repot
nar den har filen skrevs (GitHub svarade "Git Repository is empty"). Urvalet
nedan ar darfor mitt eget och ska granskas mot CLAUDE.md nar den finns.
`basrat` ar daremot specificerad direkt i uppdraget och ar inte provisorisk.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from config import CONFIG


class IndikatorFel(Exception):
    """Kastas nar en indikator inte kan beraknas pa tillgangligt underlag."""


def _serie(varden: pd.Series | pd.DataFrame, namn: str) -> pd.Series:
    if isinstance(varden, pd.DataFrame):
        if namn not in varden.columns:
            raise IndikatorFel(f"Kolumn {namn!r} saknas.")
        varden = varden[namn]
    return pd.to_numeric(varden, errors="coerce").astype(float)


def _krav_langd(serie: pd.Series, minsta: int, vad: str) -> None:
    if len(serie.dropna()) < minsta:
        raise IndikatorFel(
            f"{vad} kraver minst {minsta} varden, fick {len(serie.dropna())}."
        )


# --------------------------------------------------------------------------
# Glidande medelvarden
# --------------------------------------------------------------------------


def sma(serie: pd.Series, fonster: int) -> pd.Series:
    """Enkelt glidande medelvarde: medelvardet av de senaste `fonster` vardena."""
    if fonster < 1:
        raise IndikatorFel(f"sma kraver fonster >= 1, fick {fonster}.")
    return serie.rolling(window=fonster, min_periods=fonster).mean()


def ema(serie: pd.Series, fonster: int) -> pd.Series:
    """Exponentiellt glidande medelvarde, alpha = 2 / (fonster + 1).

    Seedas med SMA over de forsta `fonster` vardena, sedan rekursivt:
        ema[t] = alpha * x[t] + (1 - alpha) * ema[t-1]
    """
    if fonster < 1:
        raise IndikatorFel(f"ema kraver fonster >= 1, fick {fonster}.")
    return _rekursiv_utjamning(serie, alfa=2.0 / (fonster + 1.0), seedfonster=fonster)


def _wilder(serie: pd.Series, fonster: int) -> pd.Series:
    """Wilders utjamning, alpha = 1 / fonster. Anvands av RSI, ATR och ADX."""
    return _rekursiv_utjamning(serie, alfa=1.0 / fonster, seedfonster=fonster)


def _rekursiv_utjamning(serie: pd.Series, alfa: float, seedfonster: int) -> pd.Series:
    varden = serie.to_numpy(dtype=float)
    ut = np.full(varden.shape, np.nan)
    if len(varden) < seedfonster:
        return pd.Series(ut, index=serie.index, dtype=float)

    # Hoppa fram till forsta fonstret som faktiskt innehaller data. Ett fonster
    # som bara ar NaN far inget seed alls -- vi hittar inte pa ett startvarde.
    giltiga = np.flatnonzero(~np.isnan(varden))
    if len(giltiga) == 0:
        return pd.Series(ut, index=serie.index, dtype=float)

    forsta = max(seedfonster - 1, int(giltiga[0]) + seedfonster - 1)
    if forsta >= len(varden):
        return pd.Series(ut, index=serie.index, dtype=float)

    seed = np.nanmean(varden[forsta - seedfonster + 1 : forsta + 1])
    if np.isnan(seed):
        return pd.Series(ut, index=serie.index, dtype=float)

    ut[forsta] = seed
    for i in range(forsta + 1, len(varden)):
        x = varden[i]
        if np.isnan(x):
            ut[i] = ut[i - 1]
        else:
            ut[i] = alfa * x + (1.0 - alfa) * ut[i - 1]
    return pd.Series(ut, index=serie.index, dtype=float)


# --------------------------------------------------------------------------
# Momentum
# --------------------------------------------------------------------------


def rsi(serie: pd.Series, fonster: int = 14) -> pd.Series:
    """Relative Strength Index enligt Wilder.

        RS  = utjamnad uppgang / utjamnad nedgang
        RSI = 100 - 100 / (1 + RS)

    Nedgang noll ger RSI 100 (definitionsmassigt, inte division med noll).
    """
    _krav_langd(serie, fonster + 1, f"rsi({fonster})")
    forandring = serie.diff()
    upp = forandring.clip(lower=0.0)
    ner = (-forandring).clip(lower=0.0)

    upp_utjamnad = _wilder(upp.fillna(0.0), fonster)
    ner_utjamnad = _wilder(ner.fillna(0.0), fonster)

    rs = upp_utjamnad / ner_utjamnad.replace(0.0, np.nan)
    ut = 100.0 - 100.0 / (1.0 + rs)
    ut = ut.where(ner_utjamnad != 0.0, 100.0)
    ut = ut.where(~((ner_utjamnad == 0.0) & (upp_utjamnad == 0.0)), 50.0)
    ut.iloc[:fonster] = np.nan
    return ut


def macd(
    serie: pd.Series,
    snabb: int = 12,
    langsam: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """MACD-linje, signallinje och histogram."""
    if snabb >= langsam:
        raise IndikatorFel("macd kraver snabb < langsam.")
    _krav_langd(serie, langsam + signal, "macd")
    linje = ema(serie, snabb) - ema(serie, langsam)
    signallinje = _rekursiv_utjamning(
        linje, alfa=2.0 / (signal + 1.0), seedfonster=signal
    )
    return pd.DataFrame(
        {"macd": linje, "signal": signallinje, "histogram": linje - signallinje}
    )


def stochastic(
    hog: pd.Series, lag: pd.Series, slut: pd.Series, fonster: int = 14, glattning: int = 3
) -> pd.DataFrame:
    """Stochastic oscillator: var i sitt senaste intervall stangningen ligger."""
    _krav_langd(slut, fonster, f"stochastic({fonster})")
    hogsta = hog.rolling(fonster, min_periods=fonster).max()
    lagsta = lag.rolling(fonster, min_periods=fonster).min()
    spann = (hogsta - lagsta).replace(0.0, np.nan)
    k = 100.0 * (slut - lagsta) / spann
    return pd.DataFrame({"k": k, "d": k.rolling(glattning, min_periods=glattning).mean()})


def momentum(serie: pd.Series, fonster: int) -> pd.Series:
    """Procentuell forandring over `fonster` barer."""
    if fonster < 1:
        raise IndikatorFel(f"momentum kraver fonster >= 1, fick {fonster}.")
    return serie / serie.shift(fonster) - 1.0


# --------------------------------------------------------------------------
# Volatilitet och spann
# --------------------------------------------------------------------------


def true_range(hog: pd.Series, lag: pd.Series, slut: pd.Series) -> pd.Series:
    """max(hog-lag, |hog - foregaende slut|, |lag - foregaende slut|)."""
    fg = slut.shift(1)
    return pd.concat(
        [(hog - lag).abs(), (hog - fg).abs(), (lag - fg).abs()], axis=1
    ).max(axis=1)


def atr(hog: pd.Series, lag: pd.Series, slut: pd.Series, fonster: int = 14) -> pd.Series:
    """Average True Range, Wilder-utjamnad."""
    _krav_langd(slut, fonster + 1, f"atr({fonster})")
    return _wilder(true_range(hog, lag, slut).fillna(0.0), fonster)


def bollinger(serie: pd.Series, fonster: int = 20, antal_std: float = 2.0) -> pd.DataFrame:
    """Bollingerband plus bandbredd och %B (var i bandet priset ligger)."""
    _krav_langd(serie, fonster, f"bollinger({fonster})")
    mitt = sma(serie, fonster)
    std = serie.rolling(fonster, min_periods=fonster).std(ddof=0)
    ovre = mitt + antal_std * std
    undre = mitt - antal_std * std
    bredd = (ovre - undre) / mitt.replace(0.0, np.nan)
    procent_b = (serie - undre) / (ovre - undre).replace(0.0, np.nan)
    return pd.DataFrame(
        {"mitt": mitt, "ovre": ovre, "undre": undre, "bredd": bredd, "procent_b": procent_b}
    )


def realiserad_volatilitet(
    serie: pd.Series, fonster: int = 20, barer_per_ar: int = 252
) -> pd.Series:
    """Annualiserad standardavvikelse for logaritmisk avkastning."""
    _krav_langd(serie, fonster + 1, f"realiserad_volatilitet({fonster})")
    logavk = np.log(serie / serie.shift(1))
    return logavk.rolling(fonster, min_periods=fonster).std(ddof=1) * np.sqrt(barer_per_ar)


# --------------------------------------------------------------------------
# Volym
# --------------------------------------------------------------------------


def volymratio(volym: pd.Series, fonster: int = 20) -> pd.Series:
    """Senaste volym delat med sitt eget glidande medelvarde. 1.0 = normalt."""
    _krav_langd(volym, fonster, f"volymratio({fonster})")
    return volym / sma(volym, fonster).replace(0.0, np.nan)


def obv(slut: pd.Series, volym: pd.Series) -> pd.Series:
    """On Balance Volume: kumulativ volym med tecken efter prisriktning."""
    riktning = np.sign(slut.diff().fillna(0.0))
    return (riktning * volym.fillna(0.0)).cumsum()


# --------------------------------------------------------------------------
# Trendstyrka
# --------------------------------------------------------------------------


def adx(hog: pd.Series, lag: pd.Series, slut: pd.Series, fonster: int = 14) -> pd.DataFrame:
    """Average Directional Index med +DI och -DI enligt Wilder."""
    _krav_langd(slut, 2 * fonster + 1, f"adx({fonster})")
    upp = hog.diff()
    ner = -lag.diff()
    plus_dm = ((upp > ner) & (upp > 0)).astype(float) * upp.fillna(0.0)
    minus_dm = ((ner > upp) & (ner > 0)).astype(float) * ner.fillna(0.0)

    tr_utjamnad = _wilder(true_range(hog, lag, slut).fillna(0.0), fonster)
    namnare = tr_utjamnad.replace(0.0, np.nan)
    plus_di = 100.0 * _wilder(plus_dm, fonster) / namnare
    minus_di = 100.0 * _wilder(minus_dm, fonster) / namnare

    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    return pd.DataFrame(
        {"adx": _wilder(dx.fillna(0.0), fonster), "plus_di": plus_di, "minus_di": minus_di}
    )


# --------------------------------------------------------------------------
# Basrat
# --------------------------------------------------------------------------


def basrat_fran_serie(slutkurser: pd.Series, horisont: int) -> float:
    """Andel overlappande `horisont`-perioder med positiv avkastning.

    Delas ut som andel mellan 0 och 1. Endast slutkurser anvands, jamforelsen
    ar strikt: slut[t+h] > slut[t]. Oforandrat pris raknas inte som positivt.
    """
    if not isinstance(horisont, (int, np.integer)) or horisont < 1:
        raise IndikatorFel(f"basrat kraver heltalshorisont >= 1, fick {horisont!r}.")

    serie = pd.to_numeric(slutkurser, errors="coerce").astype(float).dropna()
    if len(serie) <= horisont:
        raise IndikatorFel(
            f"basrat({horisont}) kraver minst {horisont + 1} slutkurser, "
            f"fick {len(serie)}."
        )

    nu = serie.to_numpy()[:-horisont]
    sedan = serie.to_numpy()[horisont:]
    return float(np.mean(sedan > nu))


def basrat(
    ticker: str,
    horisont: int,
    *,
    df: pd.DataFrame | None = None,
    ar: float | None = None,
    nu: datetime | None = None,
) -> float:
    """Andel positiva `horisont`-dagarsperioder for `ticker` de senaste 2 aren.

    Detta ar referenspunkten som all traffsakerhet ska matas emot. En modell som
    traffar 62 procent pa en aktie med 62 procents basrat har inte last nagot ur
    marknaden -- den har bara upprepat driften.

    Hamtar dagsdata sjalv om `df` inte ges. `df` finns for att tester och
    analyze.py ska slippa dubbla natverksanrop.
    """
    if df is None:
        from data import hamta_dagsdata  # lokal import: bryter cirkulart beroende

        df, _ = hamta_dagsdata(ticker, ar=ar, nu=nu)

    return basrat_fran_serie(_serie(df, "Close"), horisont)


# --------------------------------------------------------------------------
# Marknadslage
# --------------------------------------------------------------------------

MARKNADSLAGEN = ("UPPTREND_LUGN", "UPPTREND_ORLIG", "NEDTREND_LUGN", "NEDTREND_ORLIG", "OKANT")


def marknadslage(df: pd.DataFrame, trendfonster: int = 200, volfonster: int = 20) -> str:
    """Grov regimetikett, sparas med varje prediktion for uppdelning i evaluate.py.

    Trend: slutkurs over eller under sitt SMA(trendfonster).
    Volatilitet: realiserad volatilitet over eller under sin egen median.
    Returnerar OKANT nar underlaget ar for kort -- aldrig en gissad etikett.
    """
    slut = _serie(df, "Close")
    if len(slut.dropna()) < max(trendfonster, volfonster + 1):
        return "OKANT"

    trend = "UPPTREND" if slut.iloc[-1] >= sma(slut, trendfonster).iloc[-1] else "NEDTREND"

    vol = realiserad_volatilitet(slut, volfonster)
    if vol.dropna().empty or np.isnan(vol.iloc[-1]):
        return "OKANT"
    lage = "ORLIG" if vol.iloc[-1] > float(vol.median(skipna=True)) else "LUGN"
    return f"{trend}_{lage}"


# --------------------------------------------------------------------------
# Samlad berakning
# --------------------------------------------------------------------------


def berakna_alla(df: pd.DataFrame) -> dict[str, float]:
    """Beraknar hela indikatoruppsattningen och returnerar sista vardet for var och en.

    Indikatorer som inte kan beraknas pa underlaget utelamnas helt. De far inte
    ersattas med noll eller ett neutralt varde -- det vore en tyst fallback.
    """
    hog, lag, slut, volym = (_serie(df, k) for k in ("High", "Low", "Close", "Volume"))
    ut: dict[str, float] = {"pris": float(slut.iloc[-1])}

    def spara(namn: str, berakna) -> None:
        try:
            varde = berakna()
        except (IndikatorFel, IndexError, ValueError, ZeroDivisionError):
            return
        if varde is None or (isinstance(varde, float) and np.isnan(varde)):
            return
        ut[namn] = float(varde)

    def sista(serie: pd.Series) -> float:
        rensad = serie.dropna()
        if rensad.empty:
            raise IndikatorFel("Inga giltiga varden.")
        return float(rensad.iloc[-1])

    for fonster in (20, 50, 200):
        spara(f"sma_{fonster}", lambda f=fonster: sista(sma(slut, f)))
        spara(
            f"pris_mot_sma_{fonster}",
            lambda f=fonster: float(slut.iloc[-1] / sista(sma(slut, f)) - 1.0),
        )
    for fonster in (12, 26):
        spara(f"ema_{fonster}", lambda f=fonster: sista(ema(slut, f)))

    spara("rsi_14", lambda: sista(rsi(slut, 14)))
    spara("macd", lambda: sista(macd(slut)["macd"]))
    spara("macd_signal", lambda: sista(macd(slut)["signal"]))
    spara("macd_histogram", lambda: sista(macd(slut)["histogram"]))
    spara("stoch_k", lambda: sista(stochastic(hog, lag, slut)["k"]))
    spara("stoch_d", lambda: sista(stochastic(hog, lag, slut)["d"]))

    for fonster in (1, 5, 20):
        spara(f"momentum_{fonster}", lambda f=fonster: sista(momentum(slut, f)))

    spara("atr_14", lambda: sista(atr(hog, lag, slut, 14)))
    spara("atr_14_andel", lambda: float(sista(atr(hog, lag, slut, 14)) / slut.iloc[-1]))
    spara("bollinger_bredd", lambda: sista(bollinger(slut)["bredd"]))
    spara("bollinger_procent_b", lambda: sista(bollinger(slut)["procent_b"]))
    spara("volatilitet_20", lambda: sista(realiserad_volatilitet(slut, 20)))

    spara("volymratio_20", lambda: sista(volymratio(volym, 20)))
    spara("obv", lambda: sista(obv(slut, volym)))

    spara("adx_14", lambda: sista(adx(hog, lag, slut, 14)["adx"]))
    spara("plus_di_14", lambda: sista(adx(hog, lag, slut, 14)["plus_di"]))
    spara("minus_di_14", lambda: sista(adx(hog, lag, slut, 14)["minus_di"]))

    return ut


__all__ = [
    "IndikatorFel",
    "sma", "ema", "rsi", "macd", "stochastic", "momentum",
    "true_range", "atr", "bollinger", "realiserad_volatilitet",
    "volymratio", "obv", "adx",
    "basrat", "basrat_fran_serie", "marknadslage", "MARKNADSLAGEN", "berakna_alla",
]
