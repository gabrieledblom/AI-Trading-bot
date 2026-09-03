"""Tester for farskhetskontrollen i data.py.

Kontrollen ar den enda spparren mot att analysera gammal data. Den ska passera
farsk data, avbryta gammal, och kasta -- inte gissa -- nar tidsstampeln saknas.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from data import PASSERAD, DataFel, Metadata, farskhetskontroll, krav_farsk

NU = datetime(2026, 9, 3, 15, 0, tzinfo=timezone.utc)


def bygg(alder_minuter: float = 1.0, *, antal_barer: int = 500, **overskrivningar):
    grund = dict(
        ticker="NVDA",
        kalla="yfinance",
        intervall="5m",
        antal_barer=antal_barer,
        hamtad_vid=NU,
        senaste_bar_tid=NU - timedelta(minutes=alder_minuter),
        forsta_bar_tid=NU - timedelta(days=30),
    )
    grund.update(overskrivningar)
    return Metadata(**grund)


# --- Passerar ---------------------------------------------------------------


@pytest.mark.parametrize("alder", [0.0, 1.0, 10.0, 19.9, 20.0])
def test_farsk_data_passerar_anda_till_gransen(alder):
    resultat = farskhetskontroll(bygg(alder), max_latens_minuter=20)
    assert resultat.status == PASSERAD
    assert resultat.passerad
    assert resultat.latens_minuter == pytest.approx(alder)


def test_gransen_lases_fran_config_nar_den_inte_anges():
    from config import CONFIG

    grans = CONFIG["MAX_LATENS_MINUTER"]
    assert farskhetskontroll(bygg(grans - 0.1)).passerad
    assert not farskhetskontroll(bygg(grans + 0.1)).passerad


# --- Avbryter ---------------------------------------------------------------


@pytest.mark.parametrize("alder", [20.1, 45.0, 60 * 24])
def test_gammal_data_avbryter(alder):
    resultat = farskhetskontroll(bygg(alder), max_latens_minuter=20)
    assert not resultat.passerad
    assert resultat.status.startswith("AVBRUTEN")
    # Avbrottsorsaken ska namna bade uppmatt latens och grans, annars ar den varldelos.
    assert "20" in resultat.status
    assert "NVDA" in resultat.status


def test_grans_ar_strikt_ingen_avrundning_uppat():
    # 20.4 minuter far inte avrundas till 20 och slinka igenom.
    assert not farskhetskontroll(bygg(20.4), max_latens_minuter=20).passerad


def test_bar_i_framtiden_avbryter():
    resultat = farskhetskontroll(bygg(-5.0), max_latens_minuter=20)
    assert not resultat.passerad
    assert "framtiden" in resultat.status


def test_noll_barer_avbryter():
    resultat = farskhetskontroll(bygg(1.0, antal_barer=0), max_latens_minuter=20)
    assert not resultat.passerad
    assert "noll barer" in resultat.status


# --- Kastar, ingen tyst fallback -------------------------------------------


def test_saknad_metadata_kastar():
    with pytest.raises(DataFel):
        farskhetskontroll(None)


def test_saknad_tidsstampel_kastar_i_stallet_for_att_avbryta():
    """Saknad tidsstampel ar ett trasigt datauttag, inte ett normalt avbrott."""
    trasig = bygg(1.0)
    object.__setattr__(trasig, "senaste_bar_tid", None)
    with pytest.raises(DataFel, match="senaste_bar_tid saknas"):
        farskhetskontroll(trasig)


def test_saknad_hamtningstid_kastar():
    trasig = bygg(1.0)
    object.__setattr__(trasig, "hamtad_vid", None)
    with pytest.raises(DataFel, match="hamtad_vid saknas"):
        farskhetskontroll(trasig)


def test_naiv_tidsstampel_kastar_i_stallet_for_att_anta_utc():
    trasig = bygg(1.0)
    object.__setattr__(trasig, "senaste_bar_tid", datetime(2026, 9, 3, 14, 59))
    with pytest.raises(DataFel, match="saknar tidszon"):
        farskhetskontroll(trasig)


def test_latens_raknas_over_tidszoner():
    """En bar stamplad i annan tidszon far inte se ut som timmar gammal."""
    from zoneinfo import ZoneInfo

    stockholm = (NU - timedelta(minutes=3)).astimezone(ZoneInfo("Europe/Stockholm"))
    resultat = farskhetskontroll(bygg(1.0, senaste_bar_tid=stockholm), max_latens_minuter=20)
    assert resultat.passerad
    assert resultat.latens_minuter == pytest.approx(3.0)


# --- krav_farsk -------------------------------------------------------------


def test_krav_farsk_slapper_igenom_farsk_data():
    assert krav_farsk(bygg(2.0), 20).passerad


def test_krav_farsk_kastar_pa_gammal_data():
    with pytest.raises(DataFel, match="AVBRUTEN"):
        krav_farsk(bygg(99.0), 20)
