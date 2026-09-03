"""Tester for basrat i indicators.py.

Basraten ar referenspunkten hela utvarderingen vilar pa. Ar den fel ser en
vardelos modell bra ut. Testerna ar darfor mest exakta varden, inte intervall.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from indicators import IndikatorFel, basrat, basrat_fran_serie


def serie(varden):
    idx = pd.date_range("2024-01-01", periods=len(varden), freq="D", tz="UTC")
    return pd.Series([float(v) for v in varden], index=idx)


# --- Exakta varden ----------------------------------------------------------


def test_strikt_stigande_serie_ger_ett():
    assert basrat_fran_serie(serie([1, 2, 3, 4, 5]), 1) == 1.0


def test_strikt_fallande_serie_ger_noll():
    assert basrat_fran_serie(serie([5, 4, 3, 2, 1]), 1) == 0.0


def test_oforandrat_pris_raknas_inte_som_positivt():
    # Jamforelsen ar strikt: slut[t+h] > slut[t]. Nollavkastning ar inte en uppgang.
    assert basrat_fran_serie(serie([10, 10, 10, 10]), 1) == 0.0


def test_handraknat_exempel_horisont_ett():
    # [10, 11, 10, 12] -> 10<11 ja, 11>10 nej, 10<12 ja  ->  2/3
    assert basrat_fran_serie(serie([10, 11, 10, 12]), 1) == pytest.approx(2 / 3)


def test_handraknat_exempel_horisont_tva():
    # h=2 pa [10, 11, 10, 12, 9]: 10 vs 10 nej, 11 vs 12 ja, 10 vs 9 nej -> 1/3
    assert basrat_fran_serie(serie([10, 11, 10, 12, 9]), 2) == pytest.approx(1 / 3)


def test_perioderna_overlappar():
    """n - h jamforelser, inte (n - 1) // h. Overlappande fonster ar avsikten."""
    varden = list(range(30))
    assert basrat_fran_serie(serie(varden), 5) == 1.0  # anvander alla 25 fonstren


def test_alltid_mellan_noll_och_ett():
    rng = np.random.default_rng(7)
    pris = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 500)))
    for h in (1, 5, 20, 60):
        assert 0.0 <= basrat_fran_serie(serie(pris), h) <= 1.0


def test_langre_horisont_ger_hogre_basrat_vid_positiv_drift():
    rng = np.random.default_rng(3)
    pris = 100 * np.exp(np.cumsum(rng.normal(0.001, 0.005, 1000)))
    assert basrat_fran_serie(serie(pris), 20) > basrat_fran_serie(serie(pris), 1)


# --- Fel, inga tysta fallbacks ---------------------------------------------


@pytest.mark.parametrize("horisont", [0, -1, -5])
def test_ogiltig_horisont_kastar(horisont):
    with pytest.raises(IndikatorFel):
        basrat_fran_serie(serie([1, 2, 3, 4]), horisont)


def test_icke_heltalshorisont_kastar():
    with pytest.raises(IndikatorFel):
        basrat_fran_serie(serie([1, 2, 3, 4]), 2.5)


def test_for_kort_serie_kastar_i_stallet_for_att_ge_noll():
    with pytest.raises(IndikatorFel, match="kraver minst"):
        basrat_fran_serie(serie([1, 2, 3]), 5)


def test_exakt_lika_manga_varden_som_horisonten_kastar():
    with pytest.raises(IndikatorFel):
        basrat_fran_serie(serie([1, 2, 3, 4, 5]), 5)


def test_tom_serie_kastar():
    with pytest.raises(IndikatorFel):
        basrat_fran_serie(serie([]), 1)


def test_nan_rensas_bort_inte_tolkas_som_nollpris():
    ren = serie([10, 11, 12, 13])
    med_nan = serie([10, 11, np.nan, 12, 13]).copy()
    assert basrat_fran_serie(med_nan, 1) == basrat_fran_serie(ren, 1) == 1.0


# --- basrat(ticker, horisont) ----------------------------------------------


def test_basrat_anvander_close_ur_dataframe_utan_natverk():
    idx = pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC")
    df = pd.DataFrame(
        {
            "Open": range(10),
            "High": range(10),
            "Low": range(10),
            # Close faller medan ovriga kolumner stiger: fel kolumn ger fel svar.
            "Close": list(range(10, 0, -1)),
            "Volume": [1] * 10,
        },
        index=idx,
    )
    assert basrat("NVDA", 1, df=df) == 0.0


def test_basrat_med_df_gor_inget_natverksanrop(monkeypatch):
    import data as datamodul

    def explodera(*a, **k):
        raise AssertionError("basrat hamtade data trots att df gavs")

    monkeypatch.setattr(datamodul, "hamta_dagsdata", explodera)
    df = pd.DataFrame(
        {"Close": [1.0, 2.0, 3.0]},
        index=pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC"),
    )
    assert basrat("NVDA", 1, df=df) == 1.0
