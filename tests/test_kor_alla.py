"""Tester for batchkorningen i analyze.py.

Poangen med --alla ar att en stangd bors for en ticker inte far tysta ner
resten av listan. Testerna handlar darfor mest om vad som INTE hander.
"""

from __future__ import annotations

import pytest

import analyze
from data import DataFel
from indicators import IndikatorFel


class FejkFarskhet:
    status = "PASSERAD"


def fejkresultat(ticker: str) -> dict:
    """Minsta struktur som bade kor_alla och formatera_block klarar av."""
    return {
        "rad": {
            "ticker": ticker,
            "horisont": 5,
            "kalla": "yfinance",
            "intervall": "5m",
            "senaste_bar_tid_utc": "2026-09-04T13:35:00+00:00",
            "skapad_vid_utc": "2026-09-04T13:37:00+00:00",
            "latens_minuter": 3.2,
            "riktning": "UPP",
            "konfidens": 0.62,
            "basrat_for_riktning": 0.55,
            "marknadslage": "UPPTREND_LUGN",
            "prediktion_id": "abc123",
            "mognar_tidigast_utc": "2026-09-13T13:35:00+00:00",
        },
        "indikatorer": {"pris": 100.0, "rsi_14": 55.0},
        "bidrag": {"rsi_14": 0.03},
        "farskhet": FejkFarskhet(),
    }


@pytest.fixture
def stub(monkeypatch):
    """Byter ut analysera() mot en styrbar attrapp. Inget natverk, inga filer."""

    def installera(utfall: dict):
        anropade: list[tuple[str, int]] = []

        def falsk(ticker, horisont, *, skriv=True):
            anropade.append((ticker, horisont))
            svar = utfall[ticker]
            if isinstance(svar, Exception):
                raise svar
            return svar

        monkeypatch.setattr(analyze, "analysera", falsk)
        return anropade

    return installera


# --- Ett avbrott stoppar inte resten -----------------------------------------


def test_avbruten_ticker_stoppar_inte_de_ovriga(stub):
    anropade = stub(
        {
            "AAA": analyze.AnalysAvbruten("AVBRUTEN: AAA har 400.0 min gammal data"),
            "BBB": fejkresultat("BBB"),
            "CCC": DataFel("Tom datamangd"),
            "DDD": fejkresultat("DDD"),
        }
    )

    korningar = analyze.kor_alla(["AAA", "BBB", "CCC", "DDD"], 5)

    # Alla fyra ska ha forsokts, i ordning.
    assert [t for t, _ in anropade] == ["AAA", "BBB", "CCC", "DDD"]
    assert [k.lyckades for k in korningar] == [False, True, False, True]


def test_alla_avbrutna_ger_inga_lyckade(stub):
    stub({t: analyze.AnalysAvbruten(f"AVBRUTEN: {t}") for t in ("AAA", "BBB")})
    korningar = analyze.kor_alla(["AAA", "BBB"], 5)
    assert not any(k.lyckades for k in korningar)
    assert all(k.avbrottsorsak.startswith("AVBRUTEN") for k in korningar)


def test_indikatorfel_behandlas_som_avbrott_inte_som_bugg(stub):
    stub({"AAA": IndikatorFel("rsi(14) kraver minst 15 varden")})
    (korning,) = analyze.kor_alla(["AAA"], 5)
    assert not korning.lyckades
    assert not korning.ovantat


# --- Ovantade fel fangas men markeras ---------------------------------------


def test_ovantat_fel_stoppar_inte_resten_men_markeras(stub):
    stub({"AAA": KeyError("nyckel"), "BBB": fejkresultat("BBB")})
    korningar = analyze.kor_alla(["AAA", "BBB"], 5)

    assert korningar[0].ovantat is True
    assert "OVANTAT FEL" in korningar[0].avbrottsorsak
    assert "KeyError" in korningar[0].avbrottsorsak
    # ... och nasta ticker kordes anda.
    assert korningar[1].lyckades


def test_normalt_avbrott_markeras_inte_som_bugg(stub):
    stub({"AAA": analyze.AnalysAvbruten("AVBRUTEN: stangd bors")})
    (korning,) = analyze.kor_alla(["AAA"], 5)
    assert korning.ovantat is False


def test_torrkorning_skickas_vidare(monkeypatch):
    vidare = {}

    def falsk(ticker, horisont, *, skriv=True):
        vidare["skriv"] = skriv
        return fejkresultat(ticker)

    monkeypatch.setattr(analyze, "analysera", falsk)
    analyze.kor_alla(["AAA"], 5, skriv=False)
    assert vidare["skriv"] is False


# --- Sammanfattning ---------------------------------------------------------


def test_sammanfattning_raknar_ratt(stub):
    stub({"AAA": fejkresultat("AAA"), "BBB": analyze.AnalysAvbruten("AVBRUTEN: x")})
    text = analyze.formatera_sammanfattning(analyze.kor_alla(["AAA", "BBB"], 5), 5)
    assert "1/2 igenom" in text
    assert "OK" in text and "AVBRUTEN" in text


def test_sammanfattning_sager_ifran_nar_inget_gick_igenom(stub):
    stub({"AAA": analyze.AnalysAvbruten("AVBRUTEN: x")})
    text = analyze.formatera_sammanfattning(analyze.kor_alla(["AAA"], 5), 5)
    assert "Ingen prediktion loggad" in text


def test_sammanfattning_varnar_for_buggar(stub):
    stub({"AAA": ValueError("trasig")})
    text = analyze.formatera_sammanfattning(analyze.kor_alla(["AAA"], 5), 5)
    assert "VARNING" in text and "BUGG" in text


# --- Exitkoder --------------------------------------------------------------


def test_exitkod_noll_nar_minst_en_gick_igenom(stub, capsys, monkeypatch):
    from config import CONFIG

    monkeypatch.setitem(CONFIG, "BEVAKNINGSLISTA", ["AAA", "BBB"])
    stub({"AAA": analyze.AnalysAvbruten("AVBRUTEN: x"), "BBB": fejkresultat("BBB")})
    kod = analyze.main(["--alla", "--horisont", "5", "--torrkorning"])
    capsys.readouterr()
    assert kod == analyze.EXIT_OK


def test_exitkod_tre_nar_allt_avbrots(stub, capsys, monkeypatch):
    from config import CONFIG

    monkeypatch.setitem(CONFIG, "BEVAKNINGSLISTA", ["AAA", "BBB"])
    stub({t: analyze.AnalysAvbruten(f"AVBRUTEN: {t}") for t in ("AAA", "BBB")})
    kod = analyze.main(["--alla", "--horisont", "5", "--torrkorning"])
    capsys.readouterr()
    # 3 skiljs fran 2 sa att cron kan ignorera stangd bors men larma pa fel.
    assert kod == analyze.EXIT_ALLA_AVBRUTNA


def test_exitkod_tva_vid_bugg(stub, capsys, monkeypatch):
    from config import CONFIG

    monkeypatch.setitem(CONFIG, "BEVAKNINGSLISTA", ["AAA", "BBB"])
    stub({"AAA": TypeError("trasig"), "BBB": fejkresultat("BBB")})
    kod = analyze.main(["--alla", "--horisont", "5", "--torrkorning"])
    capsys.readouterr()
    assert kod == analyze.EXIT_AVBRUTEN


def test_alla_anvander_bevakningslistan_ur_config(stub, capsys, monkeypatch):
    from config import CONFIG

    monkeypatch.setitem(CONFIG, "BEVAKNINGSLISTA", ["XXX", "YYY", "ZZZ"])
    anropade = stub({t: analyze.AnalysAvbruten("AVBRUTEN") for t in ("XXX", "YYY", "ZZZ")})
    analyze.main(["--alla", "--horisont", "5", "--torrkorning"])
    capsys.readouterr()
    assert [t for t, _ in anropade] == ["XXX", "YYY", "ZZZ"]
