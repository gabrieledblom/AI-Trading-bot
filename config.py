"""Inlasning av config.yaml. Enda stallet dar konfiguration far lasas."""

from __future__ import annotations

from pathlib import Path

import yaml

ROT = Path(__file__).resolve().parent
CONFIG_FIL = ROT / "config.yaml"

_OBLIGATORISKA = (
    "MAX_LATENS_MINUTER",
    "HORISONTER",
    "BEVAKNINGSLISTA",
    "DATAKALLA",
    "BENCHMARK_INDEX",
    "INTERVALL",
    "BASRAT_AR",
    "MIN_ANTAL_FOR_SLUTSATS",
    "LATENSINTERVALL",
    "KONFIDENSINTERVALL",
    "PREDIKTIONSFIL",
)


class Konfigurationsfel(Exception):
    """Kastas nar config.yaml saknas eller ar ofullstandig."""


def las_config(sokvag: Path | None = None) -> dict:
    """Laser och validerar config.yaml. Kastar hellre an att gissa defaultvarden."""
    sokvag = Path(sokvag) if sokvag is not None else CONFIG_FIL
    if not sokvag.exists():
        raise Konfigurationsfel(f"config.yaml saknas: {sokvag}")

    with sokvag.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    if not isinstance(data, dict):
        raise Konfigurationsfel(f"config.yaml ar inte en mappning: {sokvag}")

    saknade = [n for n in _OBLIGATORISKA if n not in data]
    if saknade:
        raise Konfigurationsfel(
            "config.yaml saknar obligatoriska nycklar: " + ", ".join(saknade)
        )

    return data


CONFIG = las_config()
PREDIKTIONSFIL = ROT / CONFIG["PREDIKTIONSFIL"]
