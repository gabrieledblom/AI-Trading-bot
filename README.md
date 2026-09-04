# Aktieanalysbot

Analyserar en aktie, loggar en prediktion med tidsstampel, och utvarderar i
efterhand om prediktionerna var battre an att bara kanna till aktiens basrat.

Systemet ar byggt for att kunna underkanna sig sjalvt. Huvudmattet ar inte
traffsakerhet utan **edge** -- traffsakerhet minus basrat. En modell som traffar
62 procent pa en aktie som gar upp 62 procent av tiden har noll edge.

## Filer

| Fil | Ansvar |
|---|---|
| `config.yaml` | All konfiguration. Ingen modul far hardkoda nagot som star har. |
| `config.py` | Laser och validerar `config.yaml`. Kastar vid saknade nycklar. |
| `data.py` | `hamta()` och `farskhetskontroll()`. Kastar vid saknad tidsstampel. |
| `indicators.py` | Alla indikatorer beraknade fran grunden, plus `basrat()`. |
| `analyze.py` | Kor kedjan och loggar en rad i `predictions.csv`. |
| `evaluate.py` | Fyller i utfall och rapporterar 7 matt. |
| `predictions_schema.py` | Radformatet i `predictions.csv`. |
| `tests/` | Tester for farskhetskontroll och basrat. |

## Anvandning

```bash
pip install -r requirements.txt
python -m pytest tests -q

python analyze.py --ticker NVDA --horisont 5
python analyze.py --alla --horisont 5              # hela BEVAKNINGSLISTA
python analyze.py --alla --horisont 20 --torrkorning

python evaluate.py                 # fyll i mogna utfall och rapportera
python evaluate.py --bara-rapport  # rapportera utan att hamta ny data
```

### Exitkoder

| Kod | Betydelse |
|---|---|
| 0 | Minst en prediktion loggad. |
| 2 | Enskild ticker avbruten, eller ovantat fel under `--alla`. |
| 3 | `--alla`: allt avbrots. Normalt utanfor borstid -- kan ignoreras i cron. |

Med `--alla` stoppar ett avbrott bara den tickern. Stangd bors i Stockholm
sager ingenting om Nasdaq, sa resten av listan kors anda.

## Regler som inte far brytas

1. **Ingen tyst fallback.** Saknas en tidsstampel kastas ett fel. Ingen
   indikator ersatts med noll eller ett neutralt varde nar underlaget saknas --
   den utelamnas.
2. **Farskhetskontrollen ar en sparr, inte en varning.** `analyze.py` vagrar
   kora och skriver ingen rad om latensen overskrider `MAX_LATENS_MINUTER`.
3. **Basraten ar referenspunkten.** Varje prediktion loggar basraten for sin
   riktning. Utvarderingen mater mot den, aldrig mot 50 procent.
4. **Under 100 utvarderade prediktioner ar resultatet brus** och rapporten
   sager det rakt ut.
5. **Ovantade fel maskeras inte som avbrott.** `--alla` fangar dem sa att
   listan kan koras klart, men markerar dem `BUGG` med varning i
   sammanfattningen. En bugg ar inte en stangd bors.

## Om BZ=F (Brent)

Brent ligger i bevakningslistan som ravarureferens, men skiljer sig fran
aktierna pa tre satt som paverkar tolkningen:

- Serien ar ett *kontinuerligt* framre terminskontrakt. Vid varje rollover
  hoppar priset av kontraktstekniska skal. `basrat()` raknar de hoppen som
  riktig avkastning, sa basraten for BZ=F ar nagot forstord.
- Terminer handlas nastan dygnet runt. Brent passerar darfor
  farskhetskontrollen nastan alltid, medan aktierna gor det bara under
  borstid. Vid schemalagd korning blir predictions.csv oljetung.
- BZ=F gar inte att kopa i en vanlig depa. Certifikat och ETF:er som foljer
  olja drabbas av contango och foljer inte spotpriset over tid.

## Oklarheter

`CLAUDE.md` fanns inte i repot nar koden skrevs (GitHub svarade
`409 Git Repository is empty`). Tva saker ar darfor mina egna val och ska
granskas mot `CLAUDE.md` nar den finns:

- **Indikatoruppsattningen** i `indicators.py`. `basrat()` ar daremot
  specificerad direkt i uppdraget och ar inte provisorisk.
- **`bedomning()` och outputblockets format** i `analyze.py`. Vikterna i
  `TILTVIKTER` ar gissningar som inte ar anpassade mot nagon data.

Bada ar markerade med `PROVISORISK` i koden och ligger isolerade sa att de gar
att byta ut utan att rora resten.
