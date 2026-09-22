# DAS2-AI

Water-sensor anomaly intelligence for PUB's SCADA telemetry.

Reads the historian's hourly CSV exports and answers one question:

> **Do we need to send someone out, or not?**

It groups abnormal sensors **by place and by time**, then attaches the context
needed to decide — which sites, which equipment types, whether the neighbours
reacted, whether it was raining there — so the call can be made from a phone
instead of from a van.

## Getting started

**[DEPLOY.md](DEPLOY.md) is the guide.** It covers Docker, the database, the
Telegram bot, and how to verify the whole thing against real data before it is
allowed to alert anyone.

The short version:

```bash
cp .env.example .env && $EDITOR .env      # data path, DB, bot token
docker compose build
docker compose run --rm das2-migrate      # create the tables
docker compose run --rm das2 python -m das2.cli demo   # prove it works
docker compose run --rm das2-check        # check YOUR setup
docker compose run --rm das2-dryrun       # a real run, alerting nobody
docker compose up -d das2 das2-ack        # go live
```

## What it produces

| | |
|---|---|
| `output/dashboard_*.html` | Regional map, region × equipment-type heatmap, every incident with the evidence behind its recommendation. Self-contained — email it, archive it, open it from a share. |
| `output/*.png` | The same as images, for Telegram |
| Telegram | One message **per incident**, not per sensor per run, with Acknowledge / Dispatched / False-alarm buttons |
| SQL Server | Incidents with stable identity across runs, plus the operator feedback that is the only ground truth this system has |

## How it works

```
ingest → classify → profile → detect → fuse → cluster → triage → alert
```

Each stage narrows the question. The detectors ask *is this number wrong?*,
fusion asks *what is wrong with this sensor?*, clustering asks *did several
places go wrong together?*, and triage asks the only question that matters:
*do we drive there?*

Two verdicts are opposites, and telling them apart is most of the value:

| Shape | Verdict |
|---|---|
| Many sensors, **one** site, similar names | Telemetry fan-out — **suppressed**, nobody drives anywhere |
| Several sensors, **several** sites, same time | Regional event — **investigate the area** |

The second case is invisible to name-based grouping, which is what the previous
system used: four different sensors at three different sites share no name, so
nothing linked them.

## Development

```bash
bash tests/run_all.sh              # 579 assertions, no network needed
python3 -m das2.cli demo           # synthetic data with known faults
python3 tools/make_fixtures.py --out /tmp/fx
```

Tests need no database, no bot token and no internet. Every parameter that
could have been guessed was instead measured against the client's real data,
and the modules say which measurement and what it ruled out.

## Status

Working end to end. `DEPLOY.md` §14 lists precisely what is built and what is
not — read it before judging the output, because several detectors (DRIFT,
NOISE_BURST, the digital-equipment set, mass balance, neighbour correlation)
are not yet implemented and therefore produce no findings.
