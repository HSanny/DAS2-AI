# DAS2-AI

Water-sensor anomaly intelligence for PUB's SCADA telemetry.

Reads the historian's hourly CSV exports and answers one question:

> **Do we need to send someone out, or not?**

It groups abnormal sensors **by place and by time**, then attaches the context
needed to decide — which sites, which equipment types, whether the neighbours
reacted, whether it was raining there — so the call can be made from a phone
instead of from a van.

## Getting started

**[RUNBOOK.md](RUNBOOK.md) — every command for a first deployment, in order.**
Follow that one.

[DEPLOY.md](DEPLOY.md) is the reference behind it: why each step exists, the
full SQL, and what is and is not built yet.

The shape of it:

```bash
sudo apt install cifs-utils               # the daemon mounts the share
cp .env.example .env && $EDITOR .env      # share creds, DB, bot token
cp /path/to/LongLat.csv ./config/         # not on the share; no map without it
docker compose build
# paste tools/drop_legacy_das2.sql into SSMS   <- DESTRUCTIVE, before migrate
docker compose run --rm das2-migrate
docker compose run --rm das2 python -m das2.cli demo   # prove the install
docker compose run --rm das2-check        # prove YOUR setup
docker compose run --rm das2-dryrun       # real data, alerting nobody
docker compose up -d das2 das2-ack        # go live
# then schedule `docker compose run --rm das2-profile` daily
```

## What it produces

| | |
|---|---|
| `output/dashboard_*.html` | The interactive report. Regional map, region × parameter heatmap, and every incident expanding to its evidence, its sensors' own traces, and what a median ± 3σ check would have made of them. Filter by region, priority, class or parameter. Self-contained — email it, archive it, open it from a share, with no server behind it. |
| `output/*.png` | The same as images, for Telegram |
| Telegram | Two notifications per run — the map of Singapore, then the PDF analysis and the interactive page together as one album. Per-incident messages with Acknowledge / Dispatched / False-alarm buttons are opt-in (`alert.p1_detail_messages`) |
| SQL Server | Incidents with stable identity across runs, the reading history the daily job learns from, plus the operator feedback that is the only ground truth this system has |

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

### Checking it against the statistics you already run

Every incident carries what a median-and-σ check would have concluded about
the same sensors, computed on the same data — including the cases where it
would have caught the event and this system added nothing. Two numbers are
worth knowing before reading it:

- **A 3σ band does not filter much at this sample count.** The probability
  that pure Gaussian noise touches 3σ somewhere in a 900-sample window is
  **91%**. Across 2,600 sensors that is around 2,370 healthy sensors crossing
  the line every run, so each crossing is scored against what noise produces
  and reported as a find only when noise does not explain it.
- **σ is computed from the window that contains the event.** A sustained
  excursion inflates its own denominator — the masking effect that robust
  statistics exist to deal with. Where that happens, the report gives both
  numbers: the score against the whole window, and the score against the hours
  before the event. On the fixture a real step reads 1.8σ one way and 17.7σ
  the other.

### Two files you are meant to edit

Both live in `das2/data/` and are read at start-up, so a change is an edit and
a restart, not a code change.

| | |
|---|---|
| `equipment_rules.yaml` | What each sensor **is**, matched from its description, and which classes may raise an alert |
| `event_signatures.yaml` | What a set of moving parameters usually **means** — "levels and flows rose together while it was raining" — in your engineers' words |

A signature only ever *describes* an incident. It cannot change the class, the
priority or the recommendation, and it never suppresses an alert: a wrong
evidence class costs a wasted trip, a wrong reading able to stop a dispatch
costs a flood. Every entry carries what would prove it wrong, and the report
prints that line underneath it. Correcting one is the ground truth this system
does not otherwise have.

## Development

```bash
bash tests/run_all.sh              # 1079 assertions, no network needed
python3 -m das2.cli demo           # synthetic data with known faults
python3 tools/make_fixtures.py --out /tmp/fx
```

Tests need no database, no bot token and no internet. Every parameter that
could have been guessed was instead measured against the client's real data,
and the modules say which measurement and what it ruled out.

## Status

Working end to end, with all fifteen detectors built. `DEPLOY.md` §14 lists
what each one does and where it runs.

Two things to know before judging the output. The **daily profile job**
(`das2 profile`) must be scheduled: `DRIFT`, `NOISE_BURST` and the whole
baseline layer depend on it, and until it has run those produce nothing. And
there is **no shadow-mode harness yet**, so there is no measured precision or
recall against real data — the evidence is 1079 assertions and a fixture
carrying 19 known faults, which is real but is not the same claim.
