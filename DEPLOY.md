# DAS2-AI — deployment guide

Everything needed to get the system running in Docker, from nothing, and to
verify it against your real data before it is allowed to alert anyone.

Work through it in order. Each step either proves something or fails loudly;
the ordering exists so that when something does go wrong, you know which one
thing it was.

---

## 0. What you are deploying

An hourly job that reads the historian's CSV exports and answers one question:

> **Do we need to send someone out, or not?**

It does that by grouping abnormal sensors **by place and time**, and attaching
the context an operator needs to decide — neighbouring sites, equipment types,
local rainfall from your own gauges — so the decision can be made from a phone
instead of from a van.

Three things come out of every run:

| Output | Where | What it is for |
|---|---|---|
| `dashboard_<run>.html` | `./output` | The regional map, the region × type matrix, and every incident with its evidence |
| `map_*.png`, `matrix_*.png` | `./output` | The same, as images, because Telegram cannot render HTML |
| Telegram messages | your chat | One message **per incident**, with Acknowledge / Dispatched / False-alarm buttons |

Plus rows in SQL Server, so an incident keeps the same identity across runs.

---

## 1. Prerequisites

* Docker Engine 20.10+ and Docker Compose v2 (`docker compose version`)
* Network access from the container to your SQL Server, and outbound HTTPS to
  `api.telegram.org`
* Read access to the historian's export folder
* A Telegram bot token and a chat id (step 4)

You do **not** need internet on the machine for the dashboard to work. The map
falls back to a drawn-from-coordinates view when the tile CDN is unreachable.

---

## 2. Get the code and lay out the data

```bash
git clone <your-repo-url> das2-ai
cd das2-ai
git checkout claude/eloquent-knuth-54nxvy
```

The container expects the historian's export in **this shape**:

```
<your data folder>/
├── HISTORY/
│   ├── hts_2026_09_HISTORY_2026Sep20-210000.csv
│   ├── hts_2026_09_HISTORY_2026Sep20-220000.csv
│   └── ...                                   one file per hour
├── HISTCURR/
│   └── histcurr_fujitsu.csv                  the sensor inventory
└── LongLat.csv                               RTU coordinates
```

Three notes, each of which has already caused a failure:

* **`LongLat.csv` is not optional.** It is the only source of coordinates. With­
  out it there is no map, no geo-clustering and no "by region" view — the
  system still runs and still finds faults, but the thing you actually asked
  for is missing.
* **`HISTCURR` must be the live format** — semicolon-separated with nine
  columns (`ROW_ID;IPADDRESS;DESCRIPTION;TAGNAME;RTUNUMBER;RAWTYPE;POINTTYPE;DATETIME;CURRVALUE`)
  and US 12-hour timestamps like `9/20/2026 9:00:00 PM`. The older
  comma-separated six-column file will be rejected with a message saying so.
* If your export is laid out differently, do not move the files — point the
  paths at it in `.env` instead (step 3).

---

## 3. Configure

```bash
cp .env.example .env
```

Edit `.env`. The four things that must be set:

```bash
# Where your data is, on the host
DAS2_DATA_DIR=/mnt/scada-export          # or C:/das2/data on Windows

# Database
DAS2_DATABASE_URL=mssql+pyodbc://das2user:PASSWORD@sqlhost:1433/DAS2?driver=ODBC+Driver+18+for+SQL+Server&TrustServerCertificate=yes

# Telegram (step 4)
DAS2_ALERT_TELEGRAM_TOKEN=
DAS2_ALERT_TELEGRAM_CHAT_ID=
```

`.env` is gitignored. Keep it that way — it holds your database password and
bot token.

> **Want to try it without a database server first?** Set
> `DAS2_DATABASE_URL=sqlite:////data/output/das2.db` and skip nothing else.
> Everything works; the data just lands in a file under `./output`.

---

## 4. Get a Telegram bot token and chat id

1. In Telegram, message **@BotFather** → `/newbot` → follow the prompts.
   He replies with a token like `8012345678:AAH...`. That is
   `DAS2_ALERT_TELEGRAM_TOKEN`.
2. For the chat id:
   * **A private chat with you** — message **@userinfobot**; it replies with
     your numeric id.
   * **A group** (what you probably want, so the duty operator sees alerts) —
     add your bot to the group, post any message, then open
     `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates` in a browser and
     read `message.chat.id`. **A group id is negative**, e.g. `-1001234567890`;
     include the minus sign.
3. If the group has topics enabled, alerts go to the General topic unless you
   use the topic's own chat id.

> **Do not set a webhook on this bot.** The acknowledge buttons use
> `getUpdates` long-polling, which Telegram refuses (HTTP 409) while a webhook
> is registered. If you have set one, remove it with
> `https://api.telegram.org/bot<TOKEN>/deleteWebhook`. The worker detects this
> case and says so in the log rather than failing silently.

---

## 5. Build

```bash
docker compose build
```

Takes a few minutes the first time — it installs Microsoft's ODBC driver 18,
which is what `pyodbc` needs to reach SQL Server and which cannot come from
pip.

---

## 6. Create the database tables

```bash
docker compose run --rm das2-migrate
```

This applies `migrations/010_das2_schema.sql` and `011_sensor_kind.sql`. It is
**idempotent and additive**: safe to re-run, and it does not touch, drop or
alter any of the v1 tables (`dim`, `data`, `inst`, `linktable`, `dateDim`,
`alarmevent`, `abnormal_sensor_history`). Both systems can run side by side.

If your DBA wants to run the SQL by hand instead, the files are plain SQL and
can be pasted into SSMS — see [§12](#12-the-sql-in-full) for exactly what they
create and how to check it.

---

## 7. Prove the install works, before trusting it on your data

```bash
docker compose run --rm das2 python -m das2.cli demo
```

This generates synthetic data containing faults whose answers are known, runs
the full pipeline over it, and tells you whether it found them. It touches
nothing — no database write, no Telegram message.

Expect to end with:

```
====================================================================
OK — the injected regional event was found across BedokPS, BedokPond4, TampinesPS.
Ingest, classification, detection, clustering, triage and the
dashboard are all working. Point it at your real data next.
====================================================================
```

This step exists because "installed wrong" and "your network is healthy" look
identical from the outside — both produce no alerts. If the demo passes and
your real data produces nothing, that is a statement about your data, not about
the install.

---

## 8. Pre-flight against your real setup

```bash
docker compose run --rm das2-check
```

Checks each of the four things that actually go wrong, and prints PASS/FAIL
with the reason:

```
Input data
  [PASS] history directory exists  — /data/input/HISTORY
  [PASS] history CSVs present  — 72 file(s)
  [PASS] inventory file exists  — /data/input/HISTCURR/histcurr_fujitsu.csv
  [PASS] LongLat.csv exists  — /data/input/LongLat.csv

Database
  [PASS] connection  — mssql+pyodbc://das2user:***@sqlhost:1433/DAS2?...
  [PASS] das2 tables present  — das2_incident has 0 row(s)

Telegram
  [PASS] bot reachable  — @your_das2_bot
  [PASS] test message delivered to chat  — -1001234567890

Classifier
  [PASS] rules loaded  — 31 equipment classes
```

**A test message lands in your Telegram chat** if that part is configured —
that is the check, not a side effect.

Fix anything that says FAIL before continuing.

---

## 9. One real run, alerting nobody

```bash
docker compose run --rm das2-dryrun
```

Reads your real data, runs everything, writes a dashboard — and makes **no**
database write and **no** Telegram message. Open the HTML it names:

```bash
ls -t output/dashboard_*.html | head -1
```

**Spend time on this before going further.** You are looking for:

* Is the incident count sane, or is it hundreds? (Volume is the first thing
  that matters — a system that emits 400 alerts a day is dead whatever its
  accuracy.)
* Do the P1/P2 incidents look like things you would actually drive to?
* Does the region × type matrix match where you know your problem areas are?
* Is the coverage figure reasonable? The run prints
  `coverage: {... analog_coverage_pct: NN ...}` — that is the share of your
  analog sensors the classifier recognises.

Re-run it as often as you like; it changes nothing.

---

## 10. Go live

```bash
docker compose up -d das2 das2-ack
docker compose logs -f
```

Two containers now run:

* **`das2`** — analyses every `DAS2_RUN_INTERVAL_MINUTES` (default 60),
  persists, and alerts.
* **`das2-ack`** — long-polls Telegram and records the Acknowledge / Dispatched
  / False-alarm button presses against the incident.

> **Never run two `das2-ack` containers on one bot token.** Telegram hands each
> update to a single `getUpdates` caller, so a second instance silently steals
> half the button presses. Do not scale that service.

### Easing in

If you would rather watch it for a week before it pages anyone, set
`DAS2_ALERT_ENABLED=false` in `.env` and restart. It will run, persist and
write dashboards, and send nothing. Turn it on when the dashboards look right.

---

## 11. Day-to-day

```bash
# what is it doing
docker compose logs -f das2

# latest dashboard
ls -t output/dashboard_*.html | head -1

# force a run now
docker compose run --rm das2 python -m das2.cli run

# after a code update
git pull && docker compose build && docker compose up -d
docker compose run --rm das2-migrate      # idempotent; safe every time
```

### Reading the alerts

Every message leads with the decision:

```
🔴 P2 · REGIONAL_EVENT
Multiple sites affected together - investigate the area, not one sensor.

Where: East — BedokPS, BedokPond4, TampinesPS
Scale: 7 sensor(s) at 3 site(s), spread 2.5 km
When:  21 Sep 05:49 → 21 Sep 06:28 (39 min)
Rain nearby: 0.0 mm

Why:
• 7 sensors across 3 sites (BedokPS, BedokPond4, TampinesPS)
• 2 equipment types affected (Flowrate, Pressure) -- unlikely to be one
  instrument failing
• spread 2.5 km around East

[✅ Acknowledge] [🚚 Dispatched] [🔕 False alarm]
```

The seven incident classes and what each means:

| Class | What the evidence says | What to do |
|---|---|---|
| `REGIONAL_EVENT` | Several sites, several equipment types, same time | **Investigate the area** |
| `SENSOR_FAULT` | One instrument misbehaving, neighbours calm | **Dispatch a technician** |
| `WEATHER_DRIVEN` | Rain at nearby gauges explains it | Monitor — **do not dispatch** |
| `PROCESS_EVENT` | Neighbours moved together; the water moved | Operational, monitor |
| `DRIFT_MAINTENANCE` | Slow drift, no abrupt failure | Schedule recalibration |
| `TELEMETRY_FANOUT` | One RTU/panel, many sensors | Suppressed — never alerts |
| `WATCH` | Weak or conflicting evidence | Suppressed — re-evaluated next run |

The last two are **deliberately silent**. The dashboard shows them; nobody is
paged. The run summary always says how many were suppressed, so silence is
visibly a decision rather than a crash.

### Press the buttons

The Acknowledge / Dispatched / False-alarm taps are the **only source of
labelled ground truth this system has**, and they are what will let the
thresholds be tuned from evidence rather than from judgement. `Dispatched` and
`False alarm` are recorded in `das2_feedback` as `real` and `noise`. A month of
honest button-pressing is worth more than any amount of further tuning in the
dark.

---

## 12. The SQL in full

### What gets created

Applied by `docker compose run --rm das2-migrate`, or by pasting
`migrations/010_das2_schema.sql` then `011_sensor_kind.sql` into SSMS. Every
table is prefixed `das2_`, and **nothing existing is modified**.

| Table | Holds |
|---|---|
| `das2_sensor` | Inventory: description, equipment class, RTU, site, lat/lon, region, unit, whether it may alert |
| `das2_reading` | Time series, if you choose to persist it |
| `das2_sensor_profile` | Per-sensor time-of-day baseline, built by the daily job |
| `das2_detection_run` | One row per run: window, counts, duration, detector version |
| `das2_sensor_anomaly` | Per-sensor findings, **typed**, with severity in engineering units |
| `das2_incident` | The unit of alerting. Stable id across runs, class, priority, region, centroid, recommendation, ack state |
| `das2_incident_member` | Which sensors are in which incident |
| `das2_incident_event` | Lifecycle audit: opened / updated / resolved / acknowledged |
| `das2_neighbour_correlation` | Did the neighbours move too? |
| `das2_rain_observation` | Rainfall from your own gauges |
| `das2_alert_delivery` | What was sent, when, and what was suppressed and why |
| `das2_feedback` | Operator button presses — the ground truth |

`das2_detection_run.detector_version` exists so v1 and v2 can write to the same
table during a shadow comparison.

### Creating a dedicated login (recommended)

Run as a sysadmin, once. Replace the password.

```sql
-- On the instance
CREATE LOGIN das2user WITH PASSWORD = 'CHANGE-THIS-TO-SOMETHING-STRONG';
GO

-- In the target database
USE DAS2;          -- or whichever database you are using
GO
CREATE USER das2user FOR LOGIN das2user;
GO

-- Enough to create its own tables and use them, and nothing more.
ALTER ROLE db_datareader  ADD MEMBER das2user;
ALTER ROLE db_datawriter  ADD MEMBER das2user;
ALTER ROLE db_ddladmin    ADD MEMBER das2user;   -- needed for `migrate`
GO
```

If your DBA will not grant `db_ddladmin`, have them run the two migration files
by hand and then drop that role — the system only needs read and write at
runtime.

### Checking it worked

```sql
-- The tables exist
SELECT name FROM sys.tables WHERE name LIKE 'das2[_]%' ORDER BY name;

-- Runs are happening
SELECT TOP 10 run_id, started_at, window_start, window_end,
       sensors_analysed, anomalies_found, incidents_open, status
  FROM das2_detection_run
 ORDER BY started_at DESC;

-- What is open right now, worst first
SELECT incident_id, priority, incident_class, region, sensor_count, site_count,
       severity, recommendation, ack_state, last_seen_at
  FROM das2_incident
 WHERE status <> 'RESOLVED'
 ORDER BY severity DESC;

-- Which sensors are in an incident
SELECT i.incident_id, i.incident_class, s.description, s.equipment, s.site
  FROM das2_incident i
  JOIN das2_incident_member m ON m.incident_id = i.incident_id
  JOIN das2_sensor          s ON s.sensor_key  = m.sensor_key
 WHERE i.incident_id = 'East-20260921-344ef706';

-- Alert volume per day -- the number that decides whether this is usable
SELECT CAST(sent_at AS DATE) AS day,
       SUM(CASE WHEN suppressed = 0 THEN 1 ELSE 0 END) AS sent,
       SUM(CASE WHEN suppressed = 1 THEN 1 ELSE 0 END) AS suppressed
  FROM das2_alert_delivery
 GROUP BY CAST(sent_at AS DATE)
 ORDER BY day DESC;

-- Operator feedback: your ground truth
SELECT label, COUNT(*) AS n
  FROM das2_feedback
 GROUP BY label;

-- Coverage: how much of the fleet is classified and alertable
SELECT equipment, COUNT(*) AS sensors,
       SUM(CASE WHEN alertable = 1 THEN 1 ELSE 0 END) AS alertable
  FROM das2_sensor
 GROUP BY equipment
 ORDER BY sensors DESC;
```

---

## 13. When something goes wrong

| Symptom | Cause | Fix |
|---|---|---|
| `Can't open lib 'ODBC Driver 18 for SQL Server'` | Driver missing — you are not running in the image | Use the container, or install `msodbcsql18` on the host |
| `Login failed for user` | Credentials, or the login has no user in that database | Re-check `.env`; run the `CREATE USER` in §12 |
| `no such table: das2_incident` | Migrations not applied | `docker compose run --rm das2-migrate` |
| `HISTCURR missing ['IPADDRESS', ...]` | Old comma-separated inventory file | Supply the live semicolon 9-column export (§2) |
| `no readings were analysed` | `DAS2_DATA_DIR` points at the wrong folder, or no files in the window | `docker compose run --rm das2-check` |
| Telegram `HTTP 409` in the ack worker | A webhook is set on the bot | `https://api.telegram.org/bot<TOKEN>/deleteWebhook` |
| Alerts arrive, buttons do nothing | `das2-ack` not running, or two of them are | `docker compose up -d das2-ack`; never run two |
| Map empty on the dashboard | `LongLat.csv` missing, or RTU numbers not joining | Check `coordinate_coverage_pct` in the run output |
| Dashboard shows tables but no map | Tile CDN unreachable | Expected — it falls back to a drawn map; nothing is lost but the basemap |
| Far too many alerts | Thresholds not yet tuned to your fleet | Set `DAS2_ALERT_ENABLED=false`, read dashboards for a week, then tune |

Turn up detail with `DAS2_LOG_LEVEL=DEBUG` in `.env` and restart.

---

## 14. What is and is not finished

Be aware of this before judging the output.

**Working and verified end to end:** ingest of the real formats · full-coverage
equipment classification · per-sensor profiles · FLATLINE, STALE,
RANGE_VIOLATION, SPIKE, REVERSE_FLOW, LEVEL_SHIFT · typed fusion ·
spatio-temporal clustering with parameters measured from your real
`LongLat.csv` · incident identity across runs · triage and recommendations ·
rain context from your own gauges · dashboard · charts · Telegram with
acknowledgement · persistence.

**Not yet built**, and therefore producing no findings:

* **DRIFT** and **NOISE_BURST** — these need ≥14 days of history and a daily
  profile job. They cannot be computed inside a 72-hour window: a 1%/day drift
  is 3% across the window, while the daily demand cycle is routinely 10–30%.
* **Digital detectors** — short-cycling, stuck-in-state, run/flow consistency.
  Roughly 1,567 pump and valve sensors are currently analysed by the analog
  stack, which produces little of value for them.
* **Mass balance** — `dLevel/dt·A ≈ Qin − Qout`, which localises *which* of
  three instruments is lying.
* **Neighbour correlation** — the field is in the schema and the alert, but is
  not yet computed, so it reads as "not available". This is the single most
  decision-relevant signal still outstanding: neighbours moving together means
  the water moved; neighbours flat means the instrument is lying.
* **Quantisation collapse** and **dithering-dead**.

**The weakest input in the system** is the per-equipment physical range limits.
They are fleet-wide defaults — "Pressure 0–20" applied to every pressure sensor
in Singapore — which is close to meaningless when your real sensors have
medians of 0.0034 and 3.90. They are deadbanded against each sensor's own noise
so they do not produce nonsense, but the right fix is your own data:

> **If Fujitsu can export the configured per-point HH/H/L/LL alarm limits and
> engineering ranges, that single file replaces about ninety hand-chosen
> thresholds with your plant's own commissioned values.** It is the
> highest-value, lowest-effort improvement available, and worth asking for now.
