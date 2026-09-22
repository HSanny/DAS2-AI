# DAS2-AI — first deployment, start to finish

Every command, in order, from an empty machine. `DEPLOY.md` explains the *why*;
this page is just the *what*.

Nothing here is reversible-by-accident except **step 5**, which is marked.

---

## 0. Host prerequisites

```bash
docker --version                 # 20.10+
docker compose version           # v2
sudo apt install cifs-utils      # REQUIRED — the daemon mounts the share, not the container
```

Without `cifs-utils` the volumes fail to mount and you get an empty input
directory, which looks like a quiet network rather than an error. Step 7
catches it, but installing it now saves the round trip.

---

## 1. Get the code

```bash
git clone <your-repo-url> das2-ai
cd das2-ai
git checkout claude/eloquent-knuth-54nxvy
git log --oneline -1
```

The last line should print `d314c13` or later.

---

## 2. Put LongLat.csv where the container can see it

It is **not** on the share — it is a one-off export, not an hourly one.

```bash
mkdir -p config output logs
cp /path/to/LongLat.csv ./config/
ls -l config/LongLat.csv
```

Without it the system runs and finds faults, but there is no map, no
geo-clustering and no by-region view — most of the point. Step 7 reports its
absence as a FAIL rather than letting it slide.

---

## 3. Write `.env`

```bash
cp .env.example .env
nano .env
```

Six values must be filled in. Everything else in the file has a working
default — read it later, not now.

```bash
# --- the CIFS share (same names as the previous deployment) ---
DAS_VM_USERNAME=your_share_user
DAS_VM_PASSWORD=your_share_password
DAS_MSSQL_HOST=10.0.0.5                # the box serving dds_share

# --- database ---
DAS2_DATABASE_URL=mssql+pyodbc://das2user:PASSWORD@10.0.0.5:1433/anomaly_db?driver=ODBC+Driver+18+for+SQL+Server&TrustServerCertificate=yes

# --- telegram ---
DAS2_ALERT_TELEGRAM_TOKEN=8012345678:AAH...
DAS2_ALERT_TELEGRAM_CHAT_ID=-1001234567890
```

Two more worth setting for the first week:

```bash
DAS2_ALERT_ENABLED=false               # runs, persists, writes dashboards, sends NOTHING
DAS2_LOG_LEVEL=DEBUG                   # turn back to INFO once it is behaving
```

### Getting the Telegram values

| | |
|---|---|
| **Token** | Message **@BotFather** → `/newbot` → follow the prompts |
| **Chat id, private** | Message **@userinfobot**; it replies with your numeric id |
| **Chat id, group** | Add the bot to the group, post a message, open `https://api.telegram.org/bot<TOKEN>/getUpdates`, read `message.chat.id` |

A **group id is negative** — `-1001234567890`. Keep the minus sign.

**Do not set a webhook on this bot.** The acknowledge buttons use
`getUpdates` long-polling, which Telegram refuses with HTTP 409 while a webhook
is registered. If one is already set:
`https://api.telegram.org/bot<TOKEN>/deleteWebhook`

---

## 4. Build

```bash
docker compose build
```

Several minutes the first time — it installs Microsoft's ODBC driver 18, which
`pyodbc` needs and pip cannot supply.

---

## 5. Clear the old tables — DESTRUCTIVE

`anomaly_db` holds ten `das2_*` tables from the previous system. One name,
`das2_feedback`, collides outright with the new schema.

**Optional, one statement, gone forever if you skip it.** `das2_feedback` holds
operator labels — the only ground truth either system has:

```sql
SELECT * INTO dbo.legacy_feedback_backup FROM dbo.das2_feedback;
```

Then paste **`tools/drop_legacy_das2.sql`** into SSMS against `anomaly_db`. It
ends with a `SELECT` that must return **zero rows** before you continue.

---

## 6. Create the new schema

```bash
docker compose run --rm das2-migrate
```

**Order matters.** Migrating before step 5 leaves the old `das2_feedback` in
place — `CREATE TABLE IF NOT EXISTS` skips it — and the new code then fails at
runtime inserting 7 columns into a 10-column table.

---

## 7. Prove the install, before trusting your data

```bash
docker compose run --rm das2 python -m das2.cli demo
```

Runs against synthetic data with known faults. Touches nothing. Must end:

```
OK — the injected regional event was found across BedokPS, BedokPond4, TampinesPS.
```

If this fails, the problem is the install. If it passes and your real data is
quiet, the problem is not the install — which is exactly the distinction this
step exists to make.

---

## 8. Check your own setup

```bash
docker compose run --rm das2-check
```

Expect:

```
Input data
  [PASS] history directory exists
  [PASS] history directory is not empty  — 72 entr(ies)
  [PASS] history CSVs present  — 72 file(s)
  [PASS] the feed is current  — newest file is 0.4 h old
  [PASS] inventory file exists
  [PASS] LongLat.csv exists

Database
  [PASS] connection
  [PASS] das2 tables present  — das2_incident has 0 row(s)

Telegram
  [PASS] bot reachable  — @your_das2_bot
  [PASS] test message delivered to chat
```

**A real message arrives in your Telegram chat.** That is the check.

Fix every FAIL before going on. The one to expect:

```
[FAIL] history directory is not empty — the directory exists but contains NOTHING.
```

That is a CIFS mount problem, not a missing folder — wrong credentials, wrong
share path, or `cifs-utils` missing. Look with:

```bash
docker compose run --rm das2 ls -la /data/input/HISTORY
```

---

## 9. One real run, alerting nobody

```bash
docker compose run --rm das2-dryrun
ls -t output/dashboard_*.html | head -1
```

No database write, no Telegram. Open the dashboard. **Stop here and look**
before going further. Re-run as often as you like; it changes nothing.

Three numbers in the run output are worth reading:

| Line | What to look for |
|---|---|
| `coverage:` | `analog_coverage_pct` — how much of your fleet the classifier recognises |
| `detection:` | anomaly count. Hundreds means thresholds need tuning for your fleet |
| `incidents:` | `alertable` vs `suppressed` — what would actually page someone |

---

## 10. Go live

```bash
docker compose up -d das2 das2-ack
docker compose logs -f
```

- **`das2`** — analyses every 60 min, persists, alerts.
- **`das2-ack`** — consumes the Acknowledge / Dispatched / False-alarm buttons.

**Never run two `das2-ack` containers on one bot token.** Telegram gives each
update to a single poller; a second instance silently eats half the presses.

If you set `DAS2_ALERT_ENABLED=false` in step 3, it is running silently. Flip it
to `true` and `docker compose up -d das2` when the dashboards look right.

---

## 11. Schedule the daily job

Once a day, via cron or Task Scheduler:

```bash
cd /path/to/das2-ai && docker compose run --rm das2-profile
```

Crontab example — 03:30 daily:

```
30 3 * * * cd /opt/das2-ai && /usr/bin/docker compose run --rm das2-profile >> /opt/das2-ai/logs/profile.log 2>&1
```

**Not optional.** `DRIFT`, `NOISE_BURST` and the whole L2 baseline layer produce
nothing until it has run with enough history.

---

## Two things that look broken and are not

**`das2 profile` says "No history yet"** on day one. Correct. `das2_reading`
starts empty and the hourly run fills it. `DRIFT` needs 14 days, `NOISE_BURST`
7, the baselines a few.

**`baselines: {'sensors': 0, 'usable': 0}`** in the hourly run, for the same
reason. It disappears once the daily job has run against real history.

---

## Day-to-day

```bash
docker compose logs -f das2                          # what it is doing
ls -t output/dashboard_*.html | head -1              # latest dashboard
docker compose run --rm das2 python -m das2.cli run  # force a run now
docker compose restart das2                          # after editing .env

# after a code update
git pull && docker compose build && docker compose up -d
docker compose run --rm das2-migrate                 # idempotent, safe every time
```

Editing `.env` needs a **restart**, not a rebuild. Editing code needs a
**rebuild** — the image bakes the source in.

---

## If something fails

| Symptom | Cause | Fix |
|---|---|---|
| `Can't open lib 'ODBC Driver 18'` | Running outside the container | Use `docker compose run`, not bare python |
| Input directory exists but empty | CIFS mount failed | `cifs-utils` on host; check credentials and share path |
| `Permission denied` reading the share | Container runs as uid 10001 | Add `uid=10001,gid=10001` to the mount options |
| `Login failed for user` | DB credentials, or no user in that database | Re-check `.env`; see `DEPLOY.md` §12 for the `CREATE USER` |
| `no such table: das2_incident` | Migrations not applied | `docker compose run --rm das2-migrate` |
| Telegram `HTTP 409` | A webhook is set on the bot | `.../deleteWebhook` |
| Buttons do nothing | `das2-ack` not running, or two of them | `docker compose up -d das2-ack` |
| Map empty on the dashboard | `LongLat.csv` missing, or RTUs not joining | Check `coordinate_coverage_pct` in the run output |
| Far too many alerts | Thresholds not yet tuned to your fleet | `DAS2_ALERT_ENABLED=false`, read dashboards, then tune |

More detail on any of these: `DEPLOY.md` §13.
