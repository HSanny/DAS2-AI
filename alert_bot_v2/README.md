# alert_bot build folder

This folder contains everything needed to build the `alert_bot` Docker image.

## Files

| File | Purpose |
|---|---|
| `Dockerfile` | Build recipe (Python 3.13.2 + ODBC Driver 17 + dependencies) |
| `alert_bot_v2.py` | Main bot script — sends Telegram alerts and the new heartbeat |
| `requirements.txt` | Python deps: requests, sqlalchemy, python-dotenv, pyodbc |
| `.dockerignore` | Keeps junk out of the image |

## One-time deploy (after the folder is on the host)

```bash
# 1) Build the new image
cd <path-to-this-folder>
docker build -t alert_bot:1.1 .

# 2) Tag the old image as a rollback (so we can revert in seconds)
docker tag alert_bot:1.0 alert_bot:rollback   # safe to skip if 1.0 doesn't exist

# 3) Edit docker-compose.yml in the main project folder:
#      image: alert_bot:1.0   ->   image: alert_bot:1.1

# 4) Recreate the alert_bot container with the new image
docker compose up -d --force-recreate alert_bot

# 5) Watch the startup logs to confirm it's healthy
docker logs -f alert_bot
```

## What you should see in the alert_bot startup log

The patched bot logs the heartbeat config on startup. Look for:

```
🚀 Alert bot started
Heartbeat enabled: grace=5min, tracker=logs/heartbeat_sent_runs.txt
```

If you see those two lines, the new build is running.

## How to verify the heartbeat works

The heartbeat fires when a SnapshotRunId is at least 5 minutes old AND has
zero alertable rows (no Plot_Path values). To test:

- Wait for the next scheduled run (00:00 / 06:00 / 12:00 / 18:00 SGT)
- Or, if no anomalies are detected at all, the heartbeat will fire ~5 min later
- Or, if all detected anomalies are suppressed, the heartbeat will fire
  ~5 min later with an "all suppressed as fan-out" variant message

## Rollback (if anything goes wrong)

```bash
# Flip back to the old image
# Edit docker-compose.yml: image: alert_bot:1.1 -> image: alert_bot:rollback
docker compose up -d --force-recreate alert_bot
```

## Tunables (env vars, all optional)

Add to docker-compose.yml under `alert_bot.environment` if you want to override:

| Env var | Default | What it does |
|---|---|---|
| `HEARTBEAT_ENABLED` | `1` | Set to `0` to disable heartbeats |
| `HEARTBEAT_GRACE_MINUTES` | `5` | Wait this long after the last DB write for a run before deciding it's complete |
| `HEARTBEAT_TRACKER_FILE` | `logs/heartbeat_sent_runs.txt` | Where the "already sent" list lives |
| `HEARTBEAT_MAX_RUNS_LOOKBACK` | `10` | How many recent runs to check each loop |

## Important: prerequisites for heartbeat to actually work

1. `analysis_bot` must be writing to the DB (no `SKIP_DB_WRITE=1`).
2. The `abnormal_sensor_history` table should have the `Suppressed` and
   `Suppression_Reason` columns. If not, run:

   ```sql
   ALTER TABLE dbo.abnormal_sensor_history
     ADD Suppressed BIT NULL,
         Suppression_Reason NVARCHAR(500) NULL;
   ```

   The heartbeat works without these columns (it falls back gracefully), but
   the "all suppressed" message variant won't say so — it'll just say "no plots".

## What the heartbeat message looks like

For a run with 0 anomalies:
```
✅ Anomaly analysis complete
• Run: 20260515_1200
• 0 anomalies detected.
• Nothing to alert this run.
```

For a run where all anomalies were suppressed as fan-out:
```
✅ Anomaly analysis complete
• Run: 20260515_0614
• 10 detected, all suppressed as fan-out.
• Nothing to alert this run.
```
