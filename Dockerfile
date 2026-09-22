# DAS2-AI — water sensor anomaly intelligence
# ===========================================================================
# One image, three roles, selected by the command:
#
#   python -m das2.cli schedule      the analysis loop
#   python -m das2.cli ack-worker    consumes Telegram acknowledge buttons
#   python -m das2.cli check         pre-flight, run this first
#
# One image rather than three keeps the analysis and the acknowledgement worker
# provably on the same code, which matters because they share the incident ids
# that connect an alert to the button press that answers it.

FROM python:3.11-slim

# --- system packages -------------------------------------------------------
# msodbcsql18 is the Microsoft ODBC driver that pyodbc needs to reach SQL
# Server. It is not on PyPI and cannot be installed by pip, which is the usual
# reason a container that works locally fails in production with
# "Can't open lib 'ODBC Driver 18 for SQL Server'".
#
# ACCEPT_EULA=Y is required by Microsoft's package and the install hangs
# waiting for input without it.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl gnupg ca-certificates tzdata \
    && curl -fsSL https://packages.microsoft.com/keys/microsoft.asc \
        | gpg --dearmor -o /usr/share/keyrings/microsoft-prod.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/microsoft-prod.gpg] \
https://packages.microsoft.com/debian/12/prod bookworm main" \
        > /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y --no-install-recommends \
        msodbcsql18 unixodbc-dev gcc g++ \
    && apt-get purge -y curl gnupg \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Singapore time. Every timestamp in an alert is read by someone standing in
# Singapore, and a container defaulting to UTC makes "03:00" mean 11:00 to the
# person deciding whether to send a crew out.
ENV TZ=Asia/Singapore
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

WORKDIR /app

# Dependencies before source, so editing code does not reinstall the world.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY das2/ ./das2/
COPY migrations/ ./migrations/
COPY tools/ ./tools/
COPY tests/ ./tests/

# matplotlib writes a font cache on first use; without a writable home it
# warns on every run and rebuilds the cache each time.
ENV MPLCONFIGDIR=/tmp/matplotlib \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Non-root. The volumes it needs to write are created and chowned here; the
# input directory is mounted read-only, because this system has no business
# modifying the historian's export.
RUN useradd --create-home --uid 10001 das2 \
    && mkdir -p /data/output /data/logs \
    && chown -R das2:das2 /app /data
USER das2

# Reports the run loop is alive. A scheduler that has silently died looks
# exactly like a quiet network from the outside, which is the failure mode
# worth catching early.
HEALTHCHECK --interval=5m --timeout=30s --start-period=2m --retries=3 \
    CMD python -c "import sys,os,time; \
p='/data/logs/heartbeat'; \
sys.exit(0 if os.path.exists(p) and time.time()-os.path.getmtime(p) < 7200 else 1)"

CMD ["python", "-m", "das2.cli", "schedule"]
