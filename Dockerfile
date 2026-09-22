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
#
# libgssapi-krb5-2 IS NOT OPTIONAL AND IS NOT REDUNDANT. Read this before
# removing it to tidy the list.
#
#   $ readelf -d libmsodbcsql-18.7.so.1.1 | grep NEEDED
#     ... libodbcinst.so.2, libkrb5.so.3, libgssapi_krb5.so.2, ...
#
# The driver links against libgssapi_krb5.so.2, but Microsoft's package
# declares only `libkrb5-3` in its Depends -- and libkrb5-3 does not depend on
# libgssapi-krb5-2 (it is the other way round). So nothing in the image
# declares a need for it. It used to arrive by accident, as a dependency of
# libcurl4, pulled in by the `curl` above; `apt-get autoremove` after purging
# curl then removed it again, since by then no installed package claimed it.
#
# The result was an image that built cleanly and could never connect, failing
# at the first query with a message that points at the wrong file entirely:
#
#   Can't open lib '/opt/microsoft/msodbcsql18/lib64/libmsodbcsql-18.7.so.1.1'
#   : file not found
#
# That file is present. unixODBC reports "file not found" whenever dlopen()
# fails for any reason, including an unresolved dependency of the library it
# was asked to open, so the name in the error is the one thing that is NOT
# missing. Naming the package explicitly makes it manually-installed and so
# beyond autoremove's reach.
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
        libgssapi-krb5-2 libkrb5-3 \
    && apt-get purge -y curl gnupg \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# --- let TLS reach an older SQL Server -------------------------------------
# Debian 12 ships OpenSSL 3 at SECLEVEL=2 with MinProtocol TLSv1.2, which
# refuses TLS 1.0/1.1, small RSA keys and SHA-1 certificates. Historian boxes
# offer exactly those, and msodbcsql18 does not report the rejection: the
# handshake stalls until the login timer expires, so the error reads
#
#     ('HYT00', '[Microsoft][ODBC Driver 18 for SQL Server]
#      Login timeout expired (0) (SQLDriverConnect)')
#
# which is indistinguishable from an unreachable server. On the client's
# deployment a plain TCP connect to 1433 succeeded while every login timed out
# at exactly 15 seconds, and the same credentials logged in from Windows --
# whose TLS stack accepts the older handshake.
#
# This relaxes the client side so the connection is still ENCRYPTED, just over
# an older protocol. That is a real reduction in TLS strength and it is the
# lesser of two evils: the alternative people reach for is Encrypt=no, which
# sends the password across the network in clear. Remove this once the server
# offers TLS 1.2.
RUN set -e; \
    conf=/etc/ssl/openssl.cnf; \
    if [ -f "$conf" ]; then \
        sed -i 's/^\(CipherString\s*=\s*DEFAULT\).*/\1:@SECLEVEL=0/' "$conf"; \
        sed -i 's/^\(MinProtocol\s*=\s*\).*/\1TLSv1/' "$conf"; \
        grep -E '^(CipherString|MinProtocol)' "$conf" || true; \
    fi

# Singapore time. Every timestamp in an alert is read by someone standing in
# Singapore, and a container defaulting to UTC makes "03:00" mean 11:00 to the
# person deciding whether to send a crew out.
ENV TZ=Asia/Singapore
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

WORKDIR /app

# Dependencies before source, so editing code does not reinstall the world.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- prove the ODBC driver actually loads ----------------------------------
# Installing the package is not evidence that the driver works. The image
# above once built green and then failed at the first query, because the
# driver's own dependency had been autoremoved -- a defect no amount of
# reading the Dockerfile reveals and that only surfaces on a machine with a
# SQL Server to point at.
#
# So load it here, at build time, with the same dlopen() the Driver Manager
# uses. Two checks, because they fail for different reasons: the first that
# odbcinst.ini registers the name the config expects (a missing entry gives
# IM002), the second that the library and its whole dependency chain resolve
# (a missing dependency gives "Can't open lib ... file not found").
#
# Neither needs a database, a password or a network. A broken image now fails
# the build, on the machine doing the building, instead of at 3 a.m. in front
# of whoever is deploying it.
RUN python -c "\
import ctypes, glob, sys, pyodbc; \
want='ODBC Driver 18 for SQL Server'; \
got=pyodbc.drivers(); \
sys.exit('odbcinst.ini has no %r -- only %r' % (want, got)) if want not in got else None; \
libs=sorted(glob.glob('/opt/microsoft/msodbcsql18/lib64/libmsodbcsql-*.so*')); \
sys.exit('driver library missing from /opt/microsoft/msodbcsql18/lib64') if not libs else None; \
ctypes.CDLL(libs[-1]); \
print('ODBC driver loads:', want, '->', libs[-1])"

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
