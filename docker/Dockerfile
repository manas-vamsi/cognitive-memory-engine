# syntax=docker/dockerfile:1
FROM python:3.12-slim

WORKDIR /app

# Dependencies first so a code change does not reinstall them.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY cme_python ./cme_python

# The registry lives on a volume; the image itself stays stateless.
ENV CME_DATABASE=/data/cme.sqlite \
    CME_WORKERS=1 \
    PYTHONUNBUFFERED=1
RUN mkdir -p /data && useradd --create-home cme && chown cme /data
USER cme
VOLUME ["/data"]

EXPOSE 8000
# Shell form so CME_WORKERS expands, `exec` so uvicorn is still PID 1 and stops
# on the first SIGTERM rather than after the runtime's kill timeout.
#
# One worker by default. Extra workers are processes, which is the only thing
# that raises throughput here — threads inside one process queue on the GIL,
# measured twice. Set CME_WORKERS above 1 with CME_DATABASE pointing at
# PostgreSQL, or at a SQLite file, which is safe across processes now that the
# registry runs in WAL mode.
CMD ["sh", "-c", "exec uvicorn cme_python.main:app --host 0.0.0.0 --port 8000 --workers ${CME_WORKERS}"]
