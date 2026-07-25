# syntax=docker/dockerfile:1
FROM python:3.12-slim

WORKDIR /app

# Dependencies first so a code change does not reinstall them.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY cme_python ./cme_python

# The registry lives on a volume; the image itself stays stateless.
ENV CME_DATABASE=/data/cme.sqlite \
    PYTHONUNBUFFERED=1
RUN mkdir -p /data && useradd --create-home cme && chown cme /data
USER cme
VOLUME ["/data"]

EXPOSE 8000
CMD ["uvicorn", "cme_python.main:app", "--host", "0.0.0.0", "--port", "8000"]
