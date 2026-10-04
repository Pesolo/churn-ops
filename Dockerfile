# Matches local Python 3.11 exactly for reproducibility.
FROM python:3.11-slim

WORKDIR /app

# Copy requirements first so Docker can cache this layer -- rebuilds
# skip the (slow) pip install step if only application code changed,
# not dependencies.
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout 120 -r requirements.txt

# Only src/ is needed at runtime -- no data/, no notebooks/.
# train.py is included for now even though api.py doesn't call it;
# harmless, but flagging as something to trim later if image size matters.
COPY src/ .

EXPOSE 8000

# 0.0.0.0, not 127.0.0.1 -- binding to localhost-only would make the
# port unreachable from outside the container even with -p mapping.
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]