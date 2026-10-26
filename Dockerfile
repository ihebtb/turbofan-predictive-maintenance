FROM python:3.12-slim

WORKDIR /app
# CPU-only PyTorch keeps the image small.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

# Trained artifacts (produced by `python -m turbofan.train`).
COPY models ./models
ENV TURBOFAN_MODELS=/app/models
EXPOSE 8000
CMD ["uvicorn", "turbofan.api:app", "--host", "0.0.0.0", "--port", "8000"]
