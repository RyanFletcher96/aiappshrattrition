# Image recipe for the HR attrition risk API (proof of concept).
#
# Build (from the project root, after training so models/model.joblib exists):
#   docker build -t attrition-api:1.0 .
# Run:
#   docker run --rm -p 8000:8000 attrition-api:1.0
# Then open http://127.0.0.1:8000/docs

# 1. Base runtime: a small official Python image, same Python version used in development.
FROM python:3.14-slim

# Don't write .pyc files and print logs straight away (useful for `docker logs`).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 2. All project files live in /app inside the image.
WORKDIR /app

# 3. Install only the packages the API needs to run.
#    This is copied and installed before the code, so Docker can reuse this
#    (slow) layer when only the code changes.
#    scikit-learn here must be the SAME version that trained model.joblib:
#    a pickled pipeline can fail to load, or silently behave differently,
#    under another scikit-learn version.
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# 4. Copy only what the API needs: code, config and the trained model.
#    The model must already exist (run training first); it is gitignored.
#    No data, notebooks, tests or secrets are copied into the image.
COPY src/ src/
COPY config/ config/
COPY models/model.joblib models/model.joblib

# 5. Run as a normal user rather than root, to limit the damage if the app is compromised.
#    The container only reads files, so it stays stateless.
#    A fixed numeric ID (10001) means the same user on any host.
RUN useradd --system --uid 10001 --no-create-home appuser
USER 10001

# 6. Document the port the API listens on.
EXPOSE 8000

# 7. Health check: healthy only if /health answers AND the model is loaded.
#    Uses Python's urllib because slim images don't include curl.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import json, sys, urllib.request; r = urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4); sys.exit(0 if json.load(r).get('model_loaded') else 1)"]

# 8. Start command: serve the API on all interfaces so it is reachable from outside the container.
CMD ["python", "-m", "uvicorn", "api:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
