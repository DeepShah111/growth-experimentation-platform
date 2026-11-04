# Growth Experimentation Platform API.   docker build -t gep-api . && docker run -p 8000:8000 gep-api
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=8000
WORKDIR /app

COPY api/requirements.txt api/requirements.txt
RUN pip install --no-cache-dir -r api/requirements.txt

COPY src/ src/
COPY sql/ sql/
COPY api/ api/
COPY sample_data/ sample_data/

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD python -c "import urllib.request,os;urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT','8000'))"
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
