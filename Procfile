web: gunicorn app:app \
  -k uvicorn.workers.UvicornWorker \
  --workers 1 \
  --bind 127.0.0.1:8000 \
  --timeout 600
