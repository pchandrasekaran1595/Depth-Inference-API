source venv/bin/activate && uvicorn main:app --host $(hostname -I) --port 9090 --workers 4