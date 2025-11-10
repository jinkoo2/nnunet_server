echo "uvicorn main:app --host 0.0.0.0 --port 8000 --reload --reload-include '*.py' --reload-exclude '_*/**'"
uvicorn main:app --host 0.0.0.0 --port 8000


