start /MAX cmd /c "cls && title Prepare Environment && py -3.9 -m venv venv && cd venv/Scripts && activate && cd .. && cd .. && pip install -r requirements.txt && timeout /t 5 /nobreak"