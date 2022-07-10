FROM python:3.9

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

COPY ./static /code/static

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./main.py /code/

COPY ./test_main.py /code/

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9090", "--workers", "4"]
