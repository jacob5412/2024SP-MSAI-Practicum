FROM python:3.11

COPY ./requirements.txt /tmp/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /tmp/requirements.txt

COPY ./app /app

COPY ./model /model

WORKDIR /app

EXPOSE 5000

RUN apt-get update 
RUN apt-get install ffmpeg libsm6 libxext6 -y

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]
