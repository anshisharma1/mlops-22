FROM python:3.8.1
WORKDIR /app

COPY . /app

RUN pip3 --no-cache-dir install -r requirements.txt

EXPOSE 5000
WORKDIR /app/api
CMD ["python3","-m","flask","run","--host=0.0.0.0"]