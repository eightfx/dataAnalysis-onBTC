FROM python:3.9.5
ENV PYTHONUNBUFFERED 1
RUN mkdir ./code
WORKDIR /code
COPY . /code/.
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN apt update -y
RUN apt install git -y
