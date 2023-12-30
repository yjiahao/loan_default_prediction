FROM python:3.9.18-slim-bullseye

WORKDIR /loan_default_prediction

COPY . /loan_default_prediction

RUN pip install -r requirements.txt