FROM python:3.7-alpine
COPY ./src/app.py /deploy/
COPY ./requirements.txt /deploy/
COPY ./saved_models/model_rfc2.pickle /deploy/
WORKDIR /deploy/
RUN pip install -r requirements.txt
EXPOSE 5000
ENTRYPOINT ["python", "app.py"]