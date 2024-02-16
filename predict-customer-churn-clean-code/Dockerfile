#FROM python:3.8
FROM dockerregistry-v2.vih.infineon.com/python:3.8

USER root

WORKDIR /app

COPY ./requirements_py3.8.txt ./requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

RUN pip install -U pytest && pytest

# Mount the local folders to the container
VOLUME /app/images
VOLUME /app/models
VOLUME /app/logs

CMD ["python", "churn_library.py"]    