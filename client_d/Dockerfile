FROM ubuntu:latest
LABEL authors="aroslavsapoval"
RUN apt-get update -qy && apt-get install -qy python3.10 python3.10-dev python3-pip
WORKDIR /app
#COPY . /app
RUN pip3 install virtualenv && virtualenv -p python3.10 venv && . venv/bin/activate
COPY ./requirements.txt /app/requirements.txt
RUN pip3 install -r requirements.txt
CMD ["python3", "client.py"]
