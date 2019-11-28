FROM ubuntu:latest

RUN apt-get update \
    && apt-get install -y python3-pip \
    && pip3 install --upgrade pip

RUN pip3 install numpy matplotlib sklearn tk tensorflow keras

WORKDIR /app

COPY project.py /app

CMD ["python3","-u","./project.py"]
