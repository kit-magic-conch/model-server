FROM python:3.8
WORKDIR /usr/src/app/model
COPY . .
RUN pip3 install -r requirements.txt
RUN apt-get update -y && apt-get install -y libsndfile1
ENTRYPOINT [ "python", "-u", "main.py" ]