FROM python:3.8
WORKDIR /usr/src/app/model
COPY . .
RUN pip3 install -r requirements.txt
ENTRYPOINT [ "python", "-u", "main.py" ]