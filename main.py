from flask import Flask, request
import os
import uuid
from model import *


app = Flask(__name__)


@app.route('/')
def root():
    filename = str(uuid.uuid1())
    request.files['audio'].save(filename)
    result = recognize_emotion(filename)
    os.remove(filename)
    return result


if __name__ == "__main__":
    app.run()
