from flask import Flask, request
import os
import uuid
from model import recognize_emotion


app = Flask(__name__)


@app.route('/')
def root():
    filename = str(uuid.uuid1())
    request.files['audio'].save(filename)
    result = recognize_emotion(filename)
    os.remove(filename)
    print(result)
    return result


if __name__ == "__main__":
    app.run(host='0.0.0.0')
