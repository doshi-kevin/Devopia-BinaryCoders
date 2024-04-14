import os
import subprocess
from flask import Flask, render_template, request

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/compile_run", methods=["POST"])
def compile_run():
    code = request.form["code"]
    input_text = request.form["input"]
    result = subprocess.run(
        ["python", "-c", code], input=input_text, capture_output=True, text=True
    )
    return result.stdout


if __name__ == "__main__":
    app.run(debug=True, port=3005)
