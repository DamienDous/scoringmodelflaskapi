from flask import Flask, request, jsonify
import pandas as pd
import requests
import os
from os.path import exists
import io
import dropbox
import joblib

app = Flask(__name__)

model_path = 'model'

dbx = dropbox.Dropbox('sl.BN0iICVUKpTKadJ3SPGjAgdxDe1LX0C2yILam1-C-7_yB_84NlvwPRt7GHwSK28bmnnVRzgqDLRILx3mX5B3IgaoE59C9geJoLWvZiUXReD6AazCGfbVtyiE9rbeH0Il2dzwo-E')

if not os.path.isdir(model_path):
	os.makedirs(model_path)


def get_pipeline():  # Load pipeline
	path = model_path+'/pipeline.joblib'
	if not exists(path):
		print(path, 'does not exist')
		filename = "/pipeline.joblib"
		s, r = dbx.files_download(filename)
		with open(model_path+'/pipeline.joblib', 'wb') as f:
			f.write(r.content)
	pipeline = joblib.load(model_path+'/pipeline.joblib')
	return pipeline


pipeline = get_pipeline()


@app.route('/api/predict', methods=['POST'])
def predict():
	data = request.data
	io_val = io.StringIO(data.decode('utf-8'))
	df = pd.read_csv(io_val, index_col=[0])
	return jsonify(
		{'prediction': pipeline.predict_proba(df)[0].tolist()})


@app.route("/")
def home_view():
	return "<h1>Hello World!</h1>"
