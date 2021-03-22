from flask import Flask, render_template, url_for, request
from sklearn.externals import joblib

app = Flask(__name__)

@app.route('/v1/api')
def home():
	return render_template('home.html')

@app.route('/v1/api/predict', methods = ['POST'])
def predict():
	clf = joblib.load('models/trained_model.pkl')
	if request.method == 'POST':
		data = str(request.form['user-input'])
		print("GOT")
		print(data)
		new_pred = clf.predict([data])
		print('\n***OUTPUT***\n')
		print(new_pred)
	return render_template('result.html', prediction = new_pred)

if __name__ == '__main__':
	app.run(debug = True)
