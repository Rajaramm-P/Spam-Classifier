from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
#from sklearn.externals import joblib
#import pickle

# load the model from disk
filename = 'Spam_Classifier.pkl'
clf = pickle.load(open(filename, 'rb'))
tf=pickle.load(open('Vectorizer.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def index():
	return render_template('spam.html')

@app.route('/predict',methods=['POST'])
def predict():

	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		vect = tf.transform(data).toarray()
		my_prediction = clf.predict(vect)
	return render_template('spam_result.html',prediction = my_prediction)

if __name__ == '__main__':
	app.run()