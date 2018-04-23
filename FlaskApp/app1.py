from flask import Flask, render_template, request, json, url_for
from sklearn.externals import joblib
import cPickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
import time

import sys
if sys.version_info[0] < 3:
    from StringIO import StringIO
else:
    from io import StringIO
from pandas import DataFrame

#   Program with Bi-grams models loaded Onloading of / route 
app1 = Flask(__name__)


@app1.route("/")
def Sentiment_Analyzer():
	#global vect
	#global mod
	#start_time = time.clock()
	#with open('Vect_cPickle_Ngrams.pkl', 'rb') as f:
		#vect = cPickle.load(f)
	#with open('Log_Reg_Model_cPickle_Ngrams.pkl', 'rb') as g:
		#mod = cPickle.load(g)
	#duration = time.clock() - start_time
	#print 'Time taken to load the models is ', duration
	return render_template('Analyze.html')

@app1.route("/results")
def results():
	print 'Inside results function ...'
	print ' Yet to save the model ...'
	return render_template('Analyze.html')

# @home.route("/sa")
# def sa():
# 	return render_template('sa.html')

# @home.route("/index")
# def index():
# 	return render_template('index.html')

@app1.route("/Analyze", methods=['GET','POST'])
def Analyze():
	if request.method == "POST":
		text = request.form['text']
		print 'Review text is ', text
		timet = time.clock()
		
		TESTDATA=StringIO("""Review
	 	;""" + text)

		df1 = DataFrame.from_csv(TESTDATA, sep=";", parse_dates=False)
		print df1

		test_bow = vect.transform(df1['Review'])
		prediction = mod.predict(test_bow)
		print prediction, type(str(prediction[0]))



		if str(prediction[0]) == '1':
			result = 'Positive'
			dict = {text : result}
			print '++++++++++++++++++++'
			print ' The Polarity of the review is  ', result
			print 'Time to predict the review = ', time.clock() - timet
			return render_template('posres.html', result = dict)

		else:
			result = 'Negative'
			dict = {text : result}
			print '--------------------'
			print ' The Polarity of the review is  ', result
			print 'Time to predict the review = ', time.clock() - timet
			return render_template('negres.html', result = dict) 


		# print ' The Polarity of the review is  ', result
		
		# print time.clock() - start_time, "seconds"
		#return render_template('Analyze.html'), result
		
		#return render_template('res.html', result = dict)
	else:
		return render_template('Analyze.html')
	



if __name__ == "__main__":
	#print 'Entering the main thread to start the program'
        app1.run()
