#!/usr/bin/python

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest
import time
import umsgpack

import sys
if sys.version_info[0] < 3:
    from StringIO import StringIO
else:
    from io import StringIO
from pandas import DataFrame


def binarize_sentiment(sentiment):
    if sentiment:
        return 1
    else:
        return 0

def main():
	#global argv
	start_time = time.clock()
	print 'Entering the main thread to start the program'
	#s = ''
	#digits = None
	#app2.run(debug=True)
	data = pd.read_csv('Review_chennai.csv',sep='|')
	print data.head()
	clean_rateofreview = lambda x: str(x).split()[1]
	data['rating'] = data['rateofreview'].apply(clean_rateofreview)
	revs = data.loc[:,['r_name','reviewtext','rating']]
	print revs.head()
	print revs.count()
	for i in list(np.where(pd.isnull(revs))):
	    revs.drop(revs.index[i], inplace=True)
	print revs.count()
	revs_new = revs[revs['rating'] != '3.0']
	revs_new['sentiment'] = revs_new['rating'] >= '3.5'
	print revs_new.head()
	revs_new['sentiment'] = revs_new['sentiment'].apply(binarize_sentiment)
	print revs_new.head()

	vectorer = CountVectorizer(min_df=2,ngram_range=(1,2))
	bow = vectorer.fit_transform(revs_new['reviewtext'])
	#new_vectorer = TfidfVectorizer(vocabulary=vectorer.vocabulary_)
	target = revs_new['sentiment'].values

	features_train, features_test, target_train, target_test = train_test_split(bow, target, test_size=0.20, random_state=1)

	print features_train.shape
	print target_train.shape
	print features_test.shape
	print target_test.shape

	logreg = LogisticRegression(C=1)
	logreg.fit(features_train, target_train)

	target_predicted = logreg.predict(features_test)
	print target_predicted

	print 'Testing Accuracy is ', accuracy_score(target_test, target_predicted)

	print 'Training Accuracy is', logreg.score(features_train, target_train)
	print 'Testing Accuracy is', logreg.score(features_test, target_test)



	########  Finding the best features ############
	# ch2 = SelectKBest(chi2, k=10000)
 #    X_train = ch2.fit_transform(features_train, target_train)
 #    X_test = ch2.transform(target_test)

    




	TESTDATA1=StringIO("""Review
	1;Sushi is Amazing
	2;Sushi is bad
	3;Sushi is not good
	4;Sushi is beautiful
	5;Sushi is bad terrible and good
	6;Sushi is amazing bad and terrible
	7;Sushi is amazing terrible horrible and bad
	8;Sushi is not awesome
	9;Sushi is not great
	10;Sushi is very bad
	11;Sushi is not brilliant
	12;Sushi is unpleasant
	13;Sushi is pleasant
	""")

	# print '################################'
	# print 'Number of arguments:', len(sys.argv), 'arguments.'
	# print 'Argument List:', str(sys.argv)
	# print 'test review is ', sys.argv[1]
	# test_review = str(sys.argv[1])
	# TESTDATA=StringIO("""Review
 # 	;""" + test_review)

	df1 = DataFrame.from_csv(TESTDATA1, sep=";", parse_dates=False)
	print df1





	test_bow = vectorer.transform(df1['Review'])
	prediction = logreg.predict(test_bow)
	print 'Old One: ', prediction


	####### Checking code ##########
	new_vectorer = CountVectorizer(vocabulary=vectorer.vocabulary_)
	test_new_bow = new_vectorer.transform(df1['Review'])
	new_prediction = logreg.predict(test_new_bow)
	print 'New one: ', new_prediction



	timedump_joblib = time.clock()
	# pickling the models
	from sklearn.externals import joblib
	joblib.dump(vectorer, 'CountVectorizer_Bigrams.pkl')
	joblib.dump(logreg,'CountLogRegModel_Bigrams.pkl')
	print 'Time for joblib dumping of models: ', time.clock() - timedump_joblib


	timedump_new_joblib = time.clock()
	# pickling the models
	from sklearn.externals import joblib
	joblib.dump(new_vectorer, 'CountVectorizer_Bigrams_new.pkl')
	joblib.dump(logreg,'CountLogRegModel_Bigrams_new.pkl')
	print 'Time for joblib dumping of double vectorized models: ', time.clock() - timedump_new_joblib

	# timedump = time.clock()
	# f = open('vect.bin', 'wb')
	# g = open('model.bin', 'wb')
	# #umsgpack.dump({u"compact": True, u"schema": 0}, f)
	# umsgpack.dump(str(vectorer), f)
	# umsgpack.dump(str(logreg), g)
	# f.close()
	# print 'Time for umsgpack dumping of models: ', time.clock() - timedump

	# timeload = time.clock()
	# f = open('vect.bin', 'rb')
	# g = open('model.bin', 'rb')
	# vectorer1 = umsgpack.load(f)
	# logreg1 = umsgpack.load(g)
	# print 'Time for umsgpack loading of models: ', time.clock() - timeload

	# print 'Loaded Vectorer is \n', vectorer1, type(vectorer1)
	# print 'Loaded Model is \n', logreg1

	# print 'Converted \n', 

	# test_bow = vectorer1.transform(df1['Review'])
	# prediction = logreg1.predict(test_bow)
	# print prediction


	print 'Overall time taken: ', time.clock() - start_time

main()

