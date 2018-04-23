#!/usr/bin/python

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import time
import cPickle

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

	#vectorer = TfidfVectorizer(min_df=2)
	vectorer = TfidfVectorizer(min_df=2, ngram_range=(1,2))
	bow = vectorer.fit_transform(revs_new['reviewtext'])
	#new_vectorer = TfidfVectorizer(vocabulary=vectorer.vocabulary_)
	target = revs_new['sentiment'].values

	features_train, features_test, target_train, target_test = train_test_split(bow, target, test_size=0.20, random_state=1)

	print features_train.shape
	print target_train.shape
	print features_test.shape
	print target_test.shape


	######## Trying to serialize the model without pickling
	print '####################'
	print '####################'
	print '####################'
	
	#print vectorer.idf_

	logreg = LogisticRegression(C=1)
	logreg.fit(features_train, target_train)

	target_predicted = logreg.predict(features_test)
	print target_predicted

	print 'Testing Accuracy is ', accuracy_score(target_test, target_predicted)

	print 'Training Accuracy is', logreg.score(features_train, target_train)
	print 'Testing Accuracy is', logreg.score(features_test, target_test)


###########  Selecting the best features ##########
	# for finding the top-k unigram features
	# indices = np.argsort(vectorer.idf_)[::-1]
	# features = vectorer.get_feature_names()
	# top_n = 20
	# top_features = [features[i] for i in indices[:top_n]]
	# print 'Top features are : \n', top_features

	# # for finding the top-k allgram features
	# features_by_gram = defaultdict(list)
	# for f, w in zip(vectorer.get_feature_names(), vectorer.idf_):
	# 	features_by_gram[len(f.split(' '))].append((f, w))
	# top_n = 20
	# for gram, features in features_by_gram.iteritems():
	# 	top_features = sorted(features, key=lambda x: x[1], reverse=True)[:top_n]
	# 	top_features = [f[0] for f in top_features]
	# 	print '{}-gram top:'.format(gram), top_features

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



	print 'Vectorizer Type = ', type(vectorer)
	print vectorer

	test_bow = vectorer.transform(df1['Review'])

	

	prediction = logreg.predict(test_bow)
	print 'Old One: ', prediction


	# ####### Checking code ##########
	# new_vectorer = TfidfVectorizer(vocabulary=vectorer.vocabulary_)
	# test_new_bow = new_vectorer.fit_transform(df1['Review'])
	# new_prediction = logreg.predict(test_new_bow)
	# print 'New one: ', new_prediction



	# timedump_joblib = time.clock()
	# # pickling the models
	# from sklearn.externals import joblib
	# joblib.dump(vectorer, 'TfidfVectorizer_Ngrams.pkl')
	# joblib.dump(logreg,'LogRegModel_Ngrams.pkl')
	# print 'Time for joblib dumping of models: ', time.clock() - timedump_joblib


	# timedload_joblib = time.clock()
	# # pickling the models
	# from sklearn.externals import joblib
	# vect2 = joblib.load('TfidfVectorizer_Ngrams.pkl')
	# mod2 = joblib.load('LogRegModel_Ngrams.pkl')
	# print 'Time for joblib loading of models: ', time.clock() - timeload_joblib
	
	# timedump_new_joblib = time.clock()
	# # pickling the models
	# from sklearn.externals import joblib
	# joblib.dump(new_vectorer, 'TfidfVectorizer_new.pkl')
	# joblib.dump(logreg,'LogRegModel_new.pkl')
	# print 'Time for joblib dumping of double vectorized models: ', time.clock() - timedump_new_joblib


	

	


	timedump = time.clock()
	with open('Vect_cPickle_Ngrams.pkl', 'wb') as f:
		cPickle.dump(vectorer, f, 2)

	with open('Log_Reg_Model_cPickle_Ngrams.pkl', 'wb') as g:
		cPickle.dump(logreg, g, 2)
	# with gzip.open('Vect_cPickle_Ngrams.pkl', 'wb') as f:
	# 	cPickle.dump(vectorer, f, 2)
	print 'Time for cPickle dumping of models: ', time.clock() - timedump

	timeload = time.clock()
	with open('Vect_cPickle_Ngrams.pkl', 'rb') as f:
		loaded_vectorer = cPickle.load(f)
	with open('Log_Reg_Model_cPickle_Ngrams.pkl', 'rb') as g:
		loaded_logreg = cPickle.load(g)

	print 'Loaded Vectorizer is \n', loaded_vectorer

	print 'Time for cPickle loading of models: ', time.clock() - timeload

	test_bow1 = loaded_vectorer.transform(df1['Review'])
	new_prediction = loaded_logreg.predict(test_bow1)
	print 'New One: ', new_prediction








	import cPickle
	import gzip


	timedump = time.clock()
	with gzip.open('Vect_cPickle_Ngrams.pkl', 'wb') as f:
		cPickle.dump(vectorer, f, 2)
	with gzip.open('Log_Reg_Model_cPickle_Ngrams.pkl', 'wb') as g:
		cPickle.dump(logreg, g, 2)

	print 'Time for cPickle dumping of models: ', time.clock() - timedump

	timeload = time.clock()
	with gzip.open('Vect_cPickle_Ngrams.pkl', 'rb') as f:
		loaded_vectorer = cPickle.load(f)
	with gzip.open('Log_Reg_Model_cPickle_Ngrams.pkl', 'rb') as g:
		loaded_model = cPickle.load(g)

	# print 'Loaded Vectorizer is \n', loaded_vectorer

	# print 'Time for cPickle loading of models: ', time.clock() - timeload

	test_bow1 = loaded_vectorer.transform(df1['Review'])
	new_prediction = loaded_model.predict(test_bow1)
	print 'New One: ', new_prediction

	# timedump = time.clock()
	# a = pd.to_msgpack('foo.msg',vectorer)
	# print 'Time for pandas.msgpack dumping of models: ', time.clock() - timedump
	# timeload = time.clock()
	# b = pd.read_msgpack('foo.msg')
	# print 'Time for pandas.msgpack loading of models: ', time.clock() - timeload



	# timedump = time.clock()
	# f = open('vect.bin', 'wb')
	# g = open('model.bin', 'wb')
	# #umsgpack.dump({u"compact": True, u"schema": 0}, f)
	# umsgpack.dump(vectorer, f)
	# umsgpack.dump(logreg, g)
	# f.close()
	# print 'Time for umsgpack dumping of models: ', time.clock() - timedump

	# timeload = time.clock()
	# f = open('vect.bin', 'rb')
	# g = open('model.bin', 'rb')
	# vectorer1 = umsgpack.load(f)
	# logreg1 = umsgpack.load(g)
	# print 'Time for umsgpack loading of models: ', time.clock() - timeload

	# print 'Loaded Vectorer is \n', vectorer1, type(vectorer1)
	# print 'Loaded Model is \n', logreg1, type(logreg1)

	# print 'Converted \n'

	# test_bow = vectorer1.transform(df1['Review'])
	# prediction = logreg1.predict(test_bow)
	# print prediction


	print 'Overall time taken: ', time.clock() - start_time

main()

