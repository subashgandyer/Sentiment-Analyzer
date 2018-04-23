import nltk
import sqlite3
import pandas as pd
#from dishlist import dishlist
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
#from idfs_c import idfs 
import scipy.sparse as sp
import re

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

# class MyVectorizer(TfidfVectorizer):
#     # plug our pre-computed IDFs
#     TfidfVectorizer.idf_ = idfs

def clean_text(row):
    # return the list of decoded cell in the Series instead 
    return [r.decode('unicode_escape').encode('ascii', 'ignore') for r in row]

def review_cleanup(x, notwanted):
	for item in notwanted:
		x = re.sub(item,'',x)
		return x


def clean_reviewtext_symbols(review):
	return re.sub('[|<>:/\@#$%^&*()!~?="'']','',review)


def clean_reviewtext_dots(review):
	return re.sub('\.+','. ',review)

grammar = "NP: { <NN\w*>+}"

def main():

	#################################################
	######      LOADING REVIEW DATA           #######
	#################################################

	start_time = time.clock()
	print 'Entering the main thread to start the program'
	data = pd.read_csv('Review_chennai.csv',sep='|')
	print data.head()
	clean_rateofreview = lambda x: str(x).split()[1]
	data['rating'] = data['rateofreview'].apply(clean_rateofreview)
	revs = data.loc[:,['rest_review', 'r_name','reviewtext', 'date', 'rating']]
	print revs.head()
	print revs.count()
	for i in list(np.where(pd.isnull(revs))):
	    revs.drop(revs.index[i], inplace=True)
	print revs.count()
	#print type(revs), revs



	

	# clean Ascii Code in the review Text for emoticons and other stuff !!!!
	revs['text'] = revs['reviewtext'].apply(lambda x: x.decode('unicode_escape').\
                                          encode('ascii', 'ignore').\
                                          strip())
	
	# clean review text from '|' pipe and |<>:/\@#$%^&*()!~?="'' symbols that are 
	# used as separators in pandas
	revs['cleanedtext'] = revs['text'].apply(clean_reviewtext_symbols)
	revs['cleanedtext_dots'] = revs['cleanedtext'].apply(clean_reviewtext_dots)



	#print 'Cleaned Text \n', revs['cleanedtext']
	#print 'With dots removed \n', revs['cleanedtext_dots']
	reviews_text = revs['cleanedtext_dots'].str.lower()
	reviews_text = reviews_text.values
	res_names = revs['r_name'].values
	print 'RESTAURANT NAMES = \n', res_names
	print res_names[0], res_names[-1]
	score = 1.0
	created_dates = revs['date'].values
	print 'REVIEWED DATES = \n', created_dates
	print created_dates[0], created_dates[-1]
	#print type(reviews_text), reviews_text
	

	revs_new = revs[revs['rating'] != '3.0']
	revs_new['sentiment'] = revs_new['rating'] >= '3.5'
	print revs_new.head()
	revs_new['sentiment'] = revs_new['sentiment'].apply(binarize_sentiment)
	print revs_new.head()

	#vectorer = TfidfVectorizer(min_df=2)
	vectorer = TfidfVectorizer(min_df=2, ngram_range=(1,2))
	#print vectorer.vocabulary_
	bow = vectorer.fit_transform(revs_new['cleanedtext_dots'])
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


##########  Selecting the best features ##########
	# for finding the top-k unigram features
	indices = np.argsort(vectorer.idf_)[::-1]
	features = vectorer.get_feature_names()
	top_n = 20
	top_features = [features[i] for i in indices[:top_n]]
	print 'Top features are : \n', top_features

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
	proba_review = logreg.predict_proba(test_bow)
	print proba_review
	if str(prediction[0]) == '1':
		result = 'Positive'
		#dict1 = {review : result}
		score = proba_review[0][1]
		print '++++++++++++++++++++'
		print ' The Polarity of the review is  ', result
	else:
		result = 'Negative'
		#dict1 = {sent : result}
		score = proba_review[0][0]
		print '--------------------'
		print ' The Polarity of the review is  ', result
	print 'Old One: ', prediction




	timedump = time.clock()
	with open('Vectorer.pkl', 'wb') as f:
		cPickle.dump(vectorer, f, 2)

	with open('Classifier.pkl', 'wb') as g:
		cPickle.dump(logreg, g, 2)
	# with gzip.open('Vect_cPickle_Ngrams.pkl', 'wb') as f:
	# 	cPickle.dump(vectorer, f, 2)
	print 'Time for cPickle dumping of models: ', time.clock() - timedump

	timeload = time.clock()
	with open('Vectorer.pkl', 'rb') as f:
		loaded_vectorer = cPickle.load(f)
	with open('Classifier.pkl', 'rb') as g:
		loaded_logreg = cPickle.load(g)

	print 'Loaded Vectorizer is \n', loaded_vectorer

	print 'Time for cPickle loading of models: ', time.clock() - timeload

	test_bow1 = loaded_vectorer.transform(df1['Review'])
	new_prediction = loaded_logreg.predict(test_bow1)
	print 'New One: ', new_prediction
	# #################################################
	# ###       MODEL LOADING WITHOUT PICKLES       ###    0.5 seconds
	# #################################################

	# global vect
	# global mod
	# start_time = time.clock()
	# vect = MyVectorizer(min_df = 2,
 #                          ngram_range=(1,2))

	# # plug _tfidf._idf_diag
	# vect._tfidf._idf_diag = sp.spdiags(idfs,
 #                                         	diags = 0,
 #                                         	m = len(idfs),
 #                                         	n = len(idfs))


	# vocabulary = json.load(open('vocabulary.json', mode = 'rb'))
	# vect.vocabulary_ = vocabulary
	# duration = time.clock() - start_time
	# #print 'Time taken to load the models is ', duration
	# with open('Log_Reg_Model_cPickle_Ngrams.pkl', 'rb') as g:
	# 	mod = cPickle.load(g)
	# duration = time.clock() - start_time
	# #print 'Time taken to load the models is ', duration


	# #############################################
	# ###       MODEL LOADING WITH PICKLES      ###    2.5 seconds
	# #############################################

	# # timeload = time.clock()
	# # with open('Vect_cPickle_Ngrams.pkl', 'rb') as f:
	# # 	vect = cPickle.load(f)
	# # with open('Log_Reg_Model_cPickle_Ngrams.pkl', 'rb') as g:
	# # 	mod = cPickle.load(g)
	# # print 'Time for cPickle loading of models: ', time.clock() - timeload

	# #######################################
	# ####     DATABASE CREATION        #####
	# #######################################

	# conn = sqlite3.connect('test1.db')
	# print "Opened database successfully"

	# # conn.execute('''CREATE TABLE RATINGS
	# #        (ID INT PRIMARY KEY     NOT NULL,
	# #        RES_NAME       TEXT    NOT NULL,
	# #        RES_ID 		  INT  	  NOT NULL,
	# #        DISH_NAME      TEXT     NOT NULL,
	# #        SENTIMENT      CHAR(10)  NOT NULL)''')
	# # print "Table created successfully"


	# conn.execute('''CREATE TABLE SA_REVIEW_SCORE
	#        (ID INT PRIMARY KEY     	NOT NULL,
	#        REVIEW_ID  	  INTEGER 	  	NOT NULL,
	#        RES_NAME       TEXT   	NOT NULL,
	#        KEYWORD_ID 	  INTEGER  	  	NOT NULL,
	#        DISH_NAME      TEXT    	NOT NULL,
	#        SCORE 		  REAL 	  	NOT NULL,
	#        SENTIMENT      CHAR(10)  NOT NULL,
	#        CREATED_DATE   DATE 		NOT NULL)''')
	# print "Table created successfully"

	# #######  Used for creating dish_list list in dish_list.py #########
	# # dish_df = pd.read_csv('Food Dishes_SA.csv')
	# # dish_list = dish_df['Table 1'].values
	# # print dish_list
	# # dish_list = [x.lower() for x in dish_list]
	# # print dish_list
	
	# #print dish_list
	# dish_list1 = []
	# dish_list1 = '\t'.join(dish_list)
	# text2 = """Sushi is amazingly bad. Service is bad. Noodles is awesome. 
	# 		Interiors were badly made. Nigiri is good. Idli is amazing. Aloo gobi is nice.
	# 		What can i say about Mutton Biriyani? It is bad. The waiters are patient. 
	# 		They are really good."""

	# text1 = """Waiters are patient. They are also amazing. Biryani is awesome.
	# 			I would come any day here."""
	
	# cp = nltk.RegexpParser(grammar)
	# counter = 0
	# review_index = 0
	# for review in reviews_text:
	# 	review_index += 1
	# 	print '########## REVIEW # %d ##############' %(review_index)
	# 	print ' '
	# 	print ' '
	# 	print ' '
	# 	print ' '
	# 	print ' '
	# 	print ' '
	# 	print ' '
	# 	print ' '
	# 	print ' ####################################'
	# 	print "REVIEW = ", review
	# 	##### Removing | symbol in the review text #######
		
	# 	#review = review_cleanup(review, unwanted_elements)
	# 	# review = re.sub('[|]','',review)
	# 	# print 'Cleaned Review = ', review


	# 	#####################################################
	# 	######     FULL REVIEW SENTIMENT PREDICTION    ######
	# 	#####################################################

	# 	REVIEWDATA=StringIO("""Review
	# 			|""" + review)

	# 	df = DataFrame.from_csv(REVIEWDATA, sep="|", parse_dates=False)
	# 	print 'FULL REVIEW SENTIMENT PREDICTION GOES ON .......'
	# 	print df
	# 	review_bow = vect.transform(df['Review'])
	# 	pred_review = mod.predict(review_bow)
	# 	proba_review = mod.predict_proba(review_bow)
	# 	print review_bow
	# 	print pred_review
	# 	print proba_review
	# 	if str(pred_review[0]) == '1':
	# 		result = 'Positive'
	# 		dict1 = {review : result}
	# 		score = proba_review[0][1]
	# 		print '++++++++++++++++++++'
	# 		print ' The Polarity of the review is  ', result
	# 	else:
	# 		result = 'Negative'
	# 		dict1 = {sent : result}
	# 		score = proba_review[0][0]
	# 		print '--------------------'
	# 		print ' The Polarity of the review is  ', result


	# 	##################################################
	# 	######     SENTENCE SENTIMENT PREDICTION    ######
	# 	##################################################

	# 	sentences = nltk.sent_tokenize(review)
	# 	sent_index = 0
	# 	prp_list_index = []
	# 	for sent in sentences:
	# 		sent_index += 1
	# 		#print 'Sentence No: ', sent_index
	# 		tagged = nltk.pos_tag(nltk.word_tokenize(sent))
	# 		#print tagged
	# 		#nouns = [word for word, pos in tagged if pos in ['NN', 'NNP', 'NNS']]
	# 		#print 'Nouns = ', nouns
	# 		#adjectives = [word for word, pos in tagged if pos in ['JJ']]
	# 		#print 'Adjectives = ', adjectives
	# 		#adverbs = [word for word, pos in tagged if pos in ['RB', 'RBS']]
	# 		#print 'Adverbs = ', adverbs
	# 		######## COLLECTING SUCCESSIVE NOUNS ########
	# 		parsed_content = cp.parse(tagged)
	# 		dish1 = re.findall(r'NP\s(.*?)/NN\w*', str(parsed_content))
	# 		dish2 = re.findall(r'NP\s(.*?)/NN\w*\s(.*?)/NN', str(parsed_content))
	# 		dish3 = re.findall(r'NP\s(.*?)/NN\w*\s(.*?)/NN\w*\s(.*?)/NN', str(parsed_content))
	# 		dish4 = re.findall(r'NP\s(.*?)/NN\w*\s(.*?)/NN\w*\s(.*?)/NN\w*\s(.*?)/NN', str(parsed_content))
	# 		#print parsed_content
	# 		#print dish1
	# 		#print dish2
	# 		#print dish3
			
	# 		nouns = []
	# 		dlist = []

	# 		if len(dish4) != 0:
	# 			for t in dish4:
	# 				noun = ' '.join(item for item in t)
	# 				nouns.append(noun)

	# 		#print 'Nouns after 1st iteration ', nouns
	# 		dlist = '\t'.join(nouns)
			
	# 		if len(dish3) != 0:
	# 			for t in dish3:
	# 				noun = ' '.join(item for item in t)
	# 				if noun not in dlist:
	# 					nouns.append(noun)

	# 		#print 'Nouns after 1st iteration ', nouns
	# 		dlist = '\t'.join(nouns)

	# 		if len(dish2) != 0:
	# 			for t in dish2:
	# 				noun = ' '.join(item for item in t)
	# 				if noun not in dlist:
	# 					nouns.append(noun)
	# 		#print 'Nouns after 2nd iteration ', nouns
	# 		dlist = '\t'.join(nouns)

	# 		if len(dish1) != 0:
	# 			for t in dish1:
	# 				if t not in dlist:
	# 					nouns.append(t)

	# 		#print 'Nouns after last iteration ', nouns

	# 		# prps = [word for word, pos in tagged if pos in ['PRP']]
	# 		# print 'PRPs = ', prps

	# 		###
	# 		####
	# 		###
	# 		###
	# 		##### Including NamedEntity
	# 		# namedEnt = nltk.ne_chunk(tagged, binary=True)
	# 		# print 'NamedEnt = ', namedEnt
	# 		#namedEnt.draw()
			
	# 		# for noun in nouns:
	# 		# 	print noun, type(noun)
	# 		flag = False
	# 		for noun in nouns:
	# 			if noun in dish_list1 and len(noun) > 3:
	# 				flag = True
	# 			else:
	# 				pass

	# 		#print 'Flag = ', flag
	
				
	# 		if flag:
				
	# 			#print 'predicting sentiment for sentence', sent

	# 			############################
	# 			### PREDICTING SENTIMENT ###
	# 			############################
	# 			test_time = time.clock()
	# 			TESTDATA=StringIO("""Review
	#  				|""" + sent)

	# 			df1 = DataFrame.from_csv(TESTDATA, sep="|", parse_dates=False)
	# 			#print df1
	# 			test_time = time.clock()
	# 			test_bow = vect.transform(df1['Review'])
	# 			#print 'Data frame creation time = ', time.clock()-test_time
				
	# 			prediction = mod.predict(test_bow)
	# 			probability = mod.predict_proba(test_bow)
	# 			#print 'Score = ', score , type(score), score[0][1] 
	# 			#print prediction, type(str(prediction[0]))
	# 			if str(prediction[0]) == '1':
	# 				result = 'Positive'
	# 				dict = {sent : result}
	# 				score = probability[0][1]
	# 				#print '++++++++++++++++++++'
	# 				#print ' The Polarity of the review is  ', result
	# 				# print 'Time to predict the review = ', time.clock() - timet
	# 				# return render_template('posres.html', result = dict)
	# 			else:
	# 				result = 'Negative'
	# 				dict = {sent : result}
	# 				score = probability[0][0]
	# 				#print '--------------------'
	# 				#print ' The Polarity of the review is  ', result
	# 				# print 'Time to predict the review = ', time.clock() - timet
	# 				# return render_template('negres.html', result = dict)
				
	# 			#print 'inserting noun,prediction,res_name,res_id,review_id into sqlite3 db'
	# 			#print 'creating a list & jsonify it as output'
	# 			#print 'Keyword %s is %s' % (noun, result)
	# 			#cur.execute("INSERT INTO Contacts VALUES (?, ?, ?, ?);", (firstname, lastname, phone, email))
	# 			for noun in nouns:
	# 				if noun in dish_list1 and len(noun) > 3:
	# 					counter += 1
	# 					#conn.execute("INSERT INTO RATINGS VALUES (?, ?, ?, ?, ?)", (counter, 'Mark', 25, noun, result))
	# 					conn.execute("INSERT INTO SA_REVIEW_SCORE VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (counter, review_index,res_names[review_index-1], counter, noun, score, result, created_dates[review_index-1]))
						
	# 					#print 'Noun = %s & Sentiment = %s ' %(noun, result)
	# 				else:
	# 					pass 

	# 			#conn.execute("INSERT INTO RATINGS VALUES (?, ?, ?, ?, ?)", (counter, 'Mark', 25, noun, result))

	# 			conn.commit()
	# 			#print "Records created successfully";
				
	# 		else:
	# 			pass
	# 	# 	for prp in prps:
	# 	# 		print prp
	# 	# 		prp_list_index.append(sent_index)
				
				


	# 	# print 'PRP LIST INDEX = ', prp_list_index
	# 	# prp_list_index = list(set(prp_list_index))
	# 	# print 'PRP LIST INDEX = ', prp_list_index
	# 	# for x in prp_list_index:
	# 	# 	print 'x in prp_list_index = ', x, type(x)
	# 	# 	if x == 1:
	# 	# 		pass
	# 	# 	else:
	# 	# 		counter += 1
	# 	# 		print 'predicting previous sentence\'s sentiment', sentences[x-2]
	# 	# 		#sentiment ='negative'

	# 	# 		TESTDATA=StringIO("""Review
	# 	# 	 		|""" + sentences[x-1])

	# 	# 		df2 = DataFrame.from_csv(TESTDATA, sep="|", parse_dates=False)
	# 	# 		#print df2

	# 	# 		test_bow = vect.transform(df1['Review'])
	# 	# 		prediction = mod.predict(test_bow)
	# 	# 		print prediction, type(str(prediction[0]))
	# 	# 		if str(prediction[0]) == '1':
	# 	# 			result = 'Positive'
	# 	# 			dict = {sent : result}
	# 	# 			print '++++++++++++++++++++'
	# 	# 			print ' The Polarity of the review is  ', result
	# 	# 			# print 'Time to predict the review = ', time.clock() - timet
	# 	# 			# return render_template('posres.html', result = dict)
	# 	# 		else:
	# 	# 			result = 'Negative'
	# 	# 			dict = {sent : result}
	# 	# 			print '--------------------'
	# 	# 			print ' The Polarity of the review is  ', result

	# 	# 		print 'inserting noun,prediction,res_name,res_id,review_id into sqlite3 db'
	# 	# 		print 'creating a list & jsonify it as output'
	# 	# 		print 'noun = noun(sentences[x-2]) choose the noun from sentences[x-2]'
	# 	# 		print sentences[x-2]
	# 	# 		tagged_prp = nltk.pos_tag(nltk.word_tokenize(sentences[x-2]))
	# 	# 		print tagged_prp
	# 	# 		nouns_prp = [word for word, pos in tagged_prp if pos in ['NN', 'NNP', 'NNS']]
	# 	# 		print 'Nouns_prp = ', nouns_prp
	# 	# 		str1 = ' '.join(str(x) for x in nouns_prp)
	# 	# 		print str1
	# 	# 		print 'please check if the previous sentence\'s noun is in dish_list'
	# 	# 		print 'Keyword %s is %s' % (str1, result)
	# 	# 		#cur.execute("INSERT INTO Contacts VALUES (?, ?, ?, ?);", (firstname, lastname, phone, email))
	# 	# 		if str1 in dish_list1:
	# 	# 			conn.execute("INSERT INTO RATINGS VALUES (?, ?, ?, ?, ?)", (counter, 'Mark', 25, str1, result))

	# 	# 			conn.commit()
				
	# 	# 			print "Records created successfully";

	# cursor = conn.execute("SELECT * from SA_REVIEW_SCORE")
	# print 'Printing Stored values in SA_REVIEW_SCORE table'
	# print "REVIEW_ID \t RES_NAME \t KEYWORD_ID \t DISH_NAME \t SCORE \t SENTIMENT \t CREATED_DATE "
	   
	# for row in cursor:
	   
	#    # print "REVIEW_ID = ", row[1]
	#    # print "RES_NAME = ", row[2]
	#    # print "KEYWORD_ID = ", row[3]
	#    # print "DISH_NAME = ", row[4]
	#    # print "SCORE = %3.2f" % (row[5])
	#    # print "SENTIMENT = ", row[6]
	#    # print "CREATED_DATE = ", row[7] ,"\n"

	#    print "%d \t %s \t %d \t %s \t %3.2f \t %s \t %s" % (row[1],row[2],row[3],row[4],row[5],row[6],row[7])
	# conn.close()

	print 'Program exiting.....'
	print 'Total time taken: ', time.clock() - start_time, ' seconds'

	return 0


main()