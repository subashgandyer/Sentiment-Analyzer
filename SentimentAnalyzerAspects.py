import nltk
import sqlite3
import pandas as pd
from dishlist import dishlist
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
from idfs_c import idfs 
import scipy.sparse as sp
import re

import sys
if sys.version_info[0] < 3:
    from StringIO import StringIO
else:
    from io import StringIO
from pandas import DataFrame



class MyVectorizer(TfidfVectorizer):
    # plug our pre-computed IDFs
    TfidfVectorizer.idf_ = idfs

def clean_text(row):
    # return the list of decoded cell in the Series instead 
    return [r.decode('unicode_escape').encode('ascii', 'ignore') for r in row]

def review_cleanup(x, notwanted):
	for item in notwanted:
		x = re.sub(item,'',x)
		return x


def clean_reviewtext_symbols(review):
	return re.sub('[|\'<>:/\@#$%^&*()!~?="'']','',review)


def clean_reviewtext_dots(review):
	return re.sub('\.+','. ',review)

grammar = "NP: { <NN\w*>+}"

dishes = {}

def main():

	#################################################
	######      LOADING REVIEW DATA           #######
	#################################################

	start_time = time.clock()
	print 'Entering the main thread to start the program'
	data = pd.read_csv('Review_chennai_1.csv',sep='|')
	#print data.head()
	revs = data.loc[:,['rest_review', 'r_name','reviewtext', 'date']]
	#print revs.head()
	#print revs.count()
	for i in list(np.where(pd.isnull(revs))):
	    revs.drop(revs.index[i], inplace=True)
	#print revs.count()
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
	


	#################################################
	###       MODEL LOADING WITHOUT PICKLES       ###    0.5 seconds
	#################################################

	global vect
	global mod
	start_time = time.clock()
	# vect = MyVectorizer(min_df = 2,
 #                          ngram_range=(1,2))

	# # plug _tfidf._idf_diag
	# vect._tfidf._idf_diag = sp.spdiags(idfs,
 #                                         	diags = 0,
 #                                         	m = len(idfs),
 #                                         	n = len(idfs))


	# vocabulary = json.load(open('vocabulary.json', mode = 'rb'))
	# vect.vocabulary_ = vocabulary
	start_time = time.clock()
	with open('Vectorer.pkl', 'rb') as g:
		vect = cPickle.load(g)
	with open('Classifier.pkl', 'rb') as g:
		mod = cPickle.load(g)
	duration = time.clock() - start_time
	print 'Time taken to load the models is ', duration


	#############################################
	###       MODEL LOADING WITH PICKLES      ###    2.5 seconds
	#############################################

	# timeload = time.clock()
	# with open('Vect_cPickle_Ngrams.pkl', 'rb') as f:
	# 	vect = cPickle.load(f)
	# with open('Log_Reg_Model_cPickle_Ngrams.pkl', 'rb') as g:
	# 	mod = cPickle.load(g)
	# print 'Time for cPickle loading of models: ', time.clock() - timeload

	#######################################
	####     DATABASE CREATION        #####
	#######################################

	conn = sqlite3.connect('show_1.db')
	print "Opened database successfully"

	# conn.execute('''CREATE TABLE RATINGS
	#        (ID INT PRIMARY KEY     NOT NULL,
	#        RES_NAME       TEXT    NOT NULL,
	#        RES_ID 		  INT  	  NOT NULL,
	#        DISH_NAME      TEXT     NOT NULL,
	#        SENTIMENT      CHAR(10)  NOT NULL)''')
	# print "Table created successfully"


	conn.execute('''CREATE TABLE SA_REVIEW_SCORE
	       (ID INT PRIMARY KEY     	NOT NULL,
	       REVIEW_ID  	  INTEGER 	  	NOT NULL,
	       RES_NAME       TEXT   	NOT NULL,
	       KEYWORD_ID 	  INTEGER  	  	NOT NULL,
	       DISH_NAME      TEXT    	NOT NULL,
	       SCORE 		  REAL 	  	NOT NULL,
	       SENTIMENT      CHAR(10)  NOT NULL,
	       CREATED_DATE   DATE 		NOT NULL)''')
	#print "Table created successfully"

	conn.execute('''CREATE TABLE ASPECTS_SCORE
	       (ID INT PRIMARY KEY     	NOT NULL,
	       REVIEW_ID  	  INTEGER 	  	NOT NULL,
	       RES_NAME       TEXT   	NOT NULL,
	       REVIEW_SENTIMENT 	  TEXT  	  	NOT NULL,
	       SERVICE_SENTIMENT      TEXT   	NOT NULL,
	       VALUE_SENTIMENT 		  TEXT 	  	NOT NULL,
	       AMBIENCE_SENTIMENT     TEXT  NOT NULL,
	       FOOD_SENTIMENT      TEXT  NOT NULL)''')
	print "Table created successfully"

	#######  Used for creating dish_list list in dish_list.py #########
	# dish_df = pd.read_csv('Food Dishes_SA.csv')
	# dish_list = dish_df['Table 1'].values
	# print dish_list
	# dish_list = [x.lower() for x in dish_list]
	# print dish_list
	
	#print dish_list
	# dish_list1 = []
	# dish_list1 = '\t'.join(dish_list)
	text2 = """Sushi is amazingly bad. Service is bad. Noodles is awesome. 
			Interiors were badly made. Nigiri is good. Idli is amazing. Aloo gobi is nice.
			What can i say about Mutton Biriyani? It is bad. The waiters are patient. 
			They are really good."""

	text1 = """Waiters are patient. They are also amazing. Biryani is awesome.
				I would come any day here."""


	servicelist = ['service', 'waiter', 'welcome', 'friendly','staff','waitress','bar tender','bartender','chef','people']
	valuelist = ['value','cheap','cost','price','economical','reasonable','budget','pricey']
	ambiencelist = ['place','environment','atmosphere','climate','surroundings','look','mood','view','serene']
	foodlist = ['meal', 'lunch', 'dinner', 'brunch','snacks','cuisine','entree','starters']

	cp = nltk.RegexpParser(grammar)
	dish_counter = 0
	review_index = 0
	nouns_list = []
	ambience_sentiment = '#'
	value_sentiment = '#'
	food_sentiment = '#'
	service_sentiment = '#'
	review_sentiment = '#'

	for review in reviews_text:
		dup_dishes = []
		review_index += 1
		print '########## REVIEW # %d ##############' %(review_index)
		print ' '
		print ' '
		print ' '
		print ' '
		print ' '
		print ' '
		print ' '
		print ' '
		print ' ####################################'
		print "REVIEW = ", review
		##### Removing | symbol in the review text #######
		
		#review = review_cleanup(review, unwanted_elements)
		# review = re.sub('[|]','',review)
		# print 'Cleaned Review = ', review


		#####################################################
		######     FULL REVIEW SENTIMENT PREDICTION    ######
		#####################################################

		REVIEWDATA=StringIO("""Review
				|""" + review)

		df = DataFrame.from_csv(REVIEWDATA, sep="|", parse_dates=False)
		print 'FULL REVIEW SENTIMENT PREDICTION GOES ON .......'
		print df
		review_bow = vect.transform(df['Review'])
		pred_review = mod.predict(review_bow)
		proba_review = mod.predict_proba(review_bow)
		#print review_bow
		print pred_review
		print proba_review
		if str(pred_review[0]) == '1':
			result = 'Positive'
			dict1 = {review : result}
			score = proba_review[0][1]
			print '++++++++++++++++++++'
			print ' The Polarity of the review is  ', result
		else:
			result = 'Negative'
			#dict1 = {sent : result}
			score = proba_review[0][0]
			print '--------------------'
			print ' The Polarity of the review is  ', result

		review_sentiment = result
		print 'Full review sentiment is ', review_sentiment
		##################################################
		######     SENTENCE SENTIMENT PREDICTION    ######
		##################################################

		sentences = nltk.sent_tokenize(review)
		sent_index = 0
		prp_list_index = []
		for sent in sentences:
			sent_index += 1
			#print 'Sentence No: ', sent_index
			tagged = nltk.pos_tag(nltk.word_tokenize(sent))
			#print tagged
			#nouns = [word for word, pos in tagged if pos in ['NN', 'NNP', 'NNS']]
			#print 'Nouns = ', nouns
			#adjectives = [word for word, pos in tagged if pos in ['JJ']]
			#print 'Adjectives = ', adjectives
			#adverbs = [word for word, pos in tagged if pos in ['RB', 'RBS']]
			#print 'Adverbs = ', adverbs
			######## COLLECTING SUCCESSIVE NOUNS ########
			parsed_content = cp.parse(tagged)
			dish1 = re.findall(r'NP\s(.*?)/NN\w*', str(parsed_content))
			dish2 = re.findall(r'NP\s(.*?)/NN\w*\s(.*?)/NN', str(parsed_content))
			dish3 = re.findall(r'NP\s(.*?)/NN\w*\s(.*?)/NN\w*\s(.*?)/NN', str(parsed_content))
			dish4 = re.findall(r'NP\s(.*?)/NN\w*\s(.*?)/NN\w*\s(.*?)/NN\w*\s(.*?)/NN', str(parsed_content))
			print parsed_content
			
			
			nouns = []
			dlist = []

			if len(dish4) != 0:
				for t in dish4:
					noun = ' '.join(item for item in t)
					nouns.append(noun)
					nouns_list.append(noun)

			#print 'Nouns after 1st iteration ', nouns
			dlist = '\t'.join(nouns)
			
			if len(dish3) != 0:
				for t in dish3:
					noun = ' '.join(item for item in t)
					if noun not in dlist:
						nouns.append(noun)
						nouns_list.append(noun)

			#print 'Nouns after 1st iteration ', nouns
			dlist = '\t'.join(nouns)

			if len(dish2) != 0:
				for t in dish2:
					noun = ' '.join(item for item in t)
					if noun not in dlist:
						nouns.append(noun)
						nouns_list.append(noun)
			#print 'Nouns after 2nd iteration ', nouns
			dlist = '\t'.join(nouns)

			if len(dish1) != 0:
				for t in dish1:
					if t not in dlist:
						nouns.append(t)
						nouns_list.append(t)

			print 'Nouns after last iteration ', nouns

			# prps = [word for word, pos in tagged if pos in ['PRP']]
			# print 'PRPs = ', prps

			###
			####
			###
			###
			##### Including NamedEntity
			# namedEnt = nltk.ne_chunk(tagged, binary=True)
			# print 'NamedEnt = ', namedEnt
			#namedEnt.draw()
			
			# for noun in nouns:
			# 	print noun, type(noun)
			flagService = False
			flagDish = False
			flagAmbience = False
			flagValue = False
			flagFood = False
			for noun in nouns:
				if noun in dishlist and len(noun) > 3:
					flagDish = True
				if noun in servicelist:
					flagService = True
				if noun in ambiencelist:
					flagAmbience = True
				if noun in valuelist:
					flagValue = True
				if noun in foodlist:
					flagFood = True
				else:
					pass

				print 'Flags : ' , flagAmbience 
			#print 'Flag = ', flag
	
				
			if flagDish:
				
				#print 'predicting sentiment for sentence', sent

				############################
				### PREDICTING SENTIMENT ###
				############################
				test_time = time.clock()
				TESTDATA=StringIO("""Review
	 				|""" + sent)

				df1 = DataFrame.from_csv(TESTDATA, sep="|", parse_dates=False)
				#print df1
				test_time = time.clock()
				test_bow = vect.transform(df1['Review'])
				#print 'Data frame creation time = ', time.clock()-test_time
				
				prediction = mod.predict(test_bow)
				probability = mod.predict_proba(test_bow)
				#print 'Score = ', score , type(score), score[0][1] 
				#print prediction, type(str(prediction[0]))
				if str(prediction[0]) == '1':
					result = 'Positive'
					dict = {sent : result}
					score = probability[0][1]
					#print '++++++++++++++++++++'
					#print ' The Polarity of the review is  ', result
					# print 'Time to predict the review = ', time.clock() - timet
					# return render_template('posres.html', result = dict)
				else:
					result = 'Negative'
					dict = {sent : result}
					score = probability[0][0]
					#print '--------------------'
					#print ' The Polarity of the review is  ', result
					# print 'Time to predict the review = ', time.clock() - timet
					# return render_template('negres.html', result = dict)
				
				#print 'inserting noun,prediction,res_name,res_id,review_id into sqlite3 db'
				#print 'creating a list & jsonify it as output'
				#print 'Keyword %s is %s' % (noun, result)
				#cur.execute("INSERT INTO Contacts VALUES (?, ?, ?, ?);", (firstname, lastname, phone, email))
				
				for noun in nouns:
					if noun in dishlist and len(noun) > 3 and noun not in dup_dishes:
						dish_counter += 1
						
						#dishes

						#conn.execute("INSERT INTO RATINGS VALUES (?, ?, ?, ?, ?)", (counter, 'Mark', 25, noun, result))
						conn.execute("INSERT INTO SA_REVIEW_SCORE VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (dish_counter, review_index,res_names[review_index-1], dish_counter, noun, score, result, created_dates[review_index-1]))
						
						print 'Noun = %s & Sentiment = %s ' %(noun, result)
						dup_dishes.append(noun)
					else:
						pass 

				#conn.execute("INSERT INTO RATINGS VALUES (?, ?, ?, ?, ?)", (counter, 'Mark', 25, noun, result))

				conn.commit()
				#print "Records created successfully";
				
			
			if flagService:
				
				#print 'predicting sentiment for sentence', sent

				############################
				### PREDICTING SENTIMENT ###
				############################
				test_time = time.clock()
				TESTDATA=StringIO("""Review
	 				|""" + sent)

				df1 = DataFrame.from_csv(TESTDATA, sep="|", parse_dates=False)
				#print df1
				test_time = time.clock()
				test_bow = vect.transform(df1['Review'])
				#print 'Data frame creation time = ', time.clock()-test_time
				
				prediction = mod.predict(test_bow)
				probability = mod.predict_proba(test_bow)
				#print 'Score = ', score , type(score), score[0][1] 
				#print prediction, type(str(prediction[0]))
				if str(prediction[0]) == '1':
					serviceresult = 'Positive'
					
					#print '++++++++++++++++++++'
					print ' The Polarity of the review is  ', serviceresult
					# print 'Time to predict the review = ', time.clock() - timet
					# return render_template('posres.html', result = dict)
				else:
					serviceresult = 'Negative'
					print ' The Polarity of the review is  ', serviceresult
				
				if serviceresult == 'Positive':
					service_sentiment = '1'
				elif serviceresult == 'Negative':
					service_sentiment = '0'
				else:
					service_sentiment = '#'
					#print '--------------------'
					#print ' The Polarity of the review is  ', result
					# print 'Time to predict the review = ', time.clock() - timet
					# return render_template('negres.html', result = dict)
				
			if flagFood:
				
				#print 'predicting sentiment for sentence', sent

				############################
				### PREDICTING SENTIMENT ###
				############################
				test_time = time.clock()
				TESTDATA=StringIO("""Review
	 				|""" + sent)

				df1 = DataFrame.from_csv(TESTDATA, sep="|", parse_dates=False)
				#print df1
				test_time = time.clock()
				test_bow = vect.transform(df1['Review'])
				#print 'Data frame creation time = ', time.clock()-test_time
				
				prediction = mod.predict(test_bow)
				probability = mod.predict_proba(test_bow)
				#print 'Score = ', score , type(score), score[0][1] 
				#print prediction, type(str(prediction[0]))
				if str(prediction[0]) == '1':
					foodresult = 'Positive'
					
					#print '++++++++++++++++++++'
					#print ' The Polarity of the review is  ', result
					# print 'Time to predict the review = ', time.clock() - timet
					# return render_template('posres.html', result = dict)
				else:
					foodresult = 'Negative'
					
				
				if foodresult == 'Positive':
					food_sentiment = '1'
				elif foodresult == 'Negative':
					food_sentiment = '0'
				else:
					food_sentiment = '#'

					#print '--------------------'
					#print ' The Polarity of the review is  ', result
					# print 'Time to predict the review = ', time.clock() - timet
					# return render_template('negres.html', result = dict)


			if flagAmbience:
				
				print 'Inside Ambience'
				#print 'predicting sentiment for sentence', sent

				############################
				### PREDICTING SENTIMENT ###
				############################
				test_time = time.clock()
				TESTDATA=StringIO("""Review
	 				|""" + sent)

				df1 = DataFrame.from_csv(TESTDATA, sep="|", parse_dates=False)
				#print df1
				test_time = time.clock()
				test_bow = vect.transform(df1['Review'])
				#print 'Data frame creation time = ', time.clock()-test_time
				
				prediction = mod.predict(test_bow)
				probability = mod.predict_proba(test_bow)
				#print 'Score = ', score , type(score), score[0][1] 
				#print prediction, type(str(prediction[0]))
				if str(prediction[0]) == '1':
					ambienceresult = 'Positive'
					
					#print '++++++++++++++++++++'
					#print ' The Polarity of the review is  ', result
					# print 'Time to predict the review = ', time.clock() - timet
					# return render_template('posres.html', result = dict)
				else:
					ambienceresult = 'Negative'
					
				
				if ambienceresult == 'Positive':
					#print 'Positive'
					ambience_sentiment = '1'
					print ambience_sentiment
				elif ambienceresult == 'Negative':
					#print 'Negative'
					ambience_sentiment = '0'
					print ambience_sentiment
				else:
					ambience_sentiment = '#'
					print ambience_sentiment

					#print '--------------------'
					#print ' The Polarity of the review is  ', result
					# print 'Time to predict the review = ', time.clock() - timet
					# return render_template('negres.html', result = dict)

			
			if flagValue:
				
				#print 'predicting sentiment for sentence', sent

				############################
				### PREDICTING SENTIMENT ###
				############################
				test_time = time.clock()
				TESTDATA=StringIO("""Review
	 				|""" + sent)

				df1 = DataFrame.from_csv(TESTDATA, sep="|", parse_dates=False)
				#print df1
				test_time = time.clock()
				test_bow = vect.transform(df1['Review'])
				#print 'Data frame creation time = ', time.clock()-test_time
				
				prediction = mod.predict(test_bow)
				probability = mod.predict_proba(test_bow)
				#print 'Score = ', score , type(score), score[0][1] 
				#print prediction, type(str(prediction[0]))
				if str(prediction[0]) == '1':
					valueresult = 'Positive'
					
					#print '++++++++++++++++++++'
					#print ' The Polarity of the review is  ', result
					# print 'Time to predict the review = ', time.clock() - timet
					# return render_template('posres.html', result = dict)
				else:
					valueresult = 'Negative'
					
				
				if valueresult == 'Positive':
					value_sentiment = '1'
				elif valueresult == 'Negative':
					value_sentiment = '0'
				else:
					value_sentiment = '#'

					#print '--------------------'
					#print ' The Polarity of the review is  ', result
					# print 'Time to predict the review = ', time.clock() - timet
					# return render_template('negres.html', result = dict)

				#print 'inserting noun,prediction,res_name,res_id,review_id into sqlite3 db'
				#print 'creating a list & jsonify it as output'
				#print 'Keyword %s is %s' % (noun, result)
				#cur.execute("INSERT INTO Contacts VALUES (?, ?, ?, ?);", (firstname, lastname, phone, email))
			

				
			if noun in servicelist or noun in ambiencelist or noun in valuelist or noun in foodlist:
			#if noun in ambiencelist:
				dish_counter += 1
				
				#dishes

				#conn.execute("INSERT INTO RATINGS VALUES (?, ?, ?, ?, ?)", (counter, 'Mark', 25, noun, result))
				print 'Inside aspects_score printing module'
				print ' '
				print ' '
				print ambience_sentiment, value_sentiment
				conn.execute("INSERT INTO ASPECTS_SCORE VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (dish_counter, review_index,res_names[review_index-1], review_sentiment, service_sentiment, value_sentiment, ambience_sentiment, food_sentiment))
				
				# print 'Noun = %s & Sentiment = %s ' %(noun, result)
				# dup_dishes.append(noun)
			else:
				pass 

					#conn.execute("INSERT INTO RATINGS VALUES (?, ?, ?, ?, ?)", (counter, 'Mark', 25, noun, result))

			conn.commit()
				#print "Records created successfully";
		


	cursor = conn.execute("SELECT * from SA_REVIEW_SCORE")
	print 'Printing Stored values in SA_REVIEW_SCORE table'
	print "REVIEW_ID \t\t RES_NAME \t\t KEYWORD_ID \t\t DISH_NAME \t\t SCORE \t\t SENTIMENT \t\t CREATED_DATE "
	   
	for row in cursor:
	   
	   # print "REVIEW_ID = ", row[1]
	   # print "RES_NAME = ", row[2]
	   # print "KEYWORD_ID = ", row[3]
	   # print "DISH_NAME = ", row[4]
	   # print "SCORE = %3.2f" % (row[5])
	   # print "SENTIMENT = ", row[6]
	   # print "CREATED_DATE = ", row[7] ,"\n"

	   print "%d \t\t %s \t\t %d \t\t %s \t\t %3.2f \t\t %s \t\t %s" % (row[1],row[2],row[3],row[4],row[5],row[6],row[7])
	



	cursor1 = conn.execute("SELECT * from ASPECTS_SCORE")
	print 'Printing Stored values in ASPECTS_SCORE table'
	print "REVIEW_ID \t\t RES_NAME \t\t REVIEW_SENTIMENT \t\t SERVICE_SENTIMENT \t\t VALUE_SENTIMENT \t\t AMBIENCE_SENTIMENT \t\t FOOD_SENTIMENT "
	   
	for row in cursor1:
	   
	   # print "REVIEW_ID = ", row[1]
	   # print "RES_NAME = ", row[2]
	   # print "KEYWORD_ID = ", row[3]
	   # print "DISH_NAME = ", row[4]
	   # print "SCORE = %3.2f" % (row[5])
	   # print "SENTIMENT = ", row[6]
	   # print "CREATED_DATE = ", row[7] ,"\n"

	   print "%d \t\t %s \t\t %s \t\t %s \t\t %s \t\t %s \t\t %s" % (row[1],row[2],row[3],row[4],row[5],row[6],row[7])
	conn.close()



	print 'Program exiting.....'
	print 'Total time taken: ', time.clock() - start_time, ' seconds'

	return 0


main()