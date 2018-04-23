import csv
import sqlite3

nouns_list = []
conn = sqlite3.connect('nouns_30000_3_2.db')
print 'Database opened successfully !!!!'


nouns = conn.execute("SELECT NOUNS FROM NOUNS")
#print nouns

for noun in nouns:
	[nouns_list.append(str(noun[0]))]


print nouns_list
print len(nouns_list)

nouns_list = set(nouns_list)
print len(nouns_list)

f = open('Nouns_30000_3_2.txt','w')

for noun in nouns_list:
	f.write(noun+'\n')

f.close()



#for noun in nouns_list:
