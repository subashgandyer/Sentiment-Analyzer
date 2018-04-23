import csv
# with open('cuisine_2000_dishes.csv', 'rb') as input:
#      print zip(csv.reader(input, delimiter = ','))
    
#      dishes_list = []
#      reader = csv.reader(input, delimiter = ',')
#      dishes_list.append(reader)
   
# print dishes_list


# import csv
with open('SanMateoNouns.csv', 'rb') as f:
    reader = csv.reader(f)
    #print reader
    your_list = list(reader)

print len(your_list)

#print your_list

nouns_list = []
for i in range(len(your_list)):
	if len(your_list[i][0]) <= 3:
		pass
	else:
		nouns_list.append(your_list[i][0])


#print nouns_list
print len(nouns_list)

setlist = list(set(nouns_list))
print len(setlist)
#values = [1, 2, 3, 4, 5]
thecsv = csv.writer(open("SanMateoNounslist.csv", 'wb'))
for value in setlist:
    thecsv.writerow([value])

