import csv
from dish_dict_keys import dish_dict_keys
#values = [1, 2, 3, 4, 5]
thecsv = csv.writer(open("final_dish_list.csv", 'wb'))
for value in dish_dict_keys:
    thecsv.writerow([value])