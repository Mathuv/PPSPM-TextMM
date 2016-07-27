import time
import os
import logging
import csv

# Log everything, and send it to stderr.
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

db_file = '../database/NOTEEVENTS_DATA_TABLE_PARTIAL_1000REC.csv'
query_file = '../query/NOTEEVENTS_DATA_TABLE_PARTIAL_20REC.csv'

with open(db_file, 'rb') as f:
    filereader = csv.reader(f)
    a_dict = {}
    count = 0
    for line in filereader:
        count += 1
        if count > 1:
            if line[6] + '_' + line[7] in a_dict:
                a_dict[line[6] + '_' + line[7]] += 1
            else:
                a_dict[line[6] + '_' + line[7]] = 1