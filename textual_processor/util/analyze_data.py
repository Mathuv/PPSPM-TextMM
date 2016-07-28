import time
import os
import logging
import csv
import errno
from types import *

# Log everything, and send it to stderr.
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

db_file = '../database/NOTEEVENTS_DATA_TABLE_PARTIAL_1000REC.csv'
query_file = '../query/NOTEEVENTS_DATA_TABLE_PARTIAL_20REC.csv'

a_dict = {}


def write_file( content, file):
        csv_rows = []

        if not os.path.exists(os.path.dirname(file)):
            try:
                os.makedirs(os.path.dirname(file))
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise

        if type(content) is DictType:
            with open(file, "wb") as f:
                writer = csv.writer(f)
                # writer.writerow(content.keys())
                # writer.writerow(content.values())
                writer.writerows(content.items())

        else:
            for line in content:
                if type(line[1]) is ListType:
                    line[1] = ' '.join(map(str,line[1]))
                    csv_rows.append(line)
                else:
                    csv_rows.append(line)

            # csv_rows = content
            with open(file, "wb") as f:
                writer = csv.writer(f)
                writer.writerows(csv_rows)
            f.close()

with open(db_file, 'rb') as f:
    filereader = csv.reader(f)
    count = 0
    for line in filereader:
        count += 1
        if count > 1:
            if line[6] + '_' + line[7] in a_dict:
                a_dict[line[6] + '_' + line[7]] += 1
            else:
                a_dict[line[6] + '_' + line[7]] = 1
                

write_file(a_dict, './results/blocking_analysis.csv')              


