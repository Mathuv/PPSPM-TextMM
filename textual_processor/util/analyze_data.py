import os
import logging
import csv
import errno
from types import *

# Log everything, and send it to stderr.
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# db_file = '../database/NOTEEVENTS_DATA_TABLE_PARTIAL_10000REC.csv'
db_file = '../database/NOTEEVENTS_DATA_TABLE.csv'
query_file = '../query/NOTEEVENTS_DATA_TABLE_PARTIAL_20REC.csv'

a_dict = {}


def write_file(content, dest_file):
        csv_rows = []

        if not os.path.exists(os.path.dirname(dest_file)):
            try:
                os.makedirs(os.path.dirname(dest_file))
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise

        if type(content) is DictType:
            with open(dest_file, "wb") as f:
                writer = csv.writer(f)
                # writer.writerow(content.keys())
                # writer.writerow(content.values())
                writer.writerows(content.items())

        else:
            for line2 in content:
                if type(line2[1]) is ListType:
                    line2[1] = ' '.join(map(str, line2[1]))
                    csv_rows.append(line2)
                else:
                    csv_rows.append(line2)

            # csv_rows = content
            with open(dest_file, "wb") as f:
                writer = csv.writer(f)
                writer.writerows(csv_rows)
            f.close()

with open(db_file, 'rb') as rf:
    filereader = csv.reader(rf)
    count = 0
    for line in filereader:
        count += 1
        if count > 1:
            if line[6] + '_' + line[7] in a_dict:
                # a_dict[line[6] + '_' + line[7]].append(line[0])
                a_dict[line[6] + '_' + line[7]] += 1
            else:
                # a_dict[line[6] + '_' + line[7]] = [line[0]]
                a_dict[line[6] + '_' + line[7]] = 1

                
logging.debug('Number of records: %i' % count)
write_file(a_dict, './results/blocking_analysis_CATEGORY_DESCRIPTION.csv')
