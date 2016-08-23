import os
import csv

no_of_records = 10000
no_of_lines = 260000
source_file_path = '..' + os.sep + '..' + os.sep + '..' + os.sep + 'NOTEEVENTS_DATA_TABLE.csv'
destination_file_path = '..' + os.sep + 'database' + os.sep + 'NOTEEVENTS_DATA_TABLE_PARTIAL_' + str(no_of_records) + 'REC.csv'
count = 0

# to include header record
no_of_records += 1

with open(source_file_path, 'rb') as src_file:
    filereader = csv.reader(src_file)
    with open(destination_file_path, 'wb') as des_file:
        filewriter = csv.writer(des_file)
        for line in filereader:
            count += 1
            filewriter.writerow(line)
            if count == no_of_records:
                break



