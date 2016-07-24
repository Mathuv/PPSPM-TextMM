import os

no_of_records = 1000
no_of_lines = 260000
source_file_path = '..' + os.sep + '..' + os.sep + '..' + os.sep + 'NOTEEVENTS_DATA_TABLE.csv'
destination_file_path = '..' + os.sep + 'database' + os.sep + 'NOTEEVENTS_DATA_TABLE_PARTIAL_' + str(no_of_records) + 'REC.csv'
count = 0
with open(source_file_path,'r') as src_file:
    for line in src_file:
        count += 1
        with open(destination_file_path,'a') as des_file:
            des_file.writelines(line)
        if count == no_of_lines:
            break

