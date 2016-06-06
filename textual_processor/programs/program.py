import csv
import nltk
import re
import os
from types import *
from nltk.corpus import stopwords


def tokenize(text):
    return re.split(r'[ \t\n]+', text)

def remove_stopwords(tokens):
    pass

def stem(tokens):
    pass

def write_file2(content, file):
    f = open(file,'w')
    f.write('')
    f.close()
    f = open(file,'a')
    for line in content:
        if type(line[1]) is ListType:
            string = [line[0], ' '.join(line[1])]
        else:
            string = line
        f.write(','.join(string)+'\n')
    f.close()

def write_file(content, file):
    csv_rows = []
    for line in content:
        if type(line[1]) is ListType:
            line[1] = ' '.join(line[1])
            csv_rows.append(line)
        else:
            csv_rows.append(line)

    # csv_rows = content
    with open(file, "wb") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)



dbfilename = 'NOTEEVENTS_DATA_TABLE_PARTIAL_20REC'
dbfile = open('..'+os.sep+'database'+os.sep+dbfilename+'.csv')
dbreader = csv.reader(dbfile)
dbdata = list(dbreader)

stopwords = nltk.corpus.stopwords.words('english')
content =[]

header_rec = dbdata[:1]  #Patient table column headers

hpi_list = []
hpi_orginal = []
hpi_list_tokenized = []
pass

for row in dbdata[1:]:
    #unprocessed data
    hpi_orginal.append(row[0::10])
    row[10] = tokenize(row[10])
    # create tokenized list
    hpi_list_tokenized.append(row[0::10])

    #create stemmed list

    #content = [w for w in row if w.lower() not in stopwords]

#write csv
raw_filename = '..'+os.sep+'database'+os.sep+'step1'+os.sep+dbfilename+'_raw.csv'
write_file(hpi_orginal,raw_filename)

#write tokenized file
tokenized_filename = '..'+os.sep+'database'+os.sep+'step2'+os.sep+dbfilename+'_tokenized.csv'
write_file(hpi_list_tokenized,tokenized_filename)

#write stop-words removed

pass




