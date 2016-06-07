import csv
import nltk
import re
import os
import errno
from types import *
from nltk.corpus import stopwords


# extract History of Present Illness
def extract_hpl(text):
    match = re.search(r'History of Present Illness:\s+((\S+\s)+)',text,re.IGNORECASE)
    return match.group(1) if match  else match
    
# tokenize text into list of words
def tokenize(text):
    return re.split(r'[ \t\n]+', text) if text else text

def remove_stopwords(tokens):
    stopwords = nltk.corpus.stopwords.words('english')
    return [word for word in tokens if word not in stopwords] if tokens else tokens

def stem(tokens):
    pass


def write_file(content, file):
    csv_rows = []

    if not os.path.exists(os.path.dirname(file)):
        try:
            os.makedirs(os.path.dirname(file))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

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
    f.close()



dbfilename = 'NOTEEVENTS_DATA_TABLE_PARTIAL_20REC'
dbfile = open('..'+os.sep+'database'+os.sep+dbfilename+'.csv')
dbreader = csv.reader(dbfile)
dbdata = list(dbreader)

stopwords = nltk.corpus.stopwords.words('english')
content =[]

header_rec = dbdata[:1]  #Patient table column headers

dsr_list = []
hpi_orginal = []
hpi_list_tokenized = []
hpi_list_stpwd_rm = []


for row in dbdata[1:]:
    #unprocessed data
    dsr_list.append(row[0::10])
    
    # extract Histry of Present Illness
    row[10] = extract_hpl(row[10])
    hpi_orginal.append(row[0::10])

    # create tokenized list
    row[10] = tokenize(row[10])
    hpi_list_tokenized.append(row[0::10])

    # create sstop word removed list
    row[10] = remove_stopwords(row[10])
    hpi_list_stpwd_rm.append(row[0::10])

    #create stemmed list



# write dischard summery text csv
raw_filename = '..'+os.sep+'database'+os.sep+'step1'+os.sep+dbfilename+'_RAW.csv'
write_file(dsr_list,raw_filename)


# write 'History of Present Illness' text csv
hpi_filename = '..'+os.sep+'database'+os.sep+'step2'+os.sep+dbfilename+'_HPI.csv'
write_file(hpi_orginal,hpi_filename)


# write tokenized  'History of Present Illness' text file
hpi_tokenized_filename = '..'+os.sep+'database'+os.sep+'step3'+os.sep+dbfilename+'_HPI_tokenized.csv'
write_file(hpi_list_tokenized,hpi_tokenized_filename)

#write stop-words removed
hpi_stpwd_rm_filename = '..'+os.sep+'database'+os.sep+'step4'+os.sep+dbfilename+'_HPI_stpwd_rm.csv'
write_file(hpi_list_stpwd_rm,hpi_stpwd_rm_filename)

pass




