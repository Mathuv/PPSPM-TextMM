from __future__ import division, unicode_literals
import csv
import nltk
import re
import os
import errno
import sys
import string
import math
from types import *
from nltk.corpus import stopwords


# extract History of Present Illness - step 2
def extract_hpi(text):
    match = re.search(r'History of Present Illness:\s+((\S+\s)+)',text,re.IGNORECASE)
    return match.group(1) if match  else match

# tokenize text into list of words - step 3
def tokenize(text):
    #return re.split(r'[ \t\n]+', text) if text else text
    return nltk.word_tokenize(text) if text else text

# remove stop words - step 4
def remove_stopwords(tokens):
    stopwords = nltk.corpus.stopwords.words('english')
    return [word for word in tokens if word not in stopwords] if tokens else tokens

# stemming - step 5 
def stem(tokens):
    porter = nltk.PorterStemmer()
    lancaster = nltk.LancasterStemmer()
    return [porter.stem(word) if str(word).isalpha() else word for word in tokens ] if tokens else tokens # please correct this.

# tagging
def pos_tagging(tokens):
    return [nltk.pos_tag(word) for word in tokens] if tokens else tokens


# Calculate TF - Step 6.1
def tf(word, tokens):
    return float(tokens.count(word)) / len(tokens)  #DV: use float to return float values - as in Python 2.7 if both the values are not floating point values, it will result in integer - e.g. 2/5 = 0 instead of 0.4

# Num of records containing a word - Step 6.2
def n_containing(word, textlist):
    return sum(1 for blob in textlist if word in textlist)

# Calculate IDF - Step 6.3
def idf(word, textlist):
    return math.log(len(textlist) / (1 + n_containing(word, textlist)))

# Calculate TF-IDF - Step 6.4
def tfidf(word, tokens, textlist):
    return tf(word, tokens) * idf(word, textlist)


def write_file(content, file):
    csv_rows = []

    if not os.path.exists(os.path.dirname(file)):
        try:
            os.makedirs(os.path.dirname(file))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise

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


def main(dbfile, id_column_header, text_column_header, text_section_header):
    #dbfile = sys.argv[1]
    #dbfile = '../database/NOTEEVENTS_DATA_TABLE_PARTIAL_20REC.csv'
    dbpath = os.path.dirname(dbfile)
    dbfilename_ext = os.path.basename(dbfile)
    dbfilename = os.path.splitext(dbfilename_ext)[0]
    dbreader = csv.reader(open(dbfile))
    dbdata = list(dbreader)


    header_rec = dbdata[0]  #Patient table column headers

    try:
        id_column_no = header_rec.index(id_column_header)
    except:
        print id_column_header + ' is not found in ' + dbfilename_ext
        raise

    try:
        text_column_no = header_rec.index(text_column_header)
    except ValueError:
        print text_column_header + ' is not found in ' + dbfilename_ext
        raise

    dsr_list = []
    hpi_orginal = []
    hpi_list_tokenized = []
    hpi_list_stpwd_rm = []
    hpi_list_stemmed = []
    hpi_list_pos_tagged = []
    hpi_list_tfidf = []


    for row in dbdata[1:]:
        #unprocessed data
        dsr_list.append(row[id_column_no::text_column_no])

        # extract Histry of Present Illness - step 2
        row[text_column_no] = extract_hpi(row[text_column_no])
        hpi_orginal.append(row[id_column_no::text_column_no])

        # cleaning data - convert to lower case
        row[text_column_no] = row[text_column_no].lower()

        # cleaning data - punctuation removal
        row[text_column_no] = str(row[text_column_no]).translate(None, string.punctuation)

        # create tokenized list - step 3
        row[text_column_no] = tokenize(row[text_column_no])
        hpi_list_tokenized.append(row[id_column_no::text_column_no])

        # create stop word removed list - step 4
        row[text_column_no] = remove_stopwords(row[text_column_no])
        hpi_list_stpwd_rm.append(row[id_column_no::text_column_no])

        #pos tagging
        # row[10] = pos_tagging(row[10])
        # hpi_list_pos_tagged.append(row[0::10])

        #create stemmed list
        row[text_column_no] = stem(row[text_column_no])
        hpi_list_stemmed.append(row[id_column_no::text_column_no])

    # TF-IDF calculation - Step 6
    for rec in hpi_list_stemmed:
        scores = {token: tfidf(token,rec[1],[l[1] for l in hpi_list_stemmed]) for token in rec[1]}
        sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        hpi_list_tfidf.append([rec[0],sorted_words])
        
    # DV: can you also calculate TF_IDF before stemming (i.e. on the hpi_list_stpwd_rm) to see how the scores look like?
    # May be save the output as Step 6b


    # write discharge summery text csv - step 1
    raw_filename = dbpath+os.sep+'step1'+os.sep+dbfilename+'_RAW.csv'
    write_file(dsr_list,raw_filename)


    # write 'History of Present Illness' text csv - step 2
    hpi_filename = dbpath+os.sep+'step2'+os.sep+dbfilename+'_HPI.csv'
    write_file(hpi_orginal,hpi_filename)


    # write tokenized  'History of Present Illness' text file - step 3
    hpi_tokenized_filename = dbpath+os.sep+'step3'+os.sep+dbfilename+'_HPI_tokenized.csv'
    write_file(hpi_list_tokenized,hpi_tokenized_filename)

    #write stop-words removed - step 4
    hpi_stpwd_rm_filename = dbpath+os.sep+'step4'+os.sep+dbfilename+'_HPI_stpwd_rm.csv'
    write_file(hpi_list_stpwd_rm,hpi_stpwd_rm_filename)

    # write stemmed list - step 5
    hpi_stemmed_filename = dbpath + os.sep + 'step5' + os.sep + dbfilename + '_HPI_stemmed.csv'
    write_file(hpi_list_stemmed, hpi_stemmed_filename)

    # write pos tagged - step
    # hpi_pos_tagged_filename = dbpath + os.sep + 'step6' + os.sep + dbfilename + '_HPI_tagged.csv'
    # write_file(hpi_list_pos_tagged, hpi_pos_tagged_filename)

    # write if-idf output - Step 6
    hpi_tfidf_filename = dbpath + os.sep + 'step6' + os.sep + dbfilename + '_HPI_tfidf.csv'
    write_file(hpi_list_tfidf, hpi_tfidf_filename)
    
    # DV: as a next step (Step 7) select the top m words for each record and write in a csv file, so that we can compare the selected words before hash-mapping them



if __name__ == "__main__":
    main(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4])





