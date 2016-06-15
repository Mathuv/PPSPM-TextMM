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
def extract_text(text,text_section_identifier):
    # match = re.search(r'History of Present Illness:\s+((\S+\s)+)',text,re.IGNORECASE)
    match = re.search(r''+text_section_identifier+'\s+((\S+\s)+)', text, re.IGNORECASE)
    return match.group(1) if match else match


# tokenize text into list of words - step 3
def tokenize(text):
    # return re.split(r'[ \t\n]+', text) if text else text
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
    return tokens.count(word) / len(tokens)


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


def main(dbfile, id_column_no, text_column_no, text_section_identifier, m):

    dbpath = os.path.dirname(dbfile)
    dbfilename_ext = os.path.basename(dbfile)
    dbfilename = os.path.splitext(dbfilename_ext)[0]
    dbreader = csv.reader(open(dbfile))
    dbdata = list(dbreader)

    header_rec = dbdata[0] # Patient table column headers

    assert len(header_rec) >= int(id_column_no) > 0, 'id column number is out of range'
    assert len(header_rec) >= int(text_column_no) > 0, 'text column number is out of range'

    dsr_list = []
    text_orginal = []
    text_list_tokenized = []
    text_list_stpwd_rm = []
    text_list_stemmed = []
    text_list_pos_tagged = []
    text_list_tfidf = []
    text_list_stpwd_rm_tfidf = []
    text_list_m_tokens = []

    for row in dbdata[1:]:
        # unprocessed data
        dsr_list.append(row[id_column_no-1::text_column_no-1])

        # extract Histry of Present Illness - step 2
        row[text_column_no-1] = extract_text(row[text_column_no-1],text_section_identifier)
        text_orginal.append(row[id_column_no-1::text_column_no-1])

        # cleaning data - convert to lower case
        row[text_column_no-1] = row[text_column_no-1].lower()

        # cleaning data - punctuation removal
        row[text_column_no-1] = str(row[text_column_no-1]).translate(None, string.punctuation)

        # create tokenized list - step 3
        row[text_column_no-1] = tokenize(row[text_column_no-1])
        text_list_tokenized.append(row[id_column_no-1::text_column_no-1])

        # create stop word removed list - step 4
        row[text_column_no-1] = remove_stopwords(row[text_column_no-1])
        text_list_stpwd_rm.append(row[id_column_no-1::text_column_no-1])

        # pos tagging
        # row[10] = pos_tagging(row[10])
        # text_list_pos_tagged.append(row[0::10])

        # create stemmed list
        row[text_column_no-1] = stem(row[text_column_no-1])
        text_list_stemmed.append(row[id_column_no-1::text_column_no-1])

    # TF-IDF calculation - Step 6
    for rec in text_list_stemmed:
        scores = {token: tfidf(token,rec[1],[l[1] for l in text_list_stemmed]) for token in rec[1]}
        sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        text_list_tfidf.append([rec[0],sorted_words])
        # top m tokens with highest tf_idf score
        text_list_m_tokens.append([rec[0],sorted_words[:int(m)]])
        
    # TF-IDF calculation before stemming - Step 6b
    for rec in text_list_stpwd_rm:
        scores = {token: tfidf(token, rec[1], [l[1] for l in text_list_stpwd_rm]) for token in rec[1]}
        sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        text_list_stpwd_rm_tfidf.append([rec[0], sorted_words])

    # write discharge summery text csv - step 1
    raw_filename = dbpath+os.sep+'step1'+os.sep+dbfilename+'_RAW.csv'
    write_file(dsr_list,raw_filename)

    # write 'History of Present Illness' text csv - step 2
    text_filename = dbpath+os.sep+'step2'+os.sep+dbfilename+'_HPI.csv'
    write_file(text_orginal,text_filename)

    # write tokenized  'History of Present Illness' text file - step 3
    text_tokenized_filename = dbpath+os.sep+'step3'+os.sep+dbfilename+'_HPI_tokenized.csv'
    write_file(text_list_tokenized,text_tokenized_filename)

    # write stop-words removed - step 4
    text_stpwd_rm_filename = dbpath+os.sep+'step4'+os.sep+dbfilename+'_HPI_stpwd_rm.csv'
    write_file(text_list_stpwd_rm,text_stpwd_rm_filename)

    # write stemmed list - step 5
    text_stemmed_filename = dbpath + os.sep + 'step5' + os.sep + dbfilename + '_HPI_stemmed.csv'
    write_file(text_list_stemmed, text_stemmed_filename)

    # write pos tagged - step
    # text_pos_tagged_filename = dbpath + os.sep + 'step6' + os.sep + dbfilename + '_HPI_tagged.csv'
    # write_file(text_list_pos_tagged, text_pos_tagged_filename)

    # write if-idf output - Step 6
    text_tfidf_filename = dbpath + os.sep + 'step6' + os.sep + dbfilename + '_HPI_tfidf.csv'
    write_file(text_list_tfidf, text_tfidf_filename)

    # write if-idf output - Step 6b
    text_stpwd_rm_tfidf_filename = dbpath + os.sep + 'step6b' + os.sep + dbfilename + '_HPI_stpwd_rm_tfidf.csv'
    write_file(text_list_stpwd_rm_tfidf, text_stpwd_rm_tfidf_filename)
    
    # write top m tokens with high if-idf score - Step 7
    text_m_tfidf_filename = dbpath + os.sep + 'step7' + os.sep + dbfilename + '_HPI_m_tfidf.csv'
    write_file(text_list_m_tokens, text_m_tfidf_filename)

if __name__ == "__main__":
    main(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5])





