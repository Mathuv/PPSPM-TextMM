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
from TBF import TBF


class TextProc:

    def __init__(self, m, length):
        self.m = m
        self.length = length

        self.db_dict = {}
        self.query_dict = {}

        self.candidate_dict = {}
        self.results_dict = {}

        self.mcandidate_dict = {}
        self.mresults_dict = {}

        self.sim_dict = {}
        self.rank_dict = {}
        self.mrank_dict = {}



        # extract History of Present Illness - step 2
    def extract_text(self,text,text_section_identifier):
        # match = re.search(r'History of Present Illness:\s+((\S+\s)+)',text,re.IGNORECASE)
        match = re.search(r''+text_section_identifier+'\s+((\S+\s)+)', text, re.IGNORECASE)
        return match.group(1) if match else match


    # tokenize text into list of words - step 3
    def tokenize(self,text):
        # return re.split(r'[ \t\n]+', text) if text else text
        return nltk.word_tokenize(text) if text else text


    # remove stop words - step 4
    def remove_stopwords(self,tokens):
        stopwords = nltk.corpus.stopwords.words('english')
        return [word for word in tokens if word not in stopwords] if tokens else tokens


    # stemming - step 5
    def stem(self,tokens):
        porter = nltk.PorterStemmer()
        lancaster = nltk.LancasterStemmer()
        return [porter.stem(word) if str(word).isalpha() else word for word in tokens ] if tokens else tokens # please correct this.


    # tagging
    def pos_tagging(self,tokens):
        return [nltk.pos_tag(word) for word in tokens] if tokens else tokens


    # Calculate TF - Step 6.1
    def tf(self,word, tokens):
        return tokens.count(word) / len(tokens)

    # Num of records containing a word - Step 6.2
    def n_containing(self,word, textlist):
        return sum(1 for blob in textlist if word in textlist)


    # Calculate IDF - Step 6.3
    def idf(self, word, textlist):
        return math.log(len(textlist) / (1 + self.n_containing(word, textlist)))


    # Calculate TF-IDF - Step 6.4
    def tfidf(self, word, tokens, textlist):
        return self.tf(word, tokens) * self.idf(word, textlist)


    def write_file(self, content, file):
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


    def main(self, dbfile, queryfile, id_column_no, text_column_no, text_section_identifier, m):
        # Preprocess database file
        self.preprocess(dbfile, id_column_no, text_column_no, text_section_identifier, m)

        # Preprocess query file
        self.preprocess(queryfile, id_column_no, text_column_no, text_section_identifier, m)


    def preprocess(self, dbfile, id_column_no, text_column_no, text_section_identifier, m):
        dbpath = os.path.dirname(dbfile)
        dbfilename_ext = os.path.basename(dbfile)
        dbfilename = os.path.splitext(dbfilename_ext)[0]
        dbreader = csv.reader(open(dbfile))
        dbdata = list(dbreader)

        header_rec = dbdata[0]  # Patient table column headers

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
        text_list_m_tokens_tf = []

        rec_dict = {}

        for row in dbdata[1:]:
            # unprocessed data
            dsr_list.append(row[id_column_no - 1::text_column_no - 1])

            # extract Histry of Present Illness - step 2
            row[text_column_no - 1] = self.extract_text(row[text_column_no - 1], text_section_identifier)
            text_orginal.append(row[id_column_no - 1::text_column_no - 1])

            # cleaning data - convert to lower case
            row[text_column_no - 1] = row[text_column_no - 1].lower()

            # cleaning data - punctuation removal
            row[text_column_no - 1] = str(row[text_column_no - 1]).translate(None, string.punctuation)

            # create tokenized list - step 3
            row[text_column_no - 1] = self.tokenize(row[text_column_no - 1])
            text_list_tokenized.append(row[id_column_no - 1::text_column_no - 1])

            # create stop word removed list - step 4
            row[text_column_no - 1] = self.remove_stopwords(row[text_column_no - 1])
            text_list_stpwd_rm.append(row[id_column_no - 1::text_column_no - 1])

            # pos tagging
            # row[10] = pos_tagging(row[10])
            # text_list_pos_tagged.append(row[0::10])

            # create stemmed list
            row[text_column_no - 1] = self.stem(row[text_column_no - 1])
            text_list_stemmed.append(row[id_column_no - 1::text_column_no - 1])

        # TF-IDF calculation - Step 6
        for rec in text_list_stemmed:
            scores = {token: self.tfidf(token, rec[1], [l[1] for l in text_list_stemmed]) for token in rec[1]}
            sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            text_list_tfidf.append([rec[0], sorted_words])
            # top m tokens with highest tf_idf score - step 7
            text_list_m_tokens.append([rec[0], sorted_words[:int(m)]])

        # TF-IDF calculation before stemming - Step 6b
        for rec in text_list_stpwd_rm:
            scores = {token: self.tfidf(token, rec[1], [l[1] for l in text_list_stpwd_rm]) for token in rec[1]}
            sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            text_list_stpwd_rm_tfidf.append([rec[0], sorted_words])

        # top m tokens from each record with their local term count - step 8
        for rec1, rec2 in zip(text_list_m_tokens, text_list_stemmed):
            tflist = [(word[0], rec2[1].count(word[0])) for word in rec1[1]]
            text_list_m_tokens_tf.append([rec1[0], tflist])
            # dictionary of records to compare
            tf_idf_list = [(word[0], rec2[1].count(word[0]), word[1]) for word in rec1[1]]
            rec_dict[rec1[0]] = tf_idf_list

        # write discharge summery text csv - step 1
        raw_filename = dbpath + os.sep + 'step1' + os.sep + dbfilename + '_RAW.csv'
        self.write_file(dsr_list, raw_filename)

        # write 'History of Present Illness' text csv - step 2
        text_filename = dbpath + os.sep + 'step2' + os.sep + dbfilename + '_TEXT.csv'
        self.write_file(text_orginal, text_filename)

        # write tokenized  'History of Present Illness' text file - step 3
        text_tokenized_filename = dbpath + os.sep + 'step3' + os.sep + dbfilename + '_TEXT_tokenized.csv'
        self.write_file(text_list_tokenized, text_tokenized_filename)

        # write stop-words removed - step 4
        text_stpwd_rm_filename = dbpath + os.sep + 'step4' + os.sep + dbfilename + '_TEXT_stpwd_rm.csv'
        self.write_file(text_list_stpwd_rm, text_stpwd_rm_filename)

        # write stemmed list - step 5
        text_stemmed_filename = dbpath + os.sep + 'step5' + os.sep + dbfilename + '_TEXT_stemmed.csv'
        self.write_file(text_list_stemmed, text_stemmed_filename)

        # write pos tagged - step
        # text_pos_tagged_filename = dbpath + os.sep + 'step6' + os.sep + dbfilename + '_TEXT_tagged.csv'
        # write_file(text_list_pos_tagged, text_pos_tagged_filename)

        # write if-idf output - Step 6
        text_tfidf_filename = dbpath + os.sep + 'step6' + os.sep + dbfilename + '_TEXT_tfidf.csv'
        self.write_file(text_list_tfidf, text_tfidf_filename)

        # write if-idf output - Step 6b
        text_stpwd_rm_tfidf_filename = dbpath + os.sep + 'step6b' + os.sep + dbfilename + '_TEXT_stpwd_rm_tfidf.csv'
        self.write_file(text_list_stpwd_rm_tfidf, text_stpwd_rm_tfidf_filename)

        # write top m tokens with high if-idf score - Step 7
        text_m_tfidf_filename = dbpath + os.sep + 'step7' + os.sep + dbfilename + '_TEXT_m_tfidf.csv'
        self.write_file(text_list_m_tokens, text_m_tfidf_filename)

        # write top m tokens of each record with their term count - step 8
        text_m_tf_filename = dbpath + os.sep + 'step8' + os.sep + dbfilename + '_TEXT_m_tf.csv'
        self.write_file(text_list_m_tokens_tf, text_m_tf_filename)

        return rec_dict

    def compare_masked(self):

        rec_dict = self.db_dict
        query_dict = self.query_dict
        m = self.m
        length = self.length

        val_list = []
        freq_list = []

        for (rec_id, clean_rec) in query_dict.iteritems():
            val_list = [item[0] for item in clean_rec]
            freq_list = [item[1] for item in clean_rec]

            tbf_q_rec = TBF()
            tbf_q_rec.add_list(val_list, freq_list)



    def compare_unmasked(self):
        pass

if __name__ == "__main__":
    db_file = sys.argv[1]
    query_file = sys.argv[2]
    id_column_no = int(sys.argv[3])
    text_column_no = int(sys.argv[4])
    text_section_identifier = sys.argv[5]
    m = int(sys.argv[6])

    length = 1000

    tproc = TextProc(m, length)

    # preprocess the databse records
    tproc.db_dict =  tproc.preprocess(db_file, id_column_no, text_column_no, text_section_identifier, m)

    # preprocess the query records
    tproc.query_dict =  tproc.preprocess(query_file, id_column_no, text_column_no, text_section_identifier, m)

    # read db to compare

    # read query to compare







