from __future__ import division, unicode_literals
import csv
import nltk
import re
import os
import errno
import string
import math
from types import *
from TBF import *
import operator
import numpy
import time
import logging

# Febrl modules
import auxiliary

# Log everything, and send it to stderr.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()


def calc_sim_tf_idf(term_list1, freq_list1, idf_list1, term_list2, freq_list2, idf_list2):
    """Calculates the Dice's Coefficient Similarity between two list of tokens
        considering their term frequency and inverse document frequency"""

    sum_min = 0
    comm_terms = list(set(term_list1).intersection(set(term_list2)))
    if comm_terms:
        for tok in comm_terms:
            idx = term_list1.index(tok)
            sum_min += min(freq_list1[idx], freq_list2[idx]) * ((idf_list1[idx] + idf_list2[idx]) / 2)

        div1 = sum([i*j for i, j in zip(freq_list1, idf_list1)])
        div2 = sum([i*j for i, j in zip(freq_list2, idf_list2)])

        sim = 2 * sum_min / div1 + div2
    else:
        sim = 0.0
    return sim


def calc_sim_freq(term_list1, freq_list1, term_list2, freq_list2):
    """Calculates the Dice's Coefficient Similarity between two list of tokens
        considering their term frequency"""

    sum_min = 0
    comm_terms = list(set(term_list1).intersection(set(term_list2)))
    if comm_terms:
        for tok in comm_terms:
            sum_min += min(freq_list1[term_list1.index(tok)], freq_list2[term_list2.index(tok)])
        sim = 2 * sum_min / (sum(freq_list1) + sum(freq_list2))
    else:
        sim = 0.0
    return sim


def argsortdup(a):
    """A function to rank a list of values.
    """

    sorted_array = numpy.sort(a)
    ranked = []
    for item in a:
        ranked.append(sorted_array.searchsorted(item) + 1)
    return numpy.array(ranked)


class TextProc:

    def __init__(self, t, m, weight, length, id_col_no, text_col_no, blk_attr_index):
        self.t = t
        self.m = m
        self.weight = weight
        self.length = length
        self.id_col_no = id_col_no - 1
        self.text_col_no = text_col_no - 1
        self.blk_attr_index = blk_attr_index

        self.db_dict = {}
        self.query_dict = {}

        self.block_index = {}

        self.db_dict_t = {}
        self.query_dict_t = {}

        self.mdb_dict = {}
        self.mquery_dict = {}

        self.candidate_dict = {}
        self.results_dict = {}

        self.mcandidate_dict = {}
        self.mresults_dict = {}

        self.sim_dict = {}
        self.rank_dict = {}
        self.mrank_dict = {}

        self.idf_index = {}

    # extract History of Present Illness - step 2
    def extract_text(self, text, regex_filter):
        match = re.search(r'' + regex_filter, text, re.IGNORECASE)
        return match.group(1) if match else match

    # tokenize text into list of words - step 3
    def tokenize(self, text):
        return nltk.word_tokenize(text) if text else text

    # remove stop words - step 4
    def remove_stopwords(self, tokens):
        stopwords = nltk.corpus.stopwords.words('english')
        return [word for word in tokens if word not in stopwords] if tokens else tokens

    # stemming - step 5
    def stem(self, tokens):
        porter = nltk.PorterStemmer()
        return [porter.stem(word) if str(word).isalpha() else word for word in tokens] if tokens else tokens

    # tagging
    def pos_tagging(self, tokens):
        return [nltk.pos_tag(word) for word in tokens] if tokens else tokens

    # Calculate TF - Step 6.1
    def tf(self, word, tokens):
        return tokens.count(word) / len(tokens)

    # Num of records containing a word - Step 6.2
    def n_containing(self, word):
        return self.idf_index[word]

    # Calculate IDF - Step 6.3
    def idf(self, word, textlist):
        return math.log(len(textlist) / (1 + self.n_containing(word)))

    # Calculate TF-IDF - Step 6.4
    def tfidf(self, word, tokens, textlist):
        return self.tf(word, tokens) * self.idf(word, textlist)

    def create_idf_index(self, rec_tokens):
        for token in rec_tokens:
            if token not in self.idf_index:
                self.idf_index[token] = 1
            else:
                self.idf_index[token] += 1

    def write_file(self, content, dest_file):
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
                writer.writerows(content.items())

        else:
            for line in content:
                if type(line[1]) is ListType:
                    line[1] = ' '.join(map(str, line[1]))
                    csv_rows.append(line)
                else:
                    csv_rows.append(line)

            # csv_rows = content
            with open(dest_file, "wb") as f:
                writer = csv.writer(f)
                writer.writerows(csv_rows)
            f.close()

    def preprocess(self, dbfile, id_column_no, text_column_no, regex_filter, t, is_db):
        dbpath = os.path.dirname(dbfile)
        dbfilename_ext = os.path.basename(dbfile)
        dbfilename = os.path.splitext(dbfilename_ext)[0]
        dbreader = csv.reader(open(dbfile))
        count = 0

        # convert the column numbers to list index
        id_column_no -= 1
        text_column_no -= 1

        raw_text_list = []
        text_list_extract = []
        text_list_tokenized = []
        text_list_stpwd_rm = []
        text_list_stemmed = []
        # text_list_pos_tagged = []
        text_list_tfidf = []
        text_list_stpwd_rm_tfidf = []
        text_list_t_tokens = []
        text_list_t_tokens_tf = []

        columns_to_extract = []
        columns_to_extract.append(id_column_no)
        columns_to_extract.append(text_column_no)
        columns_to_extract.extend(self.blk_attr_index)

        if is_db:
            db_dict = self.db_dict
        else:
            db_dict = self.query_dict

        # for row in dbdata[1:]:
        for row in dbreader:
            count += 1

            # csv header fields
            if count == 1:
                header_rec = row  # Patient table column headers
                assert len(header_rec) > int(id_column_no) >= 0, 'id column number is out of range'
                assert len(header_rec) > int(text_column_no) >= 0, 'text column number is out of range'

            else:

                if logger.level == logging.DEBUG:
                    # unprocessed data
                    raw_text_list.append([row[id_column_no], row[text_column_no]])

                # extract part of the raw text - step 2
                row[text_column_no] = self.extract_text(row[text_column_no], regex_filter)
                if logger.level == logging.DEBUG:
                    text_list_extract.append([row[id_column_no], row[text_column_no]])

                if row[text_column_no]:
                    # cleaning data - convert to lower case
                    row[text_column_no] = row[text_column_no].lower()

                    # cleaning data - punctuation removal
                    row[text_column_no] = str(row[text_column_no]).translate(None, string.punctuation)

                    # create tokenized list - step 3
                    row[text_column_no] = self.tokenize(row[text_column_no])
                    if logger.level == logging.DEBUG:
                        text_list_tokenized.append([row[id_column_no], row[text_column_no]])

                    # create stop word removed list - step 4
                    row[text_column_no] = self.remove_stopwords(row[text_column_no])
                    if logger.level == logging.DEBUG:
                        text_list_stpwd_rm.append([row[id_column_no], row[text_column_no]])

                    # pos tagging
                    # row[10] = pos_tagging(row[10])
                    # text_list_pos_tagged.append(row[0::10])

                    # create stemmed list
                    row[text_column_no] = self.stem(row[text_column_no])

                    # rec = [row[id_column_no], row[text_column_no]]
                    # # add the block attribute values
                    # rec.extend([row[i] for i in self.blk_attr_index])

                    # remove unwanted data
                    rec = row
                    for i in range(len(rec)):
                        if i not in columns_to_extract:
                            rec[i] = ''

                    text_list_stemmed.append(rec)

                    # create idf_index
                    if is_db:
                        # build the inverted index for efficient idf calculation
                        self.create_idf_index(row[text_column_no])
                else:
                    continue

        # TF-IDF calculation - Step 6
        for rec in text_list_stemmed:
            # scores = {token: self.tfidf(token, rec[1], [l[1] for l in text_list_stemmed]) for token in rec[1]}
            scores = {token: self.tfidf(token, rec[text_column_no], text_list_stemmed) for token in rec[text_column_no]}
            sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            if logger.level == logging.DEBUG:
                text_list_tfidf.append([rec[id_column_no], sorted_words])
            # top t tokens with highest tf_idf score - step 7
            # text_list_t_tokens.append([rec[id_column_no], sorted_words[:int(t)]])
            rec_temp = [val for val in rec]
            # rec_temp = rec
            rec_temp[text_column_no] = sorted_words[:int(t)]
            text_list_t_tokens.append(rec_temp)

        if logger.level == logging.DEBUG:
            # TF-IDF calculation before stemming - Step 6b
            for rec in text_list_stpwd_rm:
                # scores = {token: self.tfidf(token, rec[1], [l[1] for l in text_list_stpwd_rm]) for token in rec[1]}
                scores = {token: self.tfidf(token, rec[1], text_list_stemmed) for token in rec[1]}
                sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                text_list_stpwd_rm_tfidf.append([rec[0], sorted_words])

        # top t tokens from each record with their local term count - step 8
        for rec1, rec2 in zip(text_list_t_tokens, text_list_stemmed):
            tflist = [(word[0], rec2[text_column_no].count(word[0])) for word in rec1[text_column_no]]
            text_list_t_tokens_tf.append([rec1[id_column_no], tflist])
            # dictionary of records to compare
            tf_idf_list = [(word[0], rec2[text_column_no].count(word[0]), word[1]) for word in rec1[text_column_no]]
            # db_rec = [tf_idf_list]
            # db_rec.extend(rec2[2:])
            rec_temp = [val for val in rec2]
            rec_temp[text_column_no] = tf_idf_list
            db_dict[rec1[id_column_no]] = rec_temp

        if logger.level == logging.DEBUG:
            # write raw text- step 1
            raw_filename = dbpath + os.sep + 'step1' + os.sep + dbfilename + '_RAW.csv'
            self.write_file(raw_text_list, raw_filename)

            # write extracted part of the raw text csv - step 2
            text_filename = dbpath + os.sep + 'step2' + os.sep + dbfilename + '_TEXT.csv'
            self.write_file(text_list_extract, text_filename)

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

            # write top t tokens with high if-idf score - Step 7
            text_m_tfidf_filename = dbpath + os.sep + 'step7' + os.sep + dbfilename + '_TEXT_m_tfidf.csv'
            self.write_file(text_list_t_tokens, text_m_tfidf_filename)

            # write top t tokens of each record with their term count - step 8
            text_m_tf_filename = dbpath + os.sep + 'step8' + os.sep + dbfilename + '_TEXT_m_tf.csv'
            self.write_file(text_list_t_tokens_tf, text_m_tf_filename)

    # reads the preprocessed data
    def read_preprocessed(self, prep_file, is_db):

        with open(prep_file, 'r') as f:
            filereader = csv.reader(f)
            for line in filereader:
                if is_db:
                    self.db_dict[line[0]] = eval(line[1])
                else:
                    self.query_dict[line[0]] = eval(line[1])

    def select_t_tokens(self, t):
        for (rec_id, clean_rec) in self.db_dict.iteritems():
            t_tokens = clean_rec[self.text_col_no][:t]

            rec = [val for val in clean_rec]
            self.db_dict_t[rec_id] = rec
            self.db_dict_t[rec_id][self.text_col_no] = t_tokens
        for (rec_id, clean_rec) in self.query_dict.iteritems():
            t_tokens = clean_rec[self.text_col_no][:t]

            rec = [val for val in clean_rec]
            self.query_dict_t[rec_id] = rec
            self.query_dict_t[rec_id][self.text_col_no] = t_tokens

    def build_BI(self):
        """Build block_index data structure to store the BKVs and
           the corresponding list of record identifiers in the database.
        """

        rec_dict = self.db_dict
        block_index = self.block_index
        blk_attr_index = self.blk_attr_index

        print 'Build Block Index for attributes:', blk_attr_index

        for (rec_id, clean_rec) in rec_dict.iteritems():
            compound_bkv = ""

            if not blk_attr_index:  # No blocking keys
                compound_bkv = 'No_blk'
            else:
                for attr in blk_attr_index:  # Process selected blocking attributes
                    attr_val = clean_rec[attr]
                    attr_encode = attr_val  # Actual categorical value as BKV
                    compound_bkv += attr_encode

            if compound_bkv in block_index:  # Block value in index, only add attribute value
                rec_id_list = block_index[compound_bkv]
                rec_id_list.append(rec_id)
            else:  # A new block, add block value and attribute value
                rec_id_list = [rec_id]
                block_index[compound_bkv] = rec_id_list

        print '    Generated %d blocks' % (len(block_index))

    def compare_masked(self, comp_type='TF'):
        """Compare bloom filter encoded query records with bloom filter encoded database records
            @comp_type - similarity calculation type
        """

        db_dict = self.db_dict_t
        query_dict = self.query_dict_t
        block_index = self.block_index
        blk_attr_index = self.blk_attr_index
        self.mcandidate_dict = {}
        mcandidate_dict = self.mcandidate_dict

        self.mdb_dict = {}
        self.mquery_dict = {}

        for (q_rec_id, q_clean_rec) in query_dict.iteritems():  # Iterate over query records
            bkv = ''
            if not blk_attr_index:
                bkv = 'No_blk'
            else:
                for attr in blk_attr_index:
                    attr_val = q_clean_rec[attr]
                    bkv += attr_val

            if bkv in block_index:
                cand_recs = block_index[bkv]  # candidate records for the query

                for db_rec_id in cand_recs:  # Iterate over candidate records
                    db_clean_rec = db_dict[db_rec_id]

                    tbf_q_rec = TBF()
                    tbf_db_rec = TBF()

                    q_term_list = [item[0] for item in q_clean_rec[self.text_col_no]]
                    db_term_list = [item[0] for item in db_clean_rec[self.text_col_no]]

                    # if len(q_term_list) == len(db_term_list):
                    if comp_type == 'TF':
                        q_freq_list = [item[1] for item in q_clean_rec[self.text_col_no]]
                        db_freq_list = [item[1] for item in db_clean_rec[self.text_col_no]]

                        # get the minimum length of tokens
                        if len(q_term_list) != len(db_term_list):
                            min_length = min(len(q_term_list), len(db_term_list))
                            q_term_list = q_term_list[:min_length]
                            db_term_list = db_term_list[:min_length]
                            q_freq_list = q_freq_list[:min_length]
                            db_freq_list = db_freq_list[:min_length]

                        # Create counting bloom filters with terms and their term frequencies
                        cbf_freq_q = tbf_q_rec.add_list_tfidf(q_term_list, q_freq_list)
                        cbf_freq_db = tbf_db_rec.add_list_tfidf(db_term_list, db_freq_list)

                        sim_val = mcalc_sim_freq(cbf_freq_q, cbf_freq_db)

                    elif comp_type == 'TF-IDF':
                        q_freq_list = [item[1] for item in q_clean_rec[self.text_col_no]]
                        q_idf_list = [item[2] for item in q_clean_rec[self.text_col_no]]

                        db_freq_list = [item[1] for item in db_clean_rec[self.text_col_no]]
                        db_idf_list = [item[2] for item in db_clean_rec[self.text_col_no]]

                        # get the minimum length of tokens
                        if len(q_term_list) != len(db_term_list):
                            min_length = min(len(q_term_list), len(db_term_list))
                            q_term_list = q_term_list[:min_length]
                            db_term_list = db_term_list[:min_length]
                            q_freq_list = q_freq_list[:min_length]
                            db_freq_list = db_freq_list[:min_length]
                            q_idf_list = q_idf_list[:min_length]
                            db_idf_list = db_idf_list[:min_length]

                        # Create couting bloom filter with terms, term frequencies and idf values
                        cbf_freq_q, cbf_idf_q = tbf_q_rec.add_list_tfidf(q_term_list, q_freq_list, q_idf_list)
                        cbf_freq_db, cbf_idf_db = tbf_db_rec.add_list_tfidf(db_term_list, db_freq_list, db_idf_list)

                        sim_val = mcalc_sim_tf_idf(cbf_freq_q, cbf_idf_q, cbf_freq_db, cbf_idf_db)

                    # Store similarity results in mcandidate_dict
                    if q_rec_id in mcandidate_dict:
                        this_rec_dict = mcandidate_dict[q_rec_id]
                        this_rec_dict[db_rec_id] = sim_val
                    else:
                        this_rec_dict = {db_rec_id: sim_val}
                        mcandidate_dict[q_rec_id] = this_rec_dict

    def compare_unmasked(self, comp_type='TF'):
        """compare unmasked query records with database records
            @comp_type - similarity calculation type
        """
        db_dict = self.db_dict_t
        query_dict = self.query_dict_t
        block_index = self.block_index
        blk_attr_index = self.blk_attr_index
        self.candidate_dict = {}
        candidate_dict = self.candidate_dict

        for (q_rec_id, q_clean_rec) in query_dict.iteritems():
            bkv = ''
            if not blk_attr_index:
                bkv = 'No_blk'
            else:
                for attr in blk_attr_index:
                    attr_val = q_clean_rec[attr]
                    bkv += attr_val

            if bkv in block_index:
                cand_recs = block_index[bkv]  # candidate records for the query
                for db_rec_id in cand_recs:  # Iterate over candidate records
                    db_clean_rec = db_dict[db_rec_id]

                    q_term_list = [item[0] for item in q_clean_rec[self.text_col_no]]
                    db_term_list = [item[0] for item in db_clean_rec[self.text_col_no]]

                    # if len(q_term_list) == len(db_term_list):
                    if comp_type == 'TF':
                        q_freq_list = [item[1] for item in q_clean_rec[self.text_col_no]]
                        db_freq_list = [item[1] for item in db_clean_rec[self.text_col_no]]

                        # get the minimum length of tokens
                        if len(q_term_list) != len(db_term_list):
                            min_length = min(len(q_term_list), len(db_term_list))
                            q_term_list = q_term_list[:min_length]
                            db_term_list = db_term_list[:min_length]
                            q_freq_list = q_freq_list[:min_length]
                            db_freq_list = db_freq_list[:min_length]

                        sim_val = calc_sim_freq(q_term_list, q_freq_list, db_term_list, db_freq_list)

                    elif comp_type == 'TF-IDF':
                        q_freq_list = [item[1] for item in q_clean_rec[self.text_col_no]]
                        q_idf_list = [item[2] for item in q_clean_rec[self.text_col_no]]

                        db_freq_list = [item[1] for item in db_clean_rec[self.text_col_no]]
                        db_idf_list = [item[2] for item in db_clean_rec[self.text_col_no]]

                        # get the minimum length of tokens
                        if len(q_term_list) != len(db_term_list):
                            min_length = min(len(q_term_list), len(db_term_list))
                            q_term_list = q_term_list[:min_length]
                            db_term_list = db_term_list[:min_length]
                            q_freq_list = q_freq_list[:min_length]
                            db_freq_list = db_freq_list[:min_length]
                            q_idf_list = q_idf_list[:min_length]
                            db_idf_list = db_idf_list[:min_length]

                        sim_val = calc_sim_tf_idf(q_term_list, q_freq_list, q_idf_list, db_term_list, db_freq_list, db_idf_list)

                    # Store similarity results in candidate_dict
                    if q_rec_id in candidate_dict:
                        this_rec_dict = candidate_dict[q_rec_id]
                        this_rec_dict[db_rec_id] = sim_val
                    else:
                        this_rec_dict = {db_rec_id: sim_val}
                        candidate_dict[q_rec_id] = this_rec_dict

    def find_m_similar(self, m):
        """Find m most similar records for each query based on the overall similarity.
        """

        candidate_dict = self.candidate_dict
        # m = self.m
        results_dict = self.results_dict
        self.sim_dict = {}
        sim_dict = self.sim_dict

        for query in candidate_dict:  # loop over query records
            this_query_res = []
            this_sim_res = []
            this_query_dict = candidate_dict[query]
            if this_query_dict != {}:
                # Sort by similarity and retrieve m records with highest similarity
                #
                sorted_this_query_dict = sorted(this_query_dict.items(), key=operator.itemgetter(1))
                if len(sorted_this_query_dict) >= m:
                    for x in range(m):
                        this_query_res.append(sorted_this_query_dict[-(x + 1)][0])
                        this_sim_res.append(sorted_this_query_dict[-(x + 1)][1])
                else:
                    for x in range(len(sorted_this_query_dict)):
                        this_query_res.append(sorted_this_query_dict[-(x + 1)][0])
                        this_sim_res.append(sorted_this_query_dict[-(x + 1)][1])

                results_dict[query] = this_query_res
                sim_dict[query] = this_sim_res

            else:  # No similar records
                results_dict[query] = this_query_res
                sim_dict[query] = this_sim_res

                # print 'matches:', results_dict
                # print 'matches similarities:', sim_dict

    def find_m_similar_masked(self, m):
        """Find m most similar records for each query based on the overall similarity
           in a privacy-preserving setting.
        """

        mcandidate_dict = self.mcandidate_dict
        # m = self.m
        mresults_dict = self.mresults_dict
        self.mrank_dict = {}
        mrank_dict = self.mrank_dict

        results_dict = self.results_dict
        sim_dict = self.sim_dict
        self.rank_dict = {}
        rank_dict = self.rank_dict

        assert results_dict, 'results_dict is empty'

        for query in mcandidate_dict:  # loop over query records
            this_query_res = []
            this_sim_res = []
            this_query_dict = mcandidate_dict[query]
            if this_query_dict != {}:
                # Sort by similarity and retrieve m records with highest similarity
                #
                sorted_this_query_dict = sorted(this_query_dict.items(), key=operator.itemgetter(1))
                if len(sorted_this_query_dict) >= m:
                    for x in range(m):
                        this_query_res.append(sorted_this_query_dict[-(x + 1)][0])
                        this_sim_res.append(sorted_this_query_dict[-(x + 1)][1])
                else:
                    for x in range(len(sorted_this_query_dict)):
                        this_query_res.append(sorted_this_query_dict[-(x + 1)][0])
                        this_sim_res.append(sorted_this_query_dict[-(x + 1)][1])

                # Rank similarities
                actual_rank_res = []
                masked_rank_res = []
                intersect_items = list(set(results_dict[query]).intersection(this_query_res))
                for inter in intersect_items:
                    pos = results_dict[query].index(inter)
                    actual_rank_res.append(sim_dict[query][pos])
                    pos = this_query_res.index(inter)
                    masked_rank_res.append(this_sim_res[pos])

                mresults_dict[query] = this_query_res
                rank_dict[query] = list(argsortdup(actual_rank_res))
                mrank_dict[query] = list(argsortdup(masked_rank_res))

            else:  # No similar records
                mresults_dict[query] = this_query_res
                mrank_dict[query] = []
                rank_dict[query] = []

                # print 'masked matches:', mresults_dict
                # print 'actual matches ranking:', rank_dict
                # print 'masked matches ranking:', mrank_dict

    def calculate_accuracy(self):
        """Calculate accuracy of privacy-preserving comparison
           using actual values comparison results as the truth data.
        """

        results_dict = self.results_dict
        mresults_dict = self.mresults_dict
        rank_dict = self.rank_dict
        mrank_dict = self.mrank_dict
        accuracy_dict = {}

        for query in mresults_dict:
            query_res = mresults_dict[query]
            actual_res = results_dict[query]

            tot_m = len(actual_res)
            m = len(query_res)
            tm = 0
            for res in query_res:
                if res in actual_res:
                    tm += 1

            # Calculate precision, recall, and F1 measures
            if m > 0:
                prec = tm / float(m)
            else:
                prec = 0
            if tot_m > 0.0:
                rec = tm / float(tot_m)
            else:
                rec = 0.0
            if (prec + rec) > 0.0:
                fsco = (2 * prec * rec) / float(prec + rec)
            else:
                fsco = 0.0

            # Calculate Spearman's rank correlation
            query_rank = mrank_dict[query]
            actual_rank = rank_dict[query]
            assert len(actual_rank) == len(query_rank)
            n = len(actual_rank)

            if n == 1:
                if actual_rank[0] == query_rank[0]:
                    rank_cor = 1.0
                else:
                    rank_cor = 0.0
            elif n == 0:
                rank_cor = 0.0
            else:
                dist_sqr = 0
                for x in range(n):
                    dist = actual_rank[x] - query_rank[x]
                    dist_sqr += dist ** 2
                dist_sqr *= 6
                tot_elem = float(n * (n ** 2 - 1))
                rank_cor = 1.0 - (dist_sqr / tot_elem)

            accuracy_dict[query] = [prec, rec, fsco, rank_cor]

        logging.info('accuracy_dict: %s' % (str(accuracy_dict)))
        # print 'accuracy_dict:', accuracy_dct
        return accuracy_dict


if __name__ == "__main__":
    # absolute path of db file
    db_file = sys.argv[1]
    # absolute path of query file
    query_file = sys.argv[2]
    # column number of record identifier (primary key)
    id_column_no = int(sys.argv[3])
    # column number of the column that contains the textual data
    text_column_no = int(sys.argv[4])
    # regular expression filter for textual data
    regex_filter = sys.argv[5]
    # number of tokens to select from each record
    t = int(sys.argv[6])
    # number of similar records
    m = int(sys.argv[7])
    # weight type of tokens in similarity calculation
    weight = sys.argv[8]

    length = 1000  # length of bloom filter
    # m = 10 # number of similar records

    tproc = TextProc(t, m, weight, length)

    # preprocess the databse records
    start_time = time.time()
    tproc.preprocess(db_file, id_column_no, text_column_no, regex_filter, t, True)
    preprocess_time_db = time.time() - start_time

    # preprocess the query records
    start_time = time.time()
    tproc.preprocess(query_file, id_column_no, text_column_no, regex_filter, t, False)
    preprocess_time_query = time.time() - start_time

    # Log file to write the results
    dbfilename_ext = os.path.basename(db_file)
    dbfilename = os.path.splitext(dbfilename_ext)[0]
    result_file_name = '..' + os.sep + 'results' + os.sep + dbfilename + '_' + str(t) + '_' + str(m) + '_' + weight
    # result_file = open(result_file_name, 'w')

    tproc.select_t_tokens(t)

    # compare unmasked
    start_time = time.time()
    tproc.compare_unmasked(weight)
    tproc.find_m_similar(m)
    matching_phase_time = time.time() - start_time
    tproc.write_file(tproc.results_dict, result_file_name + '_unmasked')

    # compare masked
    start_time = time.time()
    tproc.compare_masked(weight)
    tproc.find_m_similar_masked(m)
    masked_matching_phase_time = time.time() - start_time
    tproc.write_file(tproc.mresults_dict, result_file_name + '_masked')
    
    # write effectiveness results (precision, recall, F1, and rank) 
    # into the log file (one line per query record)
    accuracy_dict = tproc.calculate_accuracy()
    log_file_name = '..' + os.sep + 'logs' + os.sep + dbfilename + '_' + str(t) + '_' + str(m) + '_' + weight

    if not os.path.exists(os.path.dirname(log_file_name)):
        try:
            os.makedirs(os.path.dirname(log_file_name))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise

    log_file = open(log_file_name, 'w')

    # Calculate total runtime for PP-SPM
    tot_time = preprocess_time_db + preprocess_time_query + matching_phase_time + masked_matching_phase_time
    str_tot_time = '%.4f' % tot_time

    # Calculate total memory usage for PP-SPM
    memo_usage = auxiliary.get_memory_usage()
    memo_usage_val = auxiliary.get_memory_usage_val()
    memo_usage_val = memo_usage_val if memo_usage_val else 0.0
    str_mem_usage = '%.4f' % memo_usage_val

    # write efficiency results into the log file
    log_file.write(str(str_tot_time) + ',' + str_mem_usage + os.linesep)

    for query in accuracy_dict:
        res_list = accuracy_dict[query]
        log_file.write(query)
        for res in res_list:
            log_file.write(',' + str(res))
        log_file.write(os.linesep)

    log_file.close()
