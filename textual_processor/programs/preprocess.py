import time
import os
import logging
from textprocessor import TextProc;

# Febrl modules
import auxiliary

# Log everything, and send it to stderr.
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

t_list = [5, 10, 20, 30, 50] # number of tokens
id_col_no = 1
text_col_no = 11
blk_attr_list = [6,7] # block attribute column numbers

db_file = '../database/NOTEEVENTS_DATA_TABLE_PARTIAL_200REC.csv'
query_file = '../query/NOTEEVENTS_DATA_TABLE_PARTIAL_20REC.csv'

# Regex filter to filter a section of the texttual data
# regex_filter = r'(.*\s*)*' # filters everything
# regex_filter = r'History of Present Illness:\s+((\S+\s)+)'
regex_filter = r'History of Present Illness:\s+((\S+([\t ]{1,2}|\n?))+)'

start_time_db = time.time()
tproc = TextProc(t_list, None, None, None, id_col_no, text_col_no, blk_attr_list)

db_path = os.path.dirname(db_file)
db_filename_ext = os.path.basename(db_file)
db_filename = os.path.splitext(db_filename_ext)[0]
db_preprocesed = db_path + os.sep + 'preprocessed' + os.sep + db_filename_ext

# preprocess database dataset
tproc.preprocess(db_file, id_col_no, text_col_no, regex_filter, max(t_list), isDB=True)
tproc.write_file(tproc.db_dict, db_preprocesed)
preprocess_time_db = time.time() - start_time_db

start_time_query = time.time()

query_path = os.path.dirname(query_file)
query_filename_ext = os.path.basename(query_file)
query_filename = os.path.splitext(query_filename_ext)[0]
# query_preprocesed = query_path + os.sep + 'preprocessed' + os.sep + query_filename_ext
query_preprocesed = query_path + os.sep + 'preprocessed' + os.sep + query_filename + '_' + db_filename + '.csv'

# preproess query dataset
tproc.preprocess(query_file, id_col_no, text_col_no, regex_filter, max(t_list), isDB=False)
tproc.write_file(tproc.query_dict, query_preprocesed)
preprocess_time_query = time.time() - start_time_query
end_time = time.time()

logging.debug('DB preprocess time %4f' % preprocess_time_db)
logging.debug('Query preprocess time %4f' % preprocess_time_query)
logging .debug('Total preprocess time %4f' %(end_time - start_time_db))



