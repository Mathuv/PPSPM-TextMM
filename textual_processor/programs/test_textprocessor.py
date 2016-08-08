import subprocess
import sys
import time
import os
import errno
import logging

# Febrl modules
import auxiliary

# Log everything, and send it to stderr.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# number of similar eecords
m_list = ['1','5','10']
# number of tokens to represent a record
t_list = ['5', '10', '20', '30', '50'] # DV: experiment more to plot the results - 5, 10, 20, 30, 50
weight_list = ['TF', 'TF-IDF']

dbfile = '../database/NOTEEVENTS_DATA_TABLE_PARTIAL_20REC.csv'
queryfile = '../query/NOTEEVENTS_DATA_TABLE_PARTIAL_20REC.csv'

# Regex filter to filter a section of the texttual data
# regex_filter = r'(.*\s*)*' # filters everything
# regex_filter = r'History of Present Illness:\s+((\S+\s)+)'
regex_filter = r'History of Present Illness:\s+((\S+([\t ]{1,2}|\n?))+)'

# cmd_line_list = ['python', 'textprocessor.py', dbfile, queryfile, '1', '11', regex_filter, '10' ,'10']

start_time_temp = time.time()

# for t in t_list:
#      for m in m_list:
#          for w in weight_list:
#              cmd_line_list = ['python', 'textprocessor.py', dbfile, queryfile, '1', '11', regex_filter, t, m, w]
#
#              party_proc = subprocess.Popen(cmd_line_list)
#
#
#              print '  Waiting for processes to complete...'
#              print
#
#              ret_code = party_proc.wait()
#              print 'finished.'
#
#              if (ret_code != 0):
#                print 'returned code:', ret_code
#                sys.exit()  # Stop experiment


time_taken1 = time.time() - start_time_temp


from textprocessor import TextProc;

m_list2 = [1,5,10]
t_list2 = [5,10,20,30,50]
blk_attr_list = [6,7]
id_col_no = 1
text_col_no = 11
regex_filter = r'History of Present Illness:\s+((\S+([\t ]{1,2}|\n?))+)'

start_time_total = time.time()

tproc = TextProc(t_list2,m_list2, weight_list,1000, id_col_no, text_col_no, blk_attr_list)

# Preprocess DB records
start_time = time.time()
tproc.preprocess(dbfile, 1, 11, regex_filter, max(t_list2), True)
preprocess_time_db = time.time() - start_time


# Preprocess query records
start_time = time.time()
tproc.preprocess(queryfile, id_col_no, text_col_no, regex_filter, max(t_list2), False)
preprocess_time_query = time.time() - start_time

# Blocking
#
start_time = time.time()
tproc.build_BI()
blocking_phase_time = time.time() - start_time


# Log file to write the results
dbfilename_ext = os.path.basename(dbfile)
dbfilename = os.path.splitext(dbfilename_ext)[0]

for t in t_list2:
    for m in m_list2:
        for w in weight_list:
            result_file_name = '..' + os.sep + 'results' + os.sep + dbfilename + '_' + str(t) + '_' + str(m) + '_' + w

            tproc.select_t_tokens(t)

            # compare unmasked
            start_time = time.time()
            tproc.compare_unmasked(w)
            tproc.find_m_similar(m)
            matching_phase_time = time.time() - start_time
            tproc.write_file(tproc.results_dict, result_file_name + '_unmasked')

            # compare masked
            start_time = time.time()
            tproc.compare_masked(w)
            tproc.find_m_similar_masked(m)
            masked_matching_phase_time = time.time() - start_time
            tproc.write_file(tproc.mresults_dict, result_file_name + '_masked')

            # write effectiveness results (precision, recall, F1, and rank)
            # into the log file (one line per query record)
            accuracy_dict = tproc.calculate_accuracy()
            log_file_name = '..' + os.sep + 'logs' + os.sep + dbfilename + '_' + str(t) + '_' + str(m) + '_' + w

            if not os.path.exists(os.path.dirname(log_file_name)):
                try:
                    os.makedirs(os.path.dirname(log_file_name))
                except OSError as exc:
                    if exc.errno != errno.EEXIST:
                        raise

            log_file = open(log_file_name, 'w')

            # Calculate total runtime for PP-SPM
            tot_time = preprocess_time_db + preprocess_time_query + matching_phase_time + masked_matching_phase_time
            str_tot_time = '%.4f' % (tot_time)

            # Calculate total memory usage for PP-SPM
            memo_usage = auxiliary.get_memory_usage()
            memo_usage_val = auxiliary.get_memory_usage_val()
            memo_usage_val = memo_usage_val if memo_usage_val else 0.0
            str_mem_usage = '%.4f' % (memo_usage_val)



            # write efficiency results into the log file
            log_file.write(str_tot_time + ',' + str_mem_usage + os.linesep)

            for query in accuracy_dict:
                res_list = accuracy_dict[query]
                log_file.write(query)
                for res in res_list:
                    log_file.write(',' + str(res))
                log_file.write(os.linesep)

            log_file.close()


time_taken2 = time.time() - start_time_total
print '\ntime_taken1: %4f' % (time_taken1)
print '\ntime_taken2: %4f' % (time_taken2)

pass















