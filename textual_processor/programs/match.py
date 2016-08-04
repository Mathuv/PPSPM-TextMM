import time
import os
import errno
import logging
from textprocessor import TextProc;

# Febrl modules
import auxiliary

# Log everything, and send it to stderr.
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

weight_list = ['TF', 'TF-IDF']
m_list = [1,5,10] # number of top matching recoords
t_list = [5,10,20,30,50] # number of tokens
id_col_no = 1
text_col_no = 11
blk_attr_list = [6,7] # block attribute column numbers

dbfile = '../database/preprocessed/NOTEEVENTS_DATA_TABLE_PARTIAL_20REC.csv'
queryfile = '../query/preprocessed/NOTEEVENTS_DATA_TABLE_PARTIAL_20REC_NOTEEVENTS_DATA_TABLE_PARTIAL_20REC.csv'

start_time_total = time.time()

# Log file to write the results
dbfilename_ext = os.path.basename(dbfile)
dbfilename = os.path.splitext(dbfilename_ext)[0]

tproc = TextProc(t_list,m_list, weight_list,1000, id_col_no, text_col_no, blk_attr_list)

tproc.read_preprocessed(dbfile, isDB=True)
tproc.read_preprocessed(queryfile, isDB=False)

for t in t_list:
    for m in m_list:
        for w in weight_list:
            result_file_name = '..' + os.sep + 'results' + os.sep + dbfilename + '_' + str(t) + '_' + str(m) + '_' + w

            tproc.select_t_tokens(t)

            # Blocking
            #
            start_time = time.time()
            tproc.build_BI()
            blocking_phase_time = time.time() - start_time

            # compare unmasked
            #
            start_time = time.time()
            tproc.compare_unmasked(w)
            tproc.find_m_similar(m)
            matching_phase_time = time.time() - start_time
            tproc.write_file(tproc.results_dict, result_file_name + '_unmasked')

            # compare masked
            #
            start_time = time.time()
            tproc.compare_masked(w)
            tproc.find_m_similar_masked(m)
            masked_matching_phase_time = time.time() - start_time
            tproc.write_file(tproc.mresults_dict, result_file_name + '_masked')

            # write effectiveness results (precision, recall, F1, and rank)
            # into the log file (one line per query record)
            #
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
            #
            tot_time = matching_phase_time + masked_matching_phase_time
            str_tot_time = '%.4f' % (tot_time)

            # Calculate total memory usage for PP-SPM
            #
            memo_usage = auxiliary.get_memory_usage()
            memo_usage_val = auxiliary.get_memory_usage_val()
            memo_usage_val = memo_usage_val if memo_usage_val else 0.0
            str_mem_usage = '%.4f' % (memo_usage_val)



            # write efficiency results into the log file
            #
            log_file.write(str_tot_time + ',' + str_mem_usage + os.linesep)

            for query in accuracy_dict:
                res_list = accuracy_dict[query]
                log_file.write(query)
                for res in res_list:
                    log_file.write(',' + str(res))
                log_file.write(os.linesep)

            log_file.close()


time_taken = time.time() - start_time_total
print '\ntime_taken2: %4f' % (time_taken)
logging.debug('Query preprocess time %4f' % time_taken)