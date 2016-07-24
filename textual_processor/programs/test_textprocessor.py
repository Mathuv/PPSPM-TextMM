import subprocess
import sys

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


for t in t_list:
  for m in m_list:
    for w in weight_list:
      cmd_line_list = ['python', 'textprocessor.py', dbfile, queryfile, '1', '11', regex_filter, t, m, w]

      party_proc = subprocess.Popen(cmd_line_list)


      print '  Waiting for processes to complete...'
      print

      ret_code = party_proc.wait()
      print 'finished.'

      if (ret_code != 0):
        print 'returned code:', ret_code
        sys.exit()  # Stop experiment
