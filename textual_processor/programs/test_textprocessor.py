import subprocess
import sys


dbfile = '../database/NOTEEVENTS_DATA_TABLE_PARTIAL_20REC.csv'
queryfile = '../query/NOTEEVENTS_DATA_TABLE_PARTIAL_20REC.csv'
cmd_line_list = ['python', 'textprocessor.py', dbfile, queryfile, '1', '11', 'History of Present Illness:', '10']

party_proc = subprocess.Popen(cmd_line_list)


print '  Waiting for processes to complete...'
print

ret_code = party_proc.wait()
print 'finished.'

if (ret_code != 0):
  print 'returned code:', ret_code
  sys.exit()  # Stop experiment
