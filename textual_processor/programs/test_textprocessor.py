import textprocessor
dbfile = '../database/NOTEEVENTS_DATA_TABLE_PARTIAL_20REC.csv'
queryfile = '../query/NOTEEVENTS_DATA_TABLE_PARTIAL_20REC.csv'
textprocessor.main(dbfile,queryfile, 1, 11, 'History of Present Illness:', 10)