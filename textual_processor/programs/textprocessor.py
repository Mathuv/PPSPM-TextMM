import csv
import nltk
import re
import os
import errno
import sys
from types import *
from nltk.corpus import stopwords


# extract History of Present Illness - step 2
def extract_hpi(text):
    match = re.search(r'History of Present Illness:\s+((\S+\s)+)',text,re.IGNORECASE)
    return match.group(1) if match  else match
    
# tokenize text into list of words - step 3
def tokenize(text):
    return re.split(r'[ \t\n]+', text) if text else text

# remove stop words - step 4
def remove_stopwords(tokens):
    stopwords = nltk.corpus.stopwords.words('english')
    return [word for word in tokens if word not in stopwords] if tokens else tokens

# stemming - step 5
def stem(tokens):
    porter = nltk.PorterStemmer()
    lancaster = nltk.LancasterStemmer()
    return [porter.stem(word) for word in tokens if str(word).isalpha()] if tokens else tokens

def pos_tagging(tokens):
    return [nltk.pos_tag(word) for word in tokens] if tokens else tokens




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
            line[1] = ' '.join(line[1])
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


    for row in dbdata[1:]:
        #unprocessed data
        dsr_list.append(row[id_column_no::text_column_no])

        # extract Histry of Present Illness - step 2
        row[text_column_no] = extract_hpi(row[text_column_no])
        hpi_orginal.append(row[id_column_no::text_column_no])

        # create tokenized list - step 3
        row[text_column_no] = tokenize(row[text_column_no])
        hpi_list_tokenized.append(row[id_column_no::text_column_no])

        # create sstop word removed list - step 4
        row[text_column_no] = remove_stopwords(row[text_column_no])
        hpi_list_stpwd_rm.append(row[id_column_no::text_column_no])

        #pos tagging
        # row[10] = pos_tagging(row[10])
        # hpi_list_pos_tagged.append(row[0::10])

        #create stemmed list
        row[text_column_no] = stem(row[text_column_no])
        hpi_list_stemmed.append(row[id_column_no::text_column_no])



    # write discharge summery text csv - step1
    raw_filename = dbpath+os.sep+'step1'+os.sep+dbfilename+'_RAW.csv'
    write_file(dsr_list,raw_filename)


    # write 'History of Present Illness' text csv - step2
    hpi_filename = dbpath+os.sep+'step2'+os.sep+dbfilename+'_HPI.csv'
    write_file(hpi_orginal,hpi_filename)


    # write tokenized  'History of Present Illness' text file - step3
    hpi_tokenized_filename = dbpath+os.sep+'step3'+os.sep+dbfilename+'_HPI_tokenized.csv'
    write_file(hpi_list_tokenized,hpi_tokenized_filename)

    #write stop-words removed - step4
    hpi_stpwd_rm_filename = dbpath+os.sep+'step4'+os.sep+dbfilename+'_HPI_stpwd_rm.csv'
    write_file(hpi_list_stpwd_rm,hpi_stpwd_rm_filename)

    # write stemmed list - step5
    hpi_stemmed_filename = dbpath + os.sep + 'step5' + os.sep + dbfilename + '_HPI_stemmed.csv'
    write_file(hpi_list_stemmed, hpi_stemmed_filename)

    # write pos tagged - step
    # hpi_pos_tagged_filename = dbpath + os.sep + 'step6' + os.sep + dbfilename + '_HPI_stemmed.csv'
    # write_file(hpi_list_pos_tagged, hpi_pos_tagged_filename)


if __name__ == "__main__":
    main(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4])





