# README #

## PPSPM-TextMM ##

### Private Medical Data Comparison  Functions for Similar Patient Matching ###

This particular implementation of privacy preserving similarity comparison between attributes of textual data of patient records extends existing functionalities of PPSPM framework ([https://github.com/Mathuv/PPSPM](https://github.com/Mathuv/PPSPM)) done under Google Summer Of Code 2016

Detailed Project Documentation can be found under [https://docs.google.com/document/d/1HhX-R-VDQh3sSZ68pkLvRzGBVc0ib0NxJWMhTKhFslA/edit?usp=sharing](https://docs.google.com/document/d/1HhX-R-VDQh3sSZ68pkLvRzGBVc0ib0NxJWMhTKhFslA/edit?usp=sharing)

### Python requirements: ###

* Python runtime environment

        python 2.7.3
        (This is not tested in python3 runtime)

* Python standard libraries

        bitarray (0.8.1)
        nltk (3.2.1)
        numpy (1.11.0)

* Additional Python libraries

        Febrl (https://sourceforge.net/projeLink URLjects/febrl/)
        PPSPM (http://dmm.anu.edu.au/PPSPM/)

* NLTK corpuses

        Stopwords
        PorterStemmer
        LancasterStemmer



### Python commands to run experiments ###

* python test_textprocessor.py
    
        Runs preprocessing, masking and matching under one instance

* python preprocess.py

        Runs preprocessing separately on raw data
    
* python match.py

        Runs masking and matching separately on preprocessed data


### The folder structure: ###

1. database
1. logs
1. programs
1. query
1. results
1. util

#### 1. database ####

  Contains the database datasets, results from each preprocessing step (step1 - step8) and final output of preprocessing under the directory /preprocessed

#### 2. logs ####

  Log files to record memory usage, execution time and precision, recall and f-measure (efficiency and effectiveness)

#### 3. programs ####

  This is the main folder containing the python implementation of PPSPM-TextMM.
       
        * match.py
        * TBF.py
        * test_textprocessor.py
        * textprocessor.py
        * auxiliary.py

#### 4. query ####

  Contains the query datasets, results from each preprocessing step (step1 - step8) and 
        final output of preprocessing under the directory /preprocessed

#### 5. results ####

  Final results showing matching records from database dataset for each record of query dataset

#### 6. util ####

  Some utility programs to create chunks of dataset and asses the raw data

        * analyze_data.py
        * create_db.py



### Who do I talk to? ###

* Repo owner or admin

        Mathu Mounasamy
        mmounasamy@gmail.com

* Advisers

        Dr Dinusha Vatsalan
        RESEARCH FELLOW
        Research School of Computer Science
        Australian National University (ANU)
        dinusha.vatsalan@anu.edu.au

        Professor Peter Christen
        PROFESSOR        
        Research School of Computer Science
        Australian National University (ANU)
        peter.christen@anu.edu.au


### License ###

The PPSPM-TextMM software is licensed under the [GNU General Public License,
version 3 (GPL-3.0)](https://www.gnu.org/licenses/gpl-3.0.en.html).
