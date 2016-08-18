# README #

### Private Medical Data Comparison  Functions for Similar Patient Matching ###

This particular implementation of privacy preserving similarity comparison between attributes of textual data of patient records extends existing functionalities of PPSPM framework ([http://dmm.anu.edu.au/PPSPM/](Link URL)) 

### Python requirements: ###

* Python runtime environment

        python 2.7.3
        (This is not tested in python3 runtime)

* Python libraries

        bitarray (0.8.1)
        nltk (3.2.1)

* NLTk corpuses

        Stopwords
        PorterStemmer
        LancasterStemmer



### Python command to run experiments: ###

* python test_textprocessor.py
    
        Runs preprocessing, masking and matching under one instance

* python preprocess.py

        Runs preprocessing on raw data
    
* python match.py

        Runs masking and matching on preprocessed data


### The folder structure: ###

1. database
1. logs
1. programs
1. query
1. results
1. util

#### 1. database ####

        Contains the database datasets, results from each preprocessing step (step1 - step8) and 
        final output of preprocessing under the directory /preprocessed

#### 2. logs ####

        Log files

#### 3. programs ####

        * auxiliary.pyc
        * match.py
        * TBF.py
        * test_textprocessor.py
        * textprocessor.py

#### 4. query ####

        Contains the query datasets, results from each preprocessing step (step1 - step8) and 
        final output of preprocessing under the directory /preprocessed

#### 5. results ####

        Final results showing matching records from database dataset for each record of query dataset

#### 6. util ####

        Some utility programs to create chunks of dataset and asses the raw data
        * analyze_data.py
        * create_db.py

### Contribution guidelines ###

* Writing tests
* Code review
* Other guidelines

### Who do I talk to? ###

* Repo owner or admin (GSoC2016 Student)

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
        