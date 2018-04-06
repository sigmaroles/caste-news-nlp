import os
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

MAX_LEN_WORD = 20

sep = os.sep

pathroot = '.' + sep + 'rawText'
path_root_processed = '.' + sep + 'processedText'
path_journ = [x for x in (os.listdir(pathroot))]
path_periods = ['2016-1S', '2016-2S', '2017-1S', '2017-2S']

table = str.maketrans(string.punctuation,'                                ')
stopwords = set(stopwords.words('english'))

def process_directory(rpath, path2):
    filenames = []
    for (_, __, files) in os.walk(rpath):
        filenames.extend(files)

    for fname in filenames:
        wfname = path2+sep+fname
        print ("writing "+wfname)
        
        with open(rpath+sep+fname, 'rt') as fh:
            text = fh.read()
        text = text.translate(table)
        words = word_tokenize(text)
        words = [w.lower() for w in words]
        words = [w for w in words if not w in stopwords]
        words = [w for w in words if w.isalpha()]
        words = [w for w in words if len(w)<MAX_LEN_WORD]
        os.makedirs(os.path.dirname(wfname), exist_ok=True)
        with open(wfname, 'wt') as fh:
            fh.write(' '.join(words))
        


if __name__=='__main__':

    for journal in path_journ:
        for period in path_periods:
            
            thispath = pathroot + sep + journal + sep + period
            thispath_processed = path_root_processed + sep + journal + sep + period
            
            print ("\n\n\n *************** initiating directory ",thispath)
            process_directory(thispath, thispath_processed)

    