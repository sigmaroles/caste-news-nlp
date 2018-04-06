import os

sep = os.sep

pathroot = '.' + sep + 'processedText'
paths_journ = [x for x in (os.listdir(pathroot))]
path_periods = ['2016-1S', '2016-2S', '2017-1S', '2017-2S']

# aim: for each journal, we want ONE big file

if __name__=='__main__':
    for journ in paths_journ:
        #print ("Journal : ", journ)
        with open(journ+'_ALL.txt', 'w') as wfh:
            for period in path_periods:
                #print ("Period : ", period, end="  ")
                
                jpath = pathroot + sep + journ + sep + period
                fnames = [x for x in os.listdir(jpath)]
                
                
                for fn in fnames:
                    with open(jpath + sep + fn, 'r') as rfh:
                        text = rfh.read()
                    wfh.write(text+'\n')
                    print ("concatenated file "+jpath+sep+fn)
