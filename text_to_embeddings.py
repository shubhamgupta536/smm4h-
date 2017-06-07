import sys
sys.path.append('/datadrive/nlp/jasper/w2v/godin/word2vec_twitter_model/')
from word2vecReader import Word2Vec
from nltk.corpus import words
import io
import numpy
from spacy.en import English
import numpy as np
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('model_path_',help='Path of the word2vec twitter model bin file')
    parser.add_argument('input_file',help='File to read the input text data')
    parser.add_argument('output_file',help='File to write the vectors')
    args = parser.parse_args()
    
    #model_path = args.model_path_
    print("Loading the model, this can take some time...")
    model = Word2Vec.load_word2vec_format(args.model_path_, binary=True)
    print("Loafing spaCy model, this can take some time...")
    nlp=English()
    #print(("The vocabulary size is: "+str(len(model.vocab))))
    #print("Vector for 'Shubham': " + str(model['Shubham']))
    #print("Embedding dimension: " + str(len(model['Shubham'])))
    #f1=open("vec1.txt","w")
    f1=open(args.output_file,'w')
    zero=[0] * 400
    #Specify encoding with io.open (careful io.open is slow, don't use for large files)
    #latin-1 is usually the culprit if the files aren't utf-8 encoded
    #with io.open("new_latin-1.txt", "r", encoding='latin-1') as f:
    with io.open(args.input_file, "r", encoding='latin-1') as f:
        for line in f:
            #spaCy would do this better :)
            #row=line.split()
            doc = nlp(line)
            arr = []
            #for i in range(0,len(doc)):
            for token in doc:
                try:
                    embedding = model[token.text]
                    print("Success for:\t" + token.text)
                except KeyError:
                    print("Fail for:\t" + token.text)
                    #TODO: set embedding to zero vector instead of continue
                    embedding = zero
                #temp=str(model[row[i]])
                #temp.replace('\n',' ')
                #f1.write(temp)
                arr.append(embedding)
                #TODO: write as one line using join method
                #f1.write(str(embedding))
                #f1.write(" ")
            rows,cols=np.shape(arr)
            temp = arr[0]
            for i in range(1,rows):
                temp=np.concatenate((temp,arr[i]),axis=0)
            rand=' '.join(map(str,temp))
            f1.write(rand)
            f1.write("\n")
            

    """
    #utf-8 is the preffered encoding, it can be changed from command line with 'iconv -f ascii -t utf8 new.txt > new.txt'
    #the standard open function works fine with utf-8
    with open("new.txt", "r") as f:
        for line in f:
            row=line.split()
            for i in range(0,len(row)):
                f1.write(model(row[i]))
                f1.write(" ")
            f1.write("\n")
    """

#zero=[0]*400
#f1=open("vec.txt","w")
#with open("new.txt","r",encoding="utf-8") as f:
 #   for line in f:
  #      row=line.split()
   #     for i in range(0,len(row)):
    #        if row[i] in words.words():
     #           f1.write((model(row[i])))
     #           f1.write(" ")            
      #      else:
       #         f1.write(zero)
        #        f1.write(" ")
       # f1.write("\n")