import sys
sys.path.append('/datadrive/nlp/jasper/w2v/godin/word2vec_twitter_model/')
from word2vecReader import Word2Vec
from nltk.corpus import words
import io
import numpy
from spacy.en import English
import numpy as np
import argparse
import math
from tempfile import TemporaryFile



def text_to_embeddings(model_path, input_file, output_file,vector_dimension):
    """
    Takes an input_file and conby converts it to an output_file by replacing words with embedding vectors based on the model at model_path
    """
        #model_path = args.model_path_
    print("Loading the model, this can take some time...")
    model = Word2Vec.load_word2vec_format(model_path, binary=True)
    print("Loafing spaCy model, this can take some time...")
    nlp=English()
    #print(("The vocabulary size is: "+str(len(model.vocab))))
    #print("Vector for 'Shubham': " + str(model['Shubham']))
    #print("Embedding dimension: " + str(len(model['Shubham'])))
    #f1=open("embedding_vectors_400.txt","w")
    f1=open(output_file,'w')
    zero =  np.zeros((vector_dimension,), dtype=np.float)
    #Specify encoding with io.open (careful io.open is slow, don't use for large files)
    #latin-1 is usually the culprit if the files aren't utf-8 encoded
    #with io.open("dataset_latin-1.txt", "r", encoding='latin-1') as f:
    count=0
    max_length=0
    with io.open(input_file, "r", encoding='latin-1') as f:
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
            if rows==0:                 #ignore the tweet if out of vocabulary and take the control to the beginning of the loop
                count=count+1
                continue
            temp = arr[0]
            if (rows>max_length): # maximum words in a sentence
                max_length=rows
            for i in range(1,rows):
                temp=np.concatenate((temp,arr[i]),axis=0)
            rand=' '.join(map(str,temp))
            f1.write(rand)
            f1.write("\n")
    print("There are"+str(count)+"out of vocabulary sentences.")
    print(max_length)   
    return None
    
    
def text_to_embeddings_npy(model_path, input_file, output_file,vector_dimension):
    """
    Takes an input_file and conby converts it to a numpy output_file by replacing words with embedding vectors based on the model at model_path
    The output file contains a 2D array where the 0th column are the ids, the 1st column are the labels and the rest are the document vector values.
    """
    f1=io.open("dataset_latin_1_new.txt","w",encoding='latin-1')
    with io.open(input_file, "r", encoding='latin-1') as f:
        for line in f:
            row=line.split()
            for i in range(2,len(row)-1):
                f1.write(row[i])
                f1.write(" ")
            f1.write(row[len(row)-1])
            f1.write("\n")
    text_to_embeddings(model_path, "dataset_latin_1_new.txt","embedding_vectors_400_new.txt",vector_dimension)
    print("working_1")
    max_length_diff = 0
    count =0
    sum_total_words = 0
    sentence_count=0

    tweet_id = []
    label = []
    with io.open(input_file, "r", encoding='latin-1') as fobj:
        for line in fobj:
            row = line.split()
            tweet_id.append(row[0])
            label.append(row[1])
    tweet_ids = np.asarray(tweet_id)
    labels = np.asarray(label)
    tweet_id_s = np.array(tweet_ids,dtype=np.float)
    label_s = np.array(labels,dtype=np.int)
    print("working_2")

    flag=0
    vector = np.array([]) 
    max_length = 58
    zero =  np.zeros((400,), dtype=np.float)
    with open("embedding_vectors_400_new.txt","r") as f:
        for line in f:
            row = line.split()
            length = math.ceil(len(row)/400)
            sum_total_words += length 
            sentence_count += 1
            rows = np.asarray(row,dtype=np.float)
            if(length<max_length):
                count = max_length - length   # to check the difference between minimum length sentence and maximum length sentence
                if(max_length_diff<count):
                    max_length_diff=count
                for i in (range(max_length-length)):
                    rows=np.concatenate((rows,zero),axis=0)
            if flag==1:
                vector = np.vstack((vector,rows))
            if flag==0:
                vector = np.concatenate((vector,rows))
                flag=1
            #rand=' '.join(map(str,row))


    data = np.column_stack((tweet_id_s,label_s,vector))
    np.save(r"/datadrive/ML/shubham/data_wrangler/outfile_new.npy",data)

    print(max_length_diff)
    print((sum_total_words)/(sentence_count))

    return None
    
    
def load_embeddings_npy(input_file_path, padding_dimension):
    data = np.load(input_file_path)
    tweet_ids = data[:,0]

    y = np.array(data[:,1],dtype=int)
    x = data[:,2:]

    return x, y, tweet_ids
    
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('model_path_',help='Path of the word2vec twitter model bin file')
    parser.add_argument('input_file',help='File to read the input text data')
    parser.add_argument('output_file',help='numpy file to write ids,labels and vectors')
    parser.add_argument('vector_dimension',help='dimension of output vector using word2vec model',type=int)
    args = parser.parse_args()
    
    text_to_embeddings_npy(args.model_path_ , args.input_file, args.output_file,args.vector_dimension)

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