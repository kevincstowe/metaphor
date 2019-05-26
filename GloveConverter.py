from gensim.scripts.glove2word2vec import glove2word2vec
import os
from Util import model_location
 

for f in os.listdir(model_location):
    if "glove" in f and ".txt" in f and "w" not in f:
        print (f)
        glove2word2vec(model_location + f, model_location + f + ".w")
