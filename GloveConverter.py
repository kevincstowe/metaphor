from gensim.scripts.glove2word2vec import glove2word2vec
import os

for f in os.listdir("models"):
    if "glove" in f and "w" not in f:
        print (f)
        glove2word2vec("models/" + f, "models/" + f + ".w")