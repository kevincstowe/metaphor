from gensim.scripts.glove2word2vec import glove2word2vec
import os

for f in os.listdir("/data/kevin/Vectors/glove_models/"):
    if "glove" in f and ".txt" in f and "w" not in f:
        print (f)
        glove2word2vec("/data/kevin/Vectors/glove_models/" + f, "/data/kevin/Vectors/glove_models/" + f + ".w")
