from Util import vn_api_path, extra_root
import sys

sys.path.append(vn_api_path)
import verbnet

from nltk.corpus import wordnet
from nltk.corpus.reader.wordnet import WordNetError

VN32 = extra_root + "verbnet3.2/"

def moh_anns(loc=extra_root + "corpora/moh/Metaphor-Emotion-Data-Files/Data-metaphoric-or-literal.txt"):
    res = {}
    with open(loc) as inp:
        inp = inp.readlines()[1:]
        for line in inp:
            data = line.split()
            synset = wordnet.synset(".".join(data[1].split("#")))
            res[synset] = data[-2]
    return res


def find_vn_senses(mohs, vn):
    res = {}
    
    for verb in vn.get_members():
        for wn_mapping in verb.wn:
            synset = None
            try:
                synset = wordnet.lemma_from_key(wn_mapping + "::").synset()
            except WordNetError as e:
                if wn_mapping[0] == "?":
                    synset = wordnet.lemma_from_key(wn_mapping[1:] + "::").synset()
                else:
                    print ("nope : ", wn_mapping)
            if synset and synset in mohs:
                if verb.name not in res:
                    res[verb.name] = {}
                if verb.class_id() not in res[verb.name]:
                    res[verb.name][verb.class_id()] = set()
                res[verb.name][verb.class_id()].add(wn_mapping + "-" + mohs[synset])
    for verb in res:
        if len(res[verb]) <= 1:
            continue
#        print ("---")
#        print (verb)
        all_lit_sense, all_met_sense = False, False
        for cl in res[verb]:
            met, lit = 0., 0.
            for sense in res[verb][cl]:
                if "literal" in sense:
                    lit += 1
                if "met" in sense:
                    met += 1
#            print ("---")
#            print ("Class:\t\t\t" + cl)
#            print ("% literal:\t\t" + str(lit/(met+lit)))
#            print ("Literal WN mappings:\t" + str(lit))
#            print ("Metaphoric WN mappings: " + str(met))

            if (lit / (met+lit)) == 1.:
                all_lit_sense = True
            if (lit / (met+lit)) == 0.:
                all_met_sense = True
        if all_lit_sense and all_met_sense:
            print ("--")
            print (verb)
            for cl in res[verb]:
                met, lit = 0., 0.
                for sense in res[verb][cl]:
                    if "literal" in sense:
                        lit += 1
                    if "met" in sense:
                        met += 1
                print (cl)
    return

if __name__ == "__main__":
    mohs = moh_anns()
    vn = verbnet.VerbNetParser(directory=VN32)
    find_vn_senses(mohs, vn)
