import CorpusLoaders

def run_pos_filter(filter, word):
    return not filter or word.pos in filter


def analyze_sentences(corpus, syn_filter):
    met_counts = {}
    for sentence in corpus.instances:
        #print (sentence.text())
        for w in sentence.words:
            print (w, w.dep)
            if w.dep and w.dep[-1] in syn_filter:
                for w2 in sentence.find_dependencies(w, rec=True):
                    if w2.lemma not in met_counts:
                        met_counts[w2.lemma] = [0, 0, 0]
                    if w2.met == "N":
                        met_counts[w2.lemma][0] += 1
                    elif w2.met == "source":
                        met_counts[w2.lemma][1] += 1
                    elif w2.met == "target":
                        met_counts[w2.lemma][2] += 1

    met_keys = [k for k in met_counts.keys() if sum(met_counts[k]) >= 0]
    met_keys = sorted(met_keys, key=lambda x : float(sum(met_counts[x])), reverse=True)
    for word in met_keys[:100]:
        print(word + "," + ",".join([str(s) for s in met_counts[word]]))

def analyze_word_domains(corpus, filters=None, corpus_class=CorpusLoaders.LCCCorpus):
    words = corpus.get_words(filters=filters)
    for word in corpus.get_words(filters=filters):
        if corpus_class != CorpusLoaders.VUAMCCorpus:
            if word.met == "source":
                print(";;".join(["s", word.text, word.sentence.text(), str(word.sentence.source_cm), str(word.sentence.target_cm)]))
            elif word.met == "target":
                print(";;".join(["t", word.text, word.sentence.text(), str(word.sentence.source_cm), str(word.sentence.target_cm)]))
            else:
                print(";;".join(["n", word.text, word.sentence.text(), str(word.sentence.source_cm), str(word.sentence.target_cm)]))
        else:
            if word.met != "N":
                print(";;".join(["m", word.text, word.sentence.text()]))
            else:
                print(";;".join(["n", word.text, word.sentence.text()]))


def get_verb_arguments(words, n=0):
    keys = set()

    if n:
        words = words[:n]

    for verb in words:
        args = {}
        # get regular arguments
        sentence = verb.sentence
        for word in sentence.words:
            if verb.text in word.allen_tags and set(word.allen_tags[verb.text]) != {"O"}:
                for tag in [t[2:] for t in word.allen_tags[verb.text] if t != "O"]:
                    keys.add(tag)
                    if tag not in args:
                        args[tag] = [[], []]
                    if word.text not in args[tag][0]:
                        args[tag][0].append(word.text)
                        args[tag][1].append(str(word.index))

        print(verb.sentence.text())
        print("V;;" + verb.text)
        for a in sorted(args.keys()):
            if a != "V":
                print(a, ";;", " ".join(args[a][0]), ";;", " ".join(args[a][1]))
        print()


def count_lemmas(corpus, filters=()):
    lemmas = {}
    for word in corpus.get_words(filters=filters):
        if word.lemma == 'None':
            continue
        if word.lemma not in lemmas:
            lemmas[word.lemma] = 0
        lemmas[word.lemma] += 1

    lemmas = [(k, lemmas[k]) for k in lemmas]

    lemmas.sort(key=lambda x : x[1], reverse=True)
    for lemma in lemmas:
        print (lemma[0] + ";;" + str(lemma[1]))


def analyze_lemma_metaphors(corpus, lemmas):
    res = {}
    for lemma in lemmas:
        res[lemma] = [[],[]]
        for w in corpus.get_training_data():
            if w.pos in CorpusLoaders.VUAMC_VERBS and w.lemma == lemma:
                if "met" in w.met:
                    res[lemma][1].append((str(w.vnc), w.sentence.text()))
                else:
                    res[lemma][0].append((str(w.vnc), w.sentence.text()))

    for key in res:
        print (key)
        print ("LIT")
        for sent in res[key][0]:
            print (sent[0] + ";;" + sent[1])
        print ("-")
        print ("MET")
        for sent in res[key][1]:
            print (sent[0] + ";;" + sent[1])
        print ("--")

def analyze_verb_met_percent(corpus, by_class=False):
    res = {}

    verbs = [{}, {}]
    for w in corpus.get_training_data():
        if w.pos in CorpusLoaders.VUAMC_VERBS:
            if by_class:
                item = w.vnc
            else:
                item = w.lemma

            if item not in res:
                res[item] = [0,0]
            if "met" in w.met:
                if w.vnc not in verbs[1]:
                    verbs[1][w.vnc] = {}
                if w.lemma not in verbs[1][w.vnc]:
                    verbs[1][w.vnc][w.lemma] = 0
                verbs[1][w.vnc][w.lemma] += 1
                res[item][1] += 1
            else:
                if w.vnc not in verbs[0]:
                    verbs[0][w.vnc] = {}
                if w.lemma not in verbs[0][w.vnc]:
                    verbs[0][w.vnc][w.lemma] = 0
                verbs[0][w.vnc][w.lemma] += 1
                res[item][0] += 1


    for w in sorted(res, key=lambda x: max(res[x][0] / sum(res[x]), res[x][1] / sum(res[x])), reverse=True):
        if sum(res[w]) >= 10:
            lit_verbs, met_verbs = [], []
            if w in verbs[0]:
                lit_verbs = {k:verbs[0][w][k] for k in sorted(verbs[0][w], key=verbs[0][w].get, reverse=True)}
            if w in verbs[1]:
                met_verbs = {k:verbs[1][w][k] for k in sorted(verbs[1][w], key=verbs[1][w].get, reverse=True)}
            print(str(w) + ";;" + str(res[w][0]) + ";;" + str(res[w][1]) + ";;" + str(max(res[w][0] / sum(res[w]), res[w][1] / sum(res[w]))) + ";;" + str(lit_verbs) + ";;" + str(met_verbs))

if __name__ == "__main__":
    corpus = CorpusLoaders.VUAMCCorpus()
#    corpus = CorpusLoaders.LCCCorpus()
#    analyze_verb_met_percent(corpus, False)
    analyze_lemma_metaphors(corpus, ["allow"])
