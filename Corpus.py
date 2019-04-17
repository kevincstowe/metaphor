import string
import json
import jsonlines
import collections

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

import SDP

PUNCTS = r"^([" + string.punctuation + "_@\|\+#\?\*~\./\:<>$,\[\]\(\)&%‘’\"\';-]+)$"

class Corpus(object):
    def __init__(self):
        self.instances = None
        self.words = None
        self.lexicon = None

    def __str__(self):
        print("Instances : " + "None" if not self.instances else str(len(self.instances)))
        print("Words : " + "None" if not self.words else str(len(self.words)))

    def write_instances(self, output_loc):
        with open(output_loc, "w", encoding="utf-8") as o:
            for instance in self.instances:
                o.write(instance.id + ";;" + instance.text() + "\n")

    def write_allennlp(self, output_loc):
        if not output_loc.endswith(".jsonl"):
            output_loc += ".jsonl"
        with jsonlines.open(output_loc, "w") as o:
            for instance in self.instances:
                line_data = {"id":instance.id, "sentence":instance.text()}
                o.write(line_data)

    def get_words(self, filters):
        res = self.words
        for f in filters:
            res = [w for w in res if (getattr(w, f[0]) and getattr(w, f[0]) in f[1])]

        return res

    def build_lexicon(self, by_lemma=True):
        res = set()
        for word in self.words:
            if by_lemma:
                res.add(word.lemma)
            else:
                res.add(word.text)
        self.lexicon = list(res)

    def get_training_data(self):
        return self.words[:int(len(self.words)*.75)]

    def get_test_data(self):
        return self.words[int(len(self.words)*.75):]

    def parse_and_save(self, output):
        res = {}
        c = 1
        for sent in self.instances:
            try:
                parse = list(SDP.parse_sentence(sent.text()))[0].to_conll(style=10)
                res[sent.id] = parse
            except Exception as e:
                print (e, sent.text())
            if c % 100 == 0:
                print (str(c) + " done of " + str(len(self.instances)))
            c += 1
        json.dump(res, open(output + ".json", "w"))

    
class Sentence(object):
    def __init__(self):
        self.words = []
        pass

    
    def __str__(self):
        return self.text()

    def __iter__(self):
        for w in self.words:
            yield w

    def text(self):
        return " ".join([w.text for w in self.words])

    def list_of_word_strings(self):
        res = []
        for w in self.words:
            res.extend(w.text)
        return res

    def find_head(self, word):
        if not word.dep:
            return None

        else:
            if word.dep[2] == 0:
                return None
            else:
                try:
                    res = self.words[int(word.dep[2])-1]
                except IndexError as e:
                    print(word.dep)
                    print([(w.text, w.dep) for w in self.words])
                    return None

                return self.words[int(word.dep[2])-1]

    def find_dependencies(self, word, rec=False):
        if not rec:
            return [w for w in self.words if w.dep and int(w.dep[2]) == word.index+1]

class Word(object):
    def __init__(self, text="", met="N", pos=None, lemma=None, sentence=None, index=None):
        self.text = text
        self.met = met
        self.pos = pos
        self.lemma = lemma

        self.sentence = sentence
        self.index = index

        self.frame = None
        self.frame_element = []

        self.vnc = None

        self.construction = None

        self.allen_tags = {}

    def __str__(self):
        return self.text + " " + self.met

    def __eq__(self, other):
        return self.text == other.text and self.met == other.met and self.pos == other.pos and self.lemma == other.lemma and self.index == other.index

    def __hash__(self):
        return hash(self.text) + hash(self.met) + hash(self.pos) + hash(self.lemma)

    def __str__(self):
        return self.text

    # this is going to return all the words in a given context around the target word, with Nones for context outside
    # of the sentence
    def get_word_context(self, context):
        word_index = self.sentence.words.index(self)
        for i in range(-context, context+1):
            if word_index + i < 0 or word_index + i >= len(self.sentence.words):
                yield None
            else:
                yield self.sentence.words[word_index + i]


def align_words(words1, words2, remove_punct=False):
    res = []
    unused_candidates = list(range(len(words2)))

    for i in range(len(words1)):
        w1 = words1[i]

        if not w1.text.strip() or (remove_punct and not w1.text.strip(PUNCTS)):
            res.append((w1, []))
        else:
            cand = []
            # check if string matches
            for j in range(len(words2)):
                w2 = words2[j]
                if (w2 == w1.text or (remove_punct and w2 == w1.text.strip(PUNCTS))) and j in unused_candidates:
                    cand = [j]
                    break

            # try combining words, increasing the number combined until it finds a match or runs over sent length
            inc = 1
            while not cand and inc < len(words2):
                for j in [i for i in range(len(words2)-inc) if i in unused_candidates]:
                    word_cand = "".join(words2[j:j+inc+1])
                    if word_cand == w1.text or (remove_punct and word_cand == w1.text.strip(PUNCTS)):
                        cand = list(range(j,j+inc+1))
                        break
                inc += 1

            # add the result, remove it from unused elements
            res.append((w1, cand))
            for c in cand:
                unused_candidates.remove(c)

    return res


def add_dependencies(metaphors, deps_loc, lex_field=0):
    lemmatizer = WordNetLemmatizer()
    pos_map = {"n":wordnet.NOUN, "j":wordnet.ADJ, "v":wordnet.VERB, "r":wordnet.ADV}
    deps = json.load(open(deps_loc))

    for met in metaphors:
        try:
            id = met.source_file + "-" + met.id
        except:
            id = met.id
        
        if id in deps.keys():
            dep_list = [d.split("\t") for d in deps[id].split("\n")]
            alignments = align_words(met.words, [d[lex_field] for d in dep_list if d != ['']], remove_punct=True)

            for i in range(len(met.words)):
                met.words[i].dep = None

                for target in alignments[i][1]:
                    met.words[i].dep = dep_list[target]
                    if not met.words[i].pos:
                        met.words[i].pos = dep_list[target][1]
                    if not met.words[i].lemma:
                        try:
                            met.words[i].lemma = lemmatizer.lemmatize(met.words[i].text.strip(string.punctuation), pos_map[met.words[i].pos[0].lower()])
                        except KeyError as e:
                            met.words[i].lemma = met.words[i].text.lower().strip(string.punctuation)
                if not met.words[i].lemma:
                    met.words[i].lemma = met.words[i].text.lower()
                    
        else:
            #print ("id not found in dependency parses : " + met.id)
            for w in met.words:
                w.dep = None

def add_vn_parse(sentences, vn_parse_location):
    def list_duplicates(seq):
        res = []
        dups = collections.defaultdict(list)
        for i, e in enumerate(seq):
            dups[e].append(i)
        for k, v in sorted(dups.items()):
            if len(v) >= 2:
                res = v
        return res

    vn_tags = {line.split(";;")[0]:line.split(";;")[1:-1] for line in open(vn_parse_location).readlines()}
    for sent in sentences:
        try:
            id = sent.source_file + "-" + sent.id
        except:
            id = sent.id
        if id in vn_tags:
            for vn_tag in vn_tags[id]:
                for word in sent.words:
                    if word.text.strip(string.punctuation) == vn_tag.split()[0] or vn_tag.split()[0] in word.text.strip(string.punctuation).split("-"):
                        word.vnc = vn_tag.split()[1]
#        else:
#            print("no vn parse:" + id)


def populate_vn_from_heads(sentences):
    for sent in sentences:
        for w in sent.words:
            count = 0
            while w and w.vnc in ["None", None, "NONE"] and count < 5:
                h = sent.find_head(w)
                if h and h.vnc:
                    w.vnc = h.vnc
                w = h
                count += 1


def add_allen_parse(sentences, allen_parse_location):
    def allen_to_json(lines):
        allen_jsons = {}
        j = {}
        for l in lines:
            if l.strip():
                j.update(json.loads(" ".join(l.split(" ")[1:]).strip()))
            else:
                allen_jsons[j["id"]] = j
                j = {}

        return allen_jsons

    allen_jsons = allen_to_json(open(allen_parse_location).readlines())

    for s in sentences:
        word_aligns = align_words(s.words, allen_jsons[s.source_file + "-" + s.id]["words"])

        for v in allen_jsons[s.source_file + "-" + s.id]["verbs"]:
            for i in range(len(s.words)):
                s.words[i].allen_tags[v["verb"]] = []
                target_words = word_aligns[i][1]
                for j in target_words:
                    s.words[i].allen_tags[v["verb"]].append(v["tags"][j])

