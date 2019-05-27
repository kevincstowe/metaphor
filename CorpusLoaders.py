from lxml import etree
import json
import jsonpickle
import re
import codecs
import csv
import re
import pickle
import string
import random

from nltk.stem import WordNetLemmatizer

import Corpus
import Util
from Util import root, TAG_NOUNS, TAG_VERBS, TAG_ADJS, TAG_ADVS, FICTION, ACADEMIC, NEWS, CONVERSATION

root = "/home/kevin/GitHub/metaphor/"

LCC_DEPS = root + "deps/lcc_deps.json"
TROFI_DEPS = root + "deps/trofi_deps.json"
MOHX_DEPS = root + "deps/mohx.json"
VUAMC_DEPS = root + "deps/vuamc_deps.json"

LCC_LOCATION = root + "corpora/lcc_metaphor_dataset/en_small.xml"
TROFI_LOCATION = root + "corpora/trofi/TroFiExampleBase.txt"
MOHX_LOCATION = root + "corpora/MOH-X/MOH-X_formatted_svo_cleaned.csv"
VUAMC_CSV = root + "corpora/vuamc_corpus_all.csv"
VUAMC_DUMP = root + "corpora/vuamc_dump.p"

EXTRA_VN = "/home/kevin/GitHub/metaphor-in-context/data/vn_extra.csv"
EXTRA_SYN = "/home/kevin/GitHub/metaphor-in-context/data/syn_extra.csv"

LCC_VN = root + "vn/lcc.vn"
VUAMC_VN = root + "vn/vuamc.vn"

VUAMC_ALLEN = root + "allen/vuamc.allen"

TRAIN_TASK_LABELS = "/home/kevin/met-shared-task/all_pos_tokens.csv"
TEST_TASK_LABELS = "/home/kevin/met-shared-task/all_pos_tokens_test.csv"
VERB_TRAIN_TASK_LABELS = "/home/kevin/met-shared-task/verb_tokens.csv"
VERB_TEST_TASK_LABELS = "/home/kevin/met-shared-task/verb_tokens_test.csv"

'''
 LCC CORPUS
'''

class LCCCorpus(Corpus.Corpus):
    def __init__(self, threshhold=1):
        super().__init__()
        self.instances, self.words = [], []

        def merge_sents(sent1, sent2):
            for i in range(len(sent1.words)):
                if sent1.words[i].met == "N" and sent2.words[i].met != "N":
                    sent1.words[i].met = sent2.words[i].met
                if sent1.words[i].met != "N" and sent2.words[i].met != "N" and sent1.words[i].met != sent2.words[i].met:
                    if sent2.words[i].met not in sent1.words[i].met:
                        sent1.words[i].met += "-" + sent2.words[i].met

        lcc_data = etree.parse(LCC_LOCATION)
        instances = lcc_data.findall(".//LmInstance")
        metaphors = set()

        for instance in instances:                
            metaphor = LCCMetaphor(instance)

            if metaphor.met_score >= threshhold:
                metaphors.add(metaphor)

        Corpus.add_dependencies(metaphors,
                                LCC_DEPS, lex_field=0)
        Corpus.add_vn_parse(metaphors, LCC_VN)
        #Corpus.add_allen_parse(metaphors, "C:/Users/Kevin/PycharmProjects/metaphor/corpora/lcc_metaphor_dataset/lcc_allen.tagged")
        #Constructions.predict_constructions(metaphors)

        for met in metaphors:
            self.instances.append(met)
            self.words.extend(met.words)
        super().build_lexicon()

    def get_verb_cv_data(self):
        return [w for w in self.words if w.pos in TAG_VERBS]

    def get_cv_data(self):
        return [w for w in self.words if w.pos in TAG_NOUNS | TAG_ADJS | TAG_ADVS | TAG_VERBS]
        
    def get_verb_train_test_data(self):
        return [w for w in self.words[:int(len(self.words)*.75)] if w.pos in TAG_VERBS], [w for w in self.words[int(len(self.words)*.75):] if w.pos in TAG_VERBS]

    def get_train_test_data(self):
        return [w for w in self.words[:int(len(self.words)*.75)] if w.pos in TAG_NOUNS | TAG_ADJS | TAG_ADVS | TAG_VERBS], [w for w in self.words [int(len(self.words)*.75):] if w.pos in TAG_NOUNS | TAG_ADJS | TAG_ADVS | TAG_VERBS]

    def get_sent_train_test_data(self):
        return [instance for instance in self.instances[:int(len(self.instances)*.75)]], [instance for instance in self.instances[int(len(self.instances)*.75):]]
    
    def write_gao(self):
        for i in range(len(self.instances)):
            instance = self.instances[i]
            print ("lcc", str(i), instance.text(), str(["1" if ("source" in w.met or "target" in w.met) else "0" for w in instance.words]), str([w.pos for w in instance.words]), str(["M_" + w.text if ("source" in w.met or "target" in w.met) else w.text for w in instance.words]), "lcc")

            
class LCCMetaphor(Corpus.Sentence):
    def __init__(self, lcc_instance_node):
        super().__init__()
        self.target_cm = [lcc_instance_node.get('targetConcept')]
        annotations_element = lcc_instance_node.find(".//Annotations")

        met_anns = annotations_element.find(".//MetaphoricityAnnotations")
        self.met_score = sum([float(m.get('score')) for m in met_anns]) / len(met_anns)

        cm_source_anns = annotations_element.find(".//CMSourceAnnotations")
        self.source_cm = []
        if cm_source_anns is not None:
            self.source_cm = set([(cm.get("sourceConcept"), float(cm.get("score"))) for cm in cm_source_anns if
                                  float(cm.get('score')) >= 0])

        self.chain = lcc_instance_node.get('chain')
        self.id = lcc_instance_node.get('id')

        all_text = lcc_instance_node.find(".//TextContent")
        self.current_text = all_text.find(".//Current")
        self.prev_text = all_text.find(".//Prev")
        self.next_text = all_text.find(".//Next")

        self.source_lm = self.current_text.find(".//LmSource").text.strip()
        self.target_lm = self.current_text.find(".//LmTarget").text.strip()

        i = 0
        all_words = []
        for word_group in self.current_text.itertext():
            if word_group.strip() == self.source_lm:
                met = ["source", self.source_cm, self.met_score]
            elif word_group.strip() == self.target_lm:
                met = ["target", self.target_cm, self.met_score]
            else:
                met = ["N", "", ""]

            for w in [w for w in re.findall(r"[\w']+|[.,?!;:\"']", word_group) if w != "="]:
                self.words.append(Corpus.Word(text=w, met=met, index=i, sentence=self))
                i += 1
                
        
def load_lcc_file(filename="lcc_sents.json"):
    with open(filename) as f:
        sentences = json.load(f)

    return [jsonpickle.decode(sentences[k]) for k in sentences]


'''
 VUAMC CORPUS 
'''

class VUAMCCorpus(Corpus.Corpus):
    def __init__(self):
        super().__init__()
        self.instances, self.words = load_vuamc_csv()
        super().build_lexicon()
        self.add_task_labels()
        self.save()

    @staticmethod
    def load():
        return pickle.load(open(VUAMC_DUMP, "rb"))

    def save(self):
        pickle.dump(self, open(VUAMC_DUMP, "wb"))

    def add_task_labels(self):
        train_task_labels = {k.split(",")[0]:k.split(",")[1].strip() for k in open(TRAIN_TASK_LABELS).readlines()}
        test_task_labels = {k.strip():"test" for k in open(TEST_TASK_LABELS).readlines()}
        verb_train_task_labels = {k.split(",")[0]:k.split(",")[1].strip() for k in open(VERB_TRAIN_TASK_LABELS).readlines()}
        verb_test_task_labels = {k.strip():"test" for k in open(VERB_TEST_TASK_LABELS).readlines()}

        for word in self.words:
            w_key = word.sentence.source_file + "_" + word.sentence.id + "_" + str(word.index+1)
            if w_key in train_task_labels:
                if (word.met == "N" and train_task_labels[w_key] != "0") or (word.met != "N" and train_task_labels == "0"):
                    print ("ERROR: tag mismatch", word.text, word.sentence.text(), word.met, train_task_labels[w_key])
                word.met = word.met + "-train"
            elif w_key in test_task_labels:
                word.met = word.met + "-test"
            if w_key in verb_train_task_labels or w_key in verb_test_task_labels:
                word.met = word.met + "-verb"

    def get_cv_data(self):
        train, test = self.get_train_test_data()
        return train + test
    
    def get_train_test_data(self):
        return [w for w in self.words if w.met in ["N-train", "met-train", "N-train-verb", "met-train-verb"]], [w for w in self.words if w.met in ["N-test", "met-test", "N-test-verb", "met-test-verb"]]

    def get_verb_train_test_data(self):
        return [w for w in self.words if w.met in ["N-train-verb", "met-train-verb"]], [w for w in self.words if w.met in ["N-test-verb", "met-test-verb"]]

    def get_train_test_sents(self):
        return [s for s in self.instances if "train" in s.set], [s for s in self.instances if "test" in s.set]

    def convert_to_gao(self, instances=None):
        if not instances:
            instances = self.instances

        with open("C:/Users/Kevin/PycharmProjects/metaphor/corpora/vuamc/vuamc_gao.vn", "w") as output_file:
            for instance in instances:
                output_file.write(instance.source_file + ";;" + instance.id + ";;" + str([w.vnc for w in instance.words]) + ";;" + instance.text() + "\n")
        return {instance.source_file + "-" + instance.id:[str([w.vnc for w in instance.words]), instance.text()] for instance in instances}

    
def load_vuamc_csv(filename=VUAMC_CSV):
    with codecs.open(filename, encoding="latin-1", errors='replace') as f:
        data = [line for line in csv.reader(f)]

    sentences = []
    all_words = []

    for sent_index in range(1, len(data[1:])):
        line_data = data[sent_index]
        if not line_data:
            continue
        sentence = Corpus.Sentence()

        sentence.source_file = line_data[0]

        if sentence.source_file in ACADEMIC:
            sentence.domain = "academic"
        elif sentence.source_file in CONVERSATION:
            sentence.domain = "conversation"
        elif sentence.source_file in FICTION:
            sentence.domain = "fiction"
        elif sentence.source_file in NEWS:
            sentence.domain = "news"
        sentence.id = line_data[1]

        words = line_data[2]

        j = 0
        for i in range(0, len(words.split())):
            w_data = words.split()[i].split(";;")
            if "M_" in w_data[-1]:
                met = "met"
                word_text = w_data[-1][2:]
            else:
                met = "N"
                word_text = w_data[-1]

            pos = w_data[0]
            lemma = w_data[1]

            for extra_words in word_text.split("_"):
                if not set(extra_words).intersection(str(string.punctuation + string.ascii_letters + string.digits)):
                    sentence.words.append(Corpus.Word(text="none", met="none", pos="none", lemma="none", sentence=sentence, index = j))
                    j += 1
                    continue
                
                word = Corpus.Word(text=extra_words, met=met, pos=pos, lemma=lemma, sentence=sentence, index=j)

                sentence.words.append(word)
                all_words.append(word)
                j += 1
        sentences.append(sentence)

    Corpus.add_dependencies(sentences, VUAMC_DEPS)
    Corpus.add_vn_parse(sentences, VUAMC_VN)
    Corpus.add_allen_parse(sentences, VUAMC_ALLEN)
    #Corpus.populate_vn_from_heads(sentences)

    return sentences, all_words

'''
 TROFI CORPUS
'''

class TrofiCorpus(Corpus.Corpus):
    def __init__(self):
        super().__init__()
        self.instances, self.words = [], []
        lemmatizer = WordNetLemmatizer()
        cur_verb, cluster = "", ""

        for line in open(TROFI_LOCATION).readlines():
            if re.match(r"\*\*\*[a-z]", line):
                cur_verb = line.split("***")[1]
                continue
            elif "*" in line or not line.strip():
                if "literal" in line:
                    cluster = "literal"
                elif "nonliteral" in line:
                    cluster = "nonliteral"
                continue

            sentence = Corpus.Sentence()
            data = line.strip().split("\t")
            sentence.id = data[0]

            met = ""
            if "N" in data[1]:
                met = "met"
            if "L" in data[1]:
                met = "N"
            if "U" in data[1]:
                met = "?"

            for i in range(len(data[2].split())):
                word = data[2].split()[i]
                v_lem = lemmatizer.lemmatize(word, "v")
                cur_met = "N"
                if v_lem == cur_verb:
                    cur_met = "tag-" + met
                w = Corpus.Word(text=word, met=cur_met, sentence=sentence, index=i)
                sentence.words.append(w)
                self.words.append(w)

            self.instances.append(sentence)

        Corpus.add_dependencies(self.instances, TROFI_DEPS, lex_field=1)

    def get_cv_data(self):
        train, test = self.get_train_test_data()
        return train + test

    def get_train_test_data(self):
        res = []
        for w in self.words:
            if "tag" in w.met and "?" not in w.met:
                res.append(w)
        random.shuffle(res)
        return res[:int(len(res)*.75)], res[int(len(res)*.75):]

    
class MOHXCorpus(Corpus.Corpus):
    def __init__(self, loc=MOHX_LOCATION):
        self.instances, self.words = [], []

        c = 0
        for line in open(loc).readlines()[1:]:
            sentence = Corpus.Sentence()
            data = line.split(",")
            sentence.id = str(c)
            c += 1
            word_data = data[3].split()
            
            for i in range(len(word_data)):
                met = "N"
                if i == int(data[-2]):
                    met = "tag-" + data[-1].strip()
                w = Corpus.Word(text=word_data[i], met=met, sentence=sentence, index=i)
                sentence.words.append(w)
                self.words.append(w)
                
            self.instances.append(sentence)

        Corpus.add_dependencies(self.instances, MOHX_DEPS, lex_field=1)

    def get_cv_data(self):
        train, test = self.get_train_test_data()
        return train + test

    def get_train_test_data(self):
        res = []
        for w in self.words:
            if "tag" in w.met:
                res.append(w)
        random.shuffle(res)
        return res[:int(len(res)*.75)], res[int(len(res)*.75):]


class VnCorpus(Corpus.Corpus):
    def __init__(self, corpus_location):
        self.instances, self.words = [], []
        data = csv.reader(open(corpus_location))
        next(data)
        for line in data:
            sentence = Corpus.Sentence()
            sentence.id = line[1]

            index = int(line[-2])
            tag = int(line[-1])

            sent_data = line[2].split()
            for i in range(len(sent_data)):
                word = sent_data[i]
                met = "N"
                if i == index:
                    met = "met"
                w = Corpus.Word(text=word, sentence=sentence, met=met, index=i)
                sentence.words.append(w)
                self.words.append(w)
                
            self.instances.append(sentence)

#        Corpus.add_dependencies(self.instances, VN_DEPS, lex_field=1)

class SynCorpus(Corpus.Corpus):
    def __init__(self, corpus_location):
        self.instances, self.words = [], []
        data = csv.reader(open(corpus_location))
        next(data)
        for line in data:
            sentence = Corpus.Sentence()
            sentence.id = line[1]

            index = int(line[-2])
            tag = int(line[-1])

            sent_data = line[3].split()
            for i in range(len(sent_data)):
                word = sent_data[i]
                met = "N"
                if i == index:
                    met = "met"
                w = Corpus.Word(text=word, sentence=sentence, met=met, index=i)
                sentence.words.append(w)
                self.words.append(w)
                
            self.instances.append(sentence)

#        Corpus.add_dependencies(self.instances, VN_DEPS, lex_field=1)


def test():
    corpus = VnCorpus(EXTRA_VN)
    print (len(corpus.instances), len(corpus.words))
    corpus.parse_and_save("deps/vn_corpus.json")
    corpus = SynCorpus(EXTRA_SYN)
    print (len(corpus.instances), len(corpus.words))
    corpus.parse_and_save("deps/syn_corpus.json")

    
if __name__ == "__main__":
    test()
