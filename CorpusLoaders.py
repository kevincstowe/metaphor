from lxml import etree
import json
import jsonpickle
import re
import codecs
import csv
import re
import pickle
import string

from nltk.stem import WordNetLemmatizer

import Corpus
import ConvertToGao


LCC_DEPS = "/home/kevin/metaphor/corpora/lcc_metaphor_dataset/lcc_deps.json"
TROFI_DEPS = "trofi_deps.json"
MOHX_DEPS = "mohx.json"

LCC_LOCATION = "/home/kevin/metaphor/corpora/lcc_metaphor_dataset/en_small.xml"

TAG_NOUNS = {"NNP", "NNS", "NN", "NNPS"}
TAG_ADJS = {"JJ", "JJR", "JJS"}
TAG_ADVS = {"RB", "RBR", "RBS"}
TAG_VERBS = {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ"}
TAG_PREPS = ["IN"]
TAG_ALL = {'LS', 'VBZ', 'WP', 'WRB', 'RB', 'FW', 'RBS', 'PRP$', 'JJR', 'RBR', 'VBG', 'NNPS', 'DT', 'NNS', 'EX', 'IN', 'UH',
       'CC', 'PDT', 'NN', 'TO', 'VB', 'PRP', 'RP', 'JJS', 'JJ', 'VBD', 'MD', 'VBP', 'CD', 'POS', 'WDT', 'VBN', 'WP$',
       'SYM', 'NNP'}

VUAMC_CSV = "/home/kevin/metaphor/corpora/vuamc_corpus_all.csv"
VUAMC_DUMP = "/home/kevin/metaphor/corpora/vuamc_dump.p"

VUAMC_NOUNS = {"NN0", "NN1", "NN2", "NP0", "PNI", "PNX"}
VUAMC_ADJS = {"AJ0", "AJC", "AJS"}
VUAMC_ADVS = {"AV0", "AVP", "AVQ"}
VUAMC_VERBS = {"VBB", "VBD", "VBG", "VBI", "VBN", "VBZ", "VDB", "VDD", "VDG", "VDI", "VDN", "VDZ", "VHB", "VHD", "VHG", "VHI", "VHZ", "VM0", "VVB", "VVD", "VVG", "VVI", "VVN", "VVZ"}
VUAMC_PREPS = {"PRF", "PRP"}
VUAMC_ALL = ['PNP', 'VHD', 'PUL', 'CRD', 'NN1-VVG', 'NP0', 'None', 'CJS', 'CJC', 'VDB', 'PNI-CRD', 'DPS', 'AJ0-VVD', 'PRF', 'VVG', 'NN1-AJ0', 'NN2', 'AJC', 'AJ0', 'VDI', 'VDN', 'VHB', 'AJ0-VVN', 'VVD-AJ0', 'AVQ', 'CRD-PNI', 'PUN', 'VDG', 'CJT-DT0', 'VVN-VVD', 'VVD-VVN', 'NN1-VVB', 'UNC', 'VVN', 'DT0-CJT', 'VBG', 'VVZ', 'VHG', 'PUR', 'VBI', 'VVD', 'AJ0-VVG', 'TO0', 'VVI', 'CJT', 'XX0', 'VBB', 'POS', 'NP0-NN1', 'AV0', 'VHZ', 'VVG-AJ0', 'ITJ', 'VHI', 'PRP-AVP', 'PRP-CJS', 'PNQ', 'CJS-AVQ', 'NN2-VVZ', 'NN0', 'AJ0-NN1', 'VVB', 'AV0-AJ0', 'NN1-NP0', 'VHN', 'ORD', 'VVN-AJ0', 'AT0', 'VBZ', 'VDD', 'AJ0-AV0', 'DTQ', 'sentence', 'AJS', 'PNI', 'VVG-NN1', 'VVZ-NN2', 'PUQ', 'DT0', 'AVP-PRP', 'VVB-NN1', 'PRP', 'CJS-PRP', 'VBN', 'VBD', 'VDZ', 'NN1', 'AVP', 'EX0', 'AVQ-CJS', 'ZZ0', 'PNX', 'VM0']

ACADEMIC = {'b17-fragment02', 'clw-fragment01', 'ecv-fragment05', 'acj-fragment01', 'a6u-fragment02', 'crs-fragment01', 'b1g-fragment02', 'ew1-fragment01', 'fef-fragment03', 'clp-fragment01', 'amm-fragment02', 'alp-fragment01', 'as6-fragment01', 'as6-fragment02', 'ea7-fragment03'}
CONVERSATION = {'kbw-fragment09', 'kb7-fragment31', 'kb7-fragment48', 'kbd-fragment21', 'kb7-fragment45','kbh-fragment04', 'kcv-fragment42', 'kbp-fragment09', 'kcf-fragment14', 'kcu-fragment02', 'kbw-fragment42', 'kbd-fragment07', 'kbh-fragment01', 'kb7-fragment10', 'kcc-fragment02', 'kbh-fragment02', 'kbh-fragment09', 'kbh-fragment03', 'kbw-fragment04', 'kbw-fragment17', 'kbj-fragment17', 'kbw-fragment11', 'kbh-fragment41', 'kbc-fragment13'}
FICTION = {'faj-fragment17', 'ccw-fragment04', 'cty-fragment03', 'bmw-fragment09', 'fpb-fragment01', 'c8t-fragment01', 'cdb-fragment04', 'fet-fragment01', 'ab9-fragment03', 'ac2-fragment06', 'bpa-fragment14', 'g0l-fragment01', 'cb5-fragment02', 'ccw-fragment03', 'cdb-fragment02'}
NEWS = {'al2-fragment16', 'ahd-fragment06', 'a3e-fragment02', 'ahc-fragment61', 'a36-fragment07', 'a3m-fragment02', 'a5e-fragment06', 'a1j-fragment33', 'ahe-fragment03', 'aa3-fragment08', 'a7w-fragment01', 'a31-fragment03', 'a1u-fragment04', 'a7t-fragment01', 'a80-fragment15', 'a1n-fragment09', 'a1e-fragment01', 'a3p-fragment09', 'a1m-fragment01', 'a1l-fragment01', 'a8m-fragment02', 'a8n-fragment19', 'a98-fragment03', 'a1g-fragment27', 'ahb-fragment51', 'ahf-fragment24', 'a1f-fragment09', 'a3e-fragment03', 'a1f-fragment12', 'ahc-fragment60', 'a1h-fragment05', 'a8r-fragment02', 'a1f-fragment08', 'a3c-fragment05', 'a1p-fragment03', 'al5-fragment03', 'a9j-fragment01', 'al0-fragment06', 'a2d-fragment05', 'a7y-fragment03', 'a1n-fragment18', 'a1f-fragment10', 'a1f-fragment06', 'a1x-fragment05', 'a4d-fragment02', 'a7s-fragment03', 'ajf-fragment07', 'a1h-fragment06', 'al2-fragment23', 'a1p-fragment01', 'a1g-fragment26', 'a1f-fragment11', 'a3k-fragment11', 'a1k-fragment02', 'a39-fragment01', 'ahf-fragment63', 'a1j-fragment34', 'a1x-fragment04', 'a1x-fragment03', 'a38-fragment01', 'ahl-fragment02', 'a8u-fragment14', 'a1f-fragment07'}

TRAIN_TASK_LABELS = "/home/kevin/met-shared-task/all_pos_tokens.csv"
TEST_TASK_LABELS = "/home/kevin/met-shared-task/all_pos_tokens_test.csv"
VERB_TRAIN_TASK_LABELS = "/home/kevin/met-shared-task/verb_tokens.csv"
VERB_TEST_TASK_LABELS = "/home/kevin/met-shared-task/verb_tokens_test.csv"

#TROFI_DEPS = "C:/Users/Kevin/PycharmProjects/metaphor/corpora/lcc_metaphor_dataset/lcc_deps.json"
TROFI_LOCATION = "/home/kevin/metaphor/corpora/trofi/TroFiExampleBase.txt"
MOHX_LOCATION = "/home/kevin/GitHub/metaphor-in-context/data/MOH-X/MOH-X_formatted_svo_cleaned.csv"

DOMAINS = ['MARRIAGE', 'VEHICLE', 'DEMOGRAPHICS', 'THEFT', 'HUMAN_BODY', 'LIGHT', 'BACKWARD_MOVEMENT', 'BARRIER', 'DESIRE', 'UPWARD_MOVEMENT', 'RESOURCE', 'STORY', 'VISION', 'BATTLE', 'A_GOD', 'ADDICTION', 'DESTROYER', 'ELECTIONS', 'MAZE', 'TEMPERATURE', 'GUN_OWNERSHIP', 'CONTROL_OF_GUNS', 'PATHWAY', 'MIGRATION', 'SHAPE', 'HAZARDOUS_GEOGRAPHIC_FEATURE', 'POSITION_AND_CHANGE_OF_POSITION_ON_A_SCALE', 'DRUG_TRAFFICKING', 'BLOOD_STREAM', 'TERRORISM', 'GAP', 'ABYSS', 'PARASITE', 'MOVEMENT', 'TAXPAYERS', 'ACCIDENT', 'GREED', 'PHYSICAL_LOCATION', 'ABORTION', 'POLITICIANS', 'WEALTH', 'PLANT', 'BUILDING', 'FORWARD_MOVEMENT', 'LOW_POINT', 'FAMILY', 'HIGH_POINT', 'OBESITY', 'SCHISM', 'OBJECT_HANDLING', 'MEDICINE', 'SIZE', 'ENSLAVEMENT', 'PORTAL', 'CONTAINER', 'DEMOCRACY', 'JOURNEY', 'NATURAL_PHYSICAL_FORCE', 'SCIENCE', 'WAR', 'FOOD', 'STRUGGLE', 'GAME', 'WEAKNESS', 'GUN_DEBATE_GROUPS', 'GUN_VIOLENCE', 'FACTORY', 'TAXATION', 'FIRE', 'A_RIGHT', 'ISLAMIC', 'MACHINE', 'RELIGION', 'FORCEFUL_EXTRACTION', 'INSANITY', 'GIFT', 'ENERGY', 'CLIMATE_CHANGE', 'SERVANT', 'INTELLECTUAL_PROPERTY', 'RULE_ENFORCER', 'PROTECTION', 'EMOTION_EXPERIENCER', 'POVERTY', 'STAGE', 'PHYSICAL_BURDEN', 'MORAL_DUTY', 'CROP', 'MENTAL_CONCEPTS', 'TAXES', 'TOOL', 'DEBT', 'BODY_OF_WATER', 'COMPETITION', 'FABRIC', 'PHYSICAL_OBJECT', 'AVERSION', 'GEOGRAPHIC_FEATURE', 'STRENGTH', 'CONFINEMENT', 'DARKNESS', 'OTHER', 'BUREAUCRACY', 'WEATHER', 'FURNISHINGS', 'LOW_LOCATION', 'GOAL_DIRECTED', 'LEADER', 'INDUSTRY', 'LIFE_STAGE', 'MOVEMENT_ON_A_VERTICAL_SCALE', 'IMPURITY', 'HIGH_LOCATION', 'GUNS', 'CRIME', 'BUSINESS', 'WELFARE', 'DOWNWARD_MOVEMENT', 'ANIMAL', 'GOVERNMENT', 'CLOTHING', 'MAGIC', 'EMPLOYEE', 'DISEASE', 'BLOOD_SYSTEM', 'MONEY', 'PHYSICAL_HARM', 'MONSTER', 'PLIABILITY', 'GUN_RIGHTS', 'CONTAMINATION']


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
                                "/home/kevin/metaphor/corpora/lcc_metaphor_dataset/lcc_deps.json", lex_field=0)
        Corpus.add_vn_parse(metaphors, "/home/kevin/metaphor/corpora/lcc_metaphor_dataset/lcc_vn")
        #        Corpus.add_allen_parse(metaphors, "C:/Users/Kevin/PycharmProjects/metaphor/corpora/lcc_metaphor_dataset/lcc_allen.tagged")
        #        Constructions.predict_constructions(metaphors)

        for met in metaphors:
            self.instances.append(met)
            self.words.extend(met.words)
        super().build_lexicon()

    def get_verb_training_data(self):
        return [w for w in self.words[:int(len(self.words)*.75)] if w.pos in TAG_VERBS]

    def get_verb_test_data(self):
        return [w for w in self.words[int(len(self.words)*.75):] if w.pos in TAG_VERBS]

    def get_training_data(self):
        return [w for w in self.words[:int(len(self.words)*.75)] if w.pos in TAG_NOUNS | TAG_ADJS | TAG_ADVS | TAG_VERBS]

    def get_test_data(self):
        return [w for w in self.words[int(len(self.words)*.75):] if w.pos in TAG_NOUNS | TAG_ADJS | TAG_ADVS | TAG_VERBS]

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

    def get_training_data(self):
        return [w for w in self.words if w.met in ["N-train", "met-train", "N-train-verb", "met-train-verb"]]

    
    def get_test_data(self):
        return [w for w in self.words if w.met in ["N-test", "met-test", "N-test-verb", "met-test-verb"]]

    def get_verb_training_data(self):
        return [w for w in self.words if w.met in ["N-train-verb", "met-train-verb"]]

    def get_verb_test_data(self):
        return [w for w in self.words if w.met in ["N-test-verb", "met-test-verb"]]

    def get_training_sents(self):
        return [s for s in self.instances if "train" in s.set]

    def get_test_sents(self):
        return [s for s in self.instances if "test" in s.set]

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

    Corpus.add_dependencies(sentences, "/home/kevin/metaphor/corpora/vuamc/vuamc_deps.json")
    Corpus.add_vn_parse(sentences, "/home/kevin/metaphor/corpora/vuamc/vuamc_vn")
    Corpus.add_allen_parse(sentences, "/home/kevin/metaphor/corpora/vuamc/vuamc_allen.tagged")
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


    def get_training_data(self):
        res = []
        for w in self.words:
            if "tag-" in w.met and "?" not in w.met:
                res.append(w)
        return res

    def get_verb_training_data(self):
        return self.get_training_data()

    
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

    def get_training_data(self):
        res = []
        for w in self.words:
#            print (w, w.lemma)
            if "tag" in w.met:
                res.append(w)
        return res
            
def test():
    c = VUAMCCorpus()
    c.parse_and_save("vuamc.json")

if __name__ == "__main__":
    test()
