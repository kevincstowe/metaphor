from CorpusLoaders import LCCCorpus
import CorpusLoaders
from csv import writer
import string

POS_DICT = {"NNS":"NOUN", "NN":"NOUN", "CD":"NOUN", "FW":"NOUN", "LS":"NOUN", "NN0":"NOUN", "NN1":"NOUN", "NN2":"NOUN", "NP0":"NOUN", "CRD":"NOUN",
            "NNP":"PROPN", "NNPS":"PROPN",
            "VB":"VERB", "VBD":"VERB", "VBG":"VERB", "VBP":"VERB", "VBZ":"VERB", "VBN":"VERB", "MD":"VERB",
            "RB":"ADV", "RBR":"ADV", "RBS":"ADV", "EX":"ADV", "WRB":"ADV", "AV0":"ADV", "AVP":"ADV", "AVQ":"ADV",
            "IN":"ADP",
            "TO":"PART", "POS":"PART", "RP":"PART",
            "JJ":"ADJ", "JJR":"ADJ", "JJS":"ADJ", "AJ0":"ADJ", "AJC":"ADJ", "AJS":"ADJ",
            "CC":"CCONJ", "CJC":"CCONJ",
            "PUN":"PUNCT", ",":"PUNCT", ".":"PUNCT", "``":"PUNCT", ":":"PUNCT", "''":"PUNCT", "-RRB-":"PUNCT", "-LRB-":"PUNCT", "SYM":"PUNCT", "$":"PUNCT", "#":"PUNCT",
            "DT":"DET", "WDT":"DET", "PDT":"DET", "AT0":"DET",
            "PRP":"PRON", "PRP$":"PRON", "WP":"PRON", "WP$":"PRON", "PNI":"PRON", "PNX":"PRON", "DPS":"PRON", "DT0":"PRON", "DTQ":"PRON", "PNQ":"PRON",
            "UH":"INTJ",
            "None":"PUNCT", None:"PUNCT"
            }


def get_domains(data):
    domains = set()
    for instance in data.instances:
        for source_cm in instance.source_cm:
            domains.add(source_cm[0])
        for target_cm in instance.target_cm:
            domains.add(target_cm)
    return ["NONE"] + sorted(list(domains))


def write_output(data, sents, file_root):
    if sents:
        with open(file_root + "_sents.txt", "w", encoding="utf-8") as output_file:
            for line in sents:
                output_file.write(line + "\n")


    with open(file_root + "_cls.csv", "w", encoding="utf-8") as output_file:
        output_writer = writer(output_file)
        for line in data:
            output_writer.writerow(line)
    

def write_gao_seq(data, multi=True):
    domains = get_domains(data)
    output = []
    c = 0
    for sentence in data.instances:
        tags = []
        met_words = []
        for word in sentence.words:
            if "source" in word.met:
                tags.append(list(set([domains.index(s[0]) for s in word.met[1] if s[1] > 0])))
            elif "target" in word.met:
                tags.append([domains.index(t) for t in word.met[1]])
            else:
                tags.append([domains.index("NONE")])
        output.append(["lcc_seq_all", c, " ".join([w.text.strip(string.punctuation) for w in sentence.words]), tags, [POS_DICT[w.pos] for w in sentence.words], " ".join(["M_" + w.text.strip(string.punctuation) if "source" in w.met or "target" in w.met else w.text for w in sentence.words]), "lcc"])

        c += 1
    write_output(output, None, "lcc_seq_all")


def write_gao_source(data, multi=True):
    domains = get_domains(data)
    output, sents = [], []
    c = 0
    for instance in data.instances:
        for word in instance:
            if word.pos and word.pos[0] == "V" and word.lemma not in ["be", "do", "have"]:
                if "source" in word.met:
                    if not multi:
                        tags = [1]
                    else:
                        if not word.met[1]:
                            tags = [domains.index("OTHER")]
                        else:
                            tags = [domains.index(s[0]) for s in word.met[1] if s[1] > 0]
                            if not tags:
                                tags = [domains.index("OTHER")]
                else:
                    tags = [0]
                output.append(["lcc", c, word.lemma, " ".join([w.text.strip(string.punctuation) for w in instance.words]), str(word.index), str(tags)])
        sents.append(" ".join([w.text.strip(string.punctuation) for w in instance.words]))
        c += 1        
    write_output(output, sents, "lcc_source")

if __name__ == "__main__":
    corpus = LCCCorpus()
    write_gao_seq(corpus)
    write_gao_source(corpus)
