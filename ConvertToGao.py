import os
import re
import csv
import string

from nltk.stem import WordNetLemmatizer

l = WordNetLemmatizer()

import CorpusLoaders
import SDP
import Util

VUA_TR_SEQ = "/home/kevin/GitHub/metaphor-in-context/data/VUAsequence/VUA_seq_formatted_train.csv.vn"
VUA_TE_SEQ = "/home/kevin/GitHub/metaphor-in-context/data/VUAsequence/VUA_seq_formatted_test.csv.vn"
VUA_VA_SEQ = "/home/kevin/GitHub/metaphor-in-context/data/VUAsequence/VUA_seq_formatted_val.csv.vn"


VN_RE = r"([a-zA-Z]+_[1-9][0-9]?[0-9]?([.-]?[0-9]+)+)"

METS = ["31.1", "29.4", "45.6", "26.6.2", "32.2", "95.1", "42.4", "29.5", "111.1", "54.6", "55.5", "27.1", "34.2", "13.5.4","66", "104", "105.3", "105.1", "46", "67", "47.9", "42.3", "45.6.2", "48.1.2", "55.7"]
LITS = ["37.9", "26.1", "40.3.2", "57", "40.3.2", "13.5.1", "51.1", "15.1", "49.2", "15.4", "54.3", "15.3", "11.3", "58.1", "68", "36.3", "36.7", "114.2", "55.4", "37.7", "47.8", "37.6"]
VUAMC_NOUNS = {"NN0", "NN1", "NN2", "NP0", "PNI", "PNX"}
VUAMC_ADJS = {"AJ0", "AJC", "AJS"}
VUAMC_ADVS = {"AV0", "AVP", "AVQ"}
VUAMC_VERBS = {"VBB", "VBD", "VBG", "VBI", "VBN", "VBZ", "VDB", "VDD", "VDG", "VDI", "VDN", "VDZ", "VHB", "VHD", "VHG", "VHI", "VHZ", "VM0", "VVB", "VVD", "VVG", "VVI", "VVN", "VVZ"}
VUAMC_PREPS = {"PRF", "PRP"}
VUAMC_ALL = ['PNP', 'VHD', 'PUL', 'CRD', 'NN1-VVG', 'NP0', 'None', 'CJS', 'CJC', 'VDB', 'PNI-CRD', 'DPS', 'AJ0-VVD', 'PRF', 'VVG', 'NN1-AJ0', 'NN2', 'AJC', 'AJ0', 'VDI', 'VDN', 'VHB', 'AJ0-VVN', 'VVD-AJ0', 'AVQ', 'CRD-PNI', 'PUN', 'VDG', 'CJT-DT0', 'VVN-VVD', 'VVD-VVN', 'NN1-VVB', 'UNC', 'VVN', 'DT0-CJT', 'VBG', 'VVZ', 'VHG', 'PUR', 'VBI', 'VVD', 'AJ0-VVG', 'TO0', 'VVI', 'CJT', 'XX0', 'VBB', 'POS', 'NP0-NN1', 'AV0', 'VHZ', 'VVG-AJ0', 'ITJ', 'VHI', 'PRP-AVP', 'PRP-CJS', 'PNQ', 'CJS-AVQ', 'NN2-VVZ', 'NN0', 'AJ0-NN1', 'VVB', 'AV0-AJ0', 'NN1-NP0', 'VHN', 'ORD', 'VVN-AJ0', 'AT0', 'VBZ', 'VDD', 'AJ0-AV0', 'DTQ', 'sentence', 'AJS', 'PNI', 'VVG-NN1', 'VVZ-NN2', 'PUQ', 'DT0', 'AVP-PRP', 'VVB-NN1', 'PRP', 'CJS-PRP', 'VBN', 'VBD', 'VDZ', 'NN1', 'AVP', 'EX0', 'AVQ-CJS', 'ZZ0', 'PNX', 'VM0']

POS_DICT = {"NNS":"NOUN", "NN":"NOUN", "CD":"NOUN", "FW":"NOUN", "LS":"NOUN", "NN0":"NOUN", "NN1":"NOUN", "NN2":"NOUN", "NP0":"NOUN", "CRD":"xNOUN",
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
            None:"NOUN", "None":"NOUN"
            }

PATTERN_DICT ={"encourage-to":0, "encourage-dobj_~to":1,
 "blow-over":0, "blow-up":1, "blow-away":1,
 "conduct-pass":0, "conduct-~pass_dobj":1,
 "find-nsubj_pro-vp":0, "find-out":0, "find-dead":0, "find-nsubj_dobj_to-be-adj":1,
 "fall-subj_adv":0, "fall-nsubj_~dobj":0, "fall-wh":1, "fall-in":1, "fall-to":1,
 "hold-nsubj_dobj_by-np":0, "hold-nsubj_dobj_out":0, "hold-onto":0, "hold-on":0, "hold-out":0, "hold-dobj_that-comp":1, "hold-dobj_adj":1, "hold-at":1, "hold-down":1,
 "bring-up":0, "bring-about":0, "bring-dobj_to-np":1, "bring-together":1, "bring-in":1,
 "allow-dobj_to_be":0, "allow-which":1,
 "spend-nsubj_dobj_on":0, "spend-nsubj_~dobj":0, "spend-time":1, "spend-life":1,
 "play-against_dobj":0, "play-down_dobj":0, "play-on_dobj":0, "play-with":0,
 "meet-nsubj_~dobj":0, "meet-for":0, "meet-at":0, "meet-to":0,
 "suggest-neg":0,
 "allow-neg":0}

for v in VUAMC_VERBS:
    POS_DICT[v] = "VERB"

def convert_met(word):
    res = 0
    if word.met[1] == "":
        pass
    elif word.met[1] == []:
        res = 1
    elif type(word.met[1]) == list:
        if word.met[1][0] in Util.DOMAINS:
            res = Util.DOMAINS.index(word.met[1][0])
        else:
            print ("target domain didn't match: " + word.met[1][0])
    else:
        for source_d in word.met[1]:
            if source_d[1] > 2:
                if source_d[0] in Util.DOMAINS:
                    res = Util.DOMAINS.index(source_d[0])
    return res


    
def write_output(data, filename):
    with open("C:/Users/Kevin/PycharmProjects/metaphor/corpora/additional_met/gao/" + filename, "w", newline="", encoding="utf-8") as gao_output:
        writer = csv.writer(gao_output)
        writer.writerow(["filler"])
        for line in data:
            writer.writerow(line)


def gao_line_data(line_data):
    verb, index, tag = None, 0, 0
    for i in range(len(line_data)):
        word = line_data[i]
        if "_None" in word:
            return None
        if re.match(VN_RE, word):
            verb, vnc = word.split("_")
            vnc = vnc.split("-")[0]
            index = i
            verb = l.lemmatize(verb.lower(), "v")

            if vnc in METS:
                tag = 1
            elif vnc in LITS:
                tag = 0
            else:
                return None

    return verb, index, tag


def clean_line(line):
    data = [w for w in line.split() if "*" not in w and (w[0] != "[" and w[-1] != "]") and (w[0] != "<" and w[-1] != ">")]
    return data


def convert_all_syn(directory="C:/Users/Kevin/PycharmProjects/metaphor/corpora/additional_met/syntax/"):
    output = {k:[] for k in PATTERN_DICT.keys()}
#    output = {k:[] for k in ["encourage-dobj_~to"]}
    verb_counts = {}
    for f in os.listdir(directory):
        verb = f.split("-")[0]
        if verb not in verb_counts:
            verb_counts[verb] = 0
        c = 0
        for line in open(directory + f):
            verb_counts[verb] += 1
            c += 1
            if f in output:
                output[f].append(line.split(";;"))
            if c >= 100:
                break
    for v in verb_counts:
        print (v, verb_counts[v])

    elmo_out = []
    for k in sorted(output.keys()):
        for l in output[k]:
            elmo_out.append(" ".join(clean_line(l[0])) + "\n")

    with open("C:/Users/Kevin/PycharmProjects/metaphor/corpora/additional_met/elmo_input/syn_extra", "w", encoding="latin-1") as output_file:
        output_file.writelines(elmo_out)

    print ("elmo done, doing seqs...")
    gao_seq = convert_syn_to_seq(output)
    write_output(gao_seq, "VUA_seq_formatted_syn_extra.csv")

    print ("seqs done, doing cls")
    gao_cls = convert_syn_to_cls(output)
    write_output(gao_cls, "VUA_syn_extra.csv")


    return


def convert_all_vn(directory):
    data = set()
    for f in os.listdir(directory):
        data |= set(open(directory + f, encoding="utf-8").readlines())
    data = sorted(list(data))

    elmo_lines, vnc_lines = convert_to_elmo(data)

    with open("C:/Users/Kevin/PycharmProjects/metaphor/corpora/additional_met/elmo_input/vn_extra", "w", encoding="utf-8") as output_file:
        output_file.writelines(elmo_lines)

    convert_vn_to_cls(vnc_lines)
    convert_vn_to_seq(vnc_lines)


def convert_to_elmo(data):
    d = {}
    extra_elmos = []
    extra_vncs = []

    for line in data:
        okay = True
        matched = False
        line_data = clean_line(line)
        for i in range(len(line_data)):
            word = line_data[i]
            if "_None" in word:
                okay = False
                break
            if re.match(VN_RE, word):
                word, vnc = word.split("_")
                matched = True
                if "_" in word:
                    okay = False
                    break
                vnc = vnc.split("-")[0]
                line_data[i] = word
                lemma = l.lemmatize(word.lower(), "v")

                if lemma not in d:
                    d[lemma] = 0
                d[lemma] += 1
                if (vnc not in LITS and vnc not in METS):
                    okay = False
                    break

        if okay and matched:
            extra_elmos.append(" ".join(line_data)+"\n")
            extra_vncs.append(line)
    print (len(extra_elmos))
    return extra_elmos, extra_vncs


def convert_syn_to_seq(line_data_dict):
    sent_type = "syn"
    sent_no = 0
    res = []

    for line_data_key in sorted(line_data_dict.keys()):
        for line_data in line_data_dict[line_data_key]:
            words = clean_line(line_data[0])
            index = int(line_data[1])-1
            if words[index] == "None":
                print (words, index)
                continue
            if PATTERN_DICT[line_data_key] == 1:
                tags = [0 if i != index else 1 for i in range(len(words))]
            else:
                tags = [0] * len(words)

            tagged_words = [words[i] if tags[i] == 0 else "M_" + words[i] for i in range(len(words))]

            poses = [POS_DICT[p[1]] for p in SDP.tag_sentence(words)]
            res.append((sent_type, str(sent_no), " ".join(words), tags, poses, " ".join(tagged_words), sent_type))
            sent_no += 1
            if sent_no % 10 == 0:
                print (sent_no)
    return res


def convert_vn_to_seq(elmo_lines):
    sent_type = "vn"

    sent_no = 0
    all_pos = set()
    csv_out = []
    d = {}
    for line in elmo_lines:
        okay = True
        sent_words, tags, tagged_words = [], [], []

        line_data = line.split()
        for word in line_data:
            if re.match(VN_RE, word):
                word, vnc = word.split("_")
                vnc = vnc.split("-")[0]

                if vnc in METS:
                    tags.append(1)
                    tagged_words.append("M_" + word)
                elif vnc in LITS:
                    tags.append(0)
                    tagged_words.append(word)
                else:
                    okay = False
                    break
            else:
                tags.append(0)
                tagged_words.append(word)

            sent_words.append(word)

        if okay:
            poses = [POS_DICT[p[1]] for p in SDP.tag_sentence(line_data)]
            all_pos |= set(poses)
            csv_out.append((sent_type, str(sent_no), " ".join(sent_words), tags, poses, " ".join(tagged_words), sent_type))
            if sent_no % 10 == 0:
                print (sent_no)
            sent_no += 1
    write_output(csv_out, "VUA_seq_formatted_vn_extra.csv")

def convert_syn_to_cls(line_data_dict):
    sent_type = "syn"
    sent_no = 0
    res = []

    for line_data_key in sorted(line_data_dict.keys()):
        tag = PATTERN_DICT[line_data_key]
        verb = line_data_key.split("-")[0]
        for line_data in line_data_dict[line_data_key]:
            words = clean_line(line_data[0])
            index = str(int(line_data[1])-1)
            res.append([sent_type, sent_no, verb, " ".join(words), index, tag])
            sent_no += 1
            if index is None:
                print (words, verb)
            if sent_no % 100 == 0:
                print (sent_no)

    return res

def convert_vn_to_cls(elmo_lines):
    text_id = "vn"
    sentence_id = 0
    csv_out = []
    counts = {}
    for line in elmo_lines:
        line_data = line.split()
        res = gao_line_data(line_data)
        if res:
            verb, index, tag = res
            if verb:
                if verb not in counts:
                    counts[verb] = 0
                counts[verb] += 1
                csv_out.append([text_id, sentence_id, verb, " ".join(line_data), index, tag])
                sentence_id += 1
            else:
                print (verb, line)
        else:
            print (line)
    print (len(csv_out))
    write_output(csv_out, "VUA_vn_extra.csv")


def convert_to_moh(data):
    arg1 = ""
    arg2 = ""
    csv_out = []

    for line in data:
        line_data = clean_line(line)
        verb, index, tag = gao_line_data(line_data)
        csv_out.append([arg1, arg2, verb, " ".join(line_data), index, tag])

    write_output(csv_out, "extra_mohx.gao")

    return


def convert_to_trofi(data):
    csv_out = []
    sent_no = 0
    for line in data:
        line_data = clean_line(line)
        verb, index, tag = gao_line_data(line_data)
        csv_out.append((verb, str(sent_no), " ".join(line_data), index, tag))

        sent_no += 1

    write_output(csv_out, "extra_trofi.gao")

def convert_lcc(corpus):
    elmo = []
    seq = []
    m_seq = []
    
    for i in range(len(corpus.instances)):
        instance = corpus.instances[i]
        elmo.append(instance.text() + "\n")
        seq.append(("lcc", instance.id, instance.text(), str(["1" if ("source" in w.met or "target" in w.met) else "0" for w in instance.words]), str([POS_DICT[w.pos] for w in instance.words]), " ".join(["M_" + w.text if ("source" in w.met or "target" in w.met) else w.text for w in instance.words]), "lcc"))

        m_seq.append(("lcc", instance.id, instance.text(), str([convert_met(w) for w in instance.words]), str([POS_DICT[w.pos] for w in instance.words]), " ".join(["M_" + w.text if convert_met(w) > 0 else w.text for w in instance.words]), "lcc"))
        

    with open("lcc_elmo.txt", "w", encoding="utf-8") as elm_out:
        elm_out.writelines(elmo)
        
    with open("lcc_seq.csv", "w", encoding="utf-8") as seq_out:
        w = csv.writer(seq_out)
        w.writerows(seq)

    with open("lcc_seq_mult.csv", "w", encoding="utf-8") as mult_seq_out:
        w = csv.writer(mult_seq_out)
        w.writerows(m_seq)

def add_deps(input_file, corpus):
    lines = csv.reader(open(input_file, encoding='latin-1'))
    with open(input_file + ".dep", "w", encoding="latin-1") as output_file:
        wr = csv.writer(output_file)
    
        for l in lines:
            key = l[0] + "-" + l[1]
            for sent in corpus.instances:
                if sent.source_file + "-" + sent.id == key:
                    if len(l[2].split()) != len(sent.words):
                        l.append([""]*len(l[2].split()))
                        l.append([""]*len(l[2].split()))
                        l.append([""]*len(l[2].split()))
                        l.append([""]*len(l[2].split()))
                    else:
                        l.append([sent.find_head(w).text if sent.find_head(w) else "" for w in sent.words])
                        l.append([" ".join([w2.text for w2 in sent.find_dependencies(w)]) for w in sent.words])
                        l.append([w.dep[-3] if w.dep else "" for w in sent.words])
                        l.append([" ".join([w2.dep[-3] for w2 in sent.find_dependencies(w)]) for w in sent.words])
            wr.writerow(l)
        
def main():
    corpus = CorpusLoaders.VUAMCCorpus()
    add_deps(VUA_TR_SEQ, corpus)
    add_deps(VUA_TE_SEQ, corpus)
    add_deps(VUA_VA_SEQ, corpus)
    
if __name__ == "__main__":
    main()
