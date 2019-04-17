import SDP
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

PREPS = ["for", "which", "over", "up", "away", "on", "onto", "off", "out", "over", "into", "in", "on", "by", "dead", "onto", "at", "about", "together", "in", "down", "against", "with", "time", "life"]

RECOGNIZED_PATTERNS = PREPS + ["neg", "to", "~to", "~dobj", "be", "that-comp", "nsubj", "dobj", "pass", "~pass", "to-be-adj", "by-np", "wh", "adv", "pro-vp", "adj", "nmod", "to-np"]

def match_line(patterns, line, verb):
    parse = SDP.parse_sentence(line)
    verb_node = None
    conll_data = [p.split() for p in list(parse)[0].to_conll(style=10).split("\n")]

    for word_data in conll_data:
        if len(word_data) > 1 and lemmatizer.lemmatize(word_data[1], "v") == verb and word_data[3][0] == "V":
            verb_node = word_data

    if not verb_node:
        return {}

    neg, nmod, adj, pro_vp, adv, nsubj, wh, dobj, subj, to_phrase, passive, verb_compliment, pn_subj_of_complement, to_be_adj, that_comp, by_np, to_be = False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False
    to_np, pro_found, to_found, be_found, that_found, by_found, last_dobj = False, False, False, False, False, False, False

    prep_results = {p: False for p in PREPS}

    for i in range(len(conll_data)):
        word_data = conll_data[i]

        if len(word_data) > 1:
            if word_data[1] == "to":
                to_found = word_data[-4]
            if word_data[1] == "be":
                be_found = word_data[-4]
            if word_data[1] == "that":
                that_found = word_data[-4]
            if word_data[1] == "by":
                by_found = word_data[-4]
            if word_data[3] in ["PRP", "PRP$"]:
                pro_found = word_data[-4]

            if word_data[-4] == verb_node[0]:      # a direct dependent
                if "dobj" == word_data[-3]:
                    dobj = True
                    last_dobj = word_data
                if "nsubj" == word_data[-3]:
                    nsubj = True
                if word_data[3][0] == "W":
                    wh = True
                if word_data[0] == to_found:
                    to_phrase = True
                    if word_data[3][0] == "N":
                        to_np = True
                if word_data[0] == that_found:
                    that_comp = True
                if word_data[0] == by_found:
                    by_np = True
                if "pass" in word_data[-3]:
                    passive = True
                if word_data[3] == "RB":
                    adv = True
                if word_data[3] == "JJ":
                    adj = True
                if word_data[3][0] == "V" and word_data[0] == pro_found:
                    pro_vp = True
                if word_data[-3] == "nmod":
                    nmod = True
                if word_data[-3] == "neg":
                    neg = True

            if last_dobj and word_data[-4] == last_dobj[0]:
                if word_data[0] == to_found and word_data[0] == be_found:
                        to_be = True
                        if word_data[3][0] == "J":
                            to_be_adj = True


            for p in PREPS:
                try:
                    if p == word_data[1] and (word_data[-4] == verb_node[0] or conll_data[int(word_data[-4])-1][-4] == verb_node[0]):
                        prep_results[p] = True
                except IndexError as e:
                    print (e)
                    continue
    res = {}

    for pattern_set in patterns:
        matches = True
        if not pattern_set or len(pattern_set) < 1:
            raise Exception("We don't have a rule for this pattern: " + pattern_set)

        for pattern in pattern_set.split():
            if pattern not in RECOGNIZED_PATTERNS:
                raise Exception ("We don't have a rule for this pattern: " + pattern)
            if pattern == "to" and not to_phrase:
                matches = False
            if pattern == "~to" and to_phrase:
                matches = False
            if pattern == "pass" and not passive:
                matches = False
            if pattern == "~pass" and passive:
                matches = False
            if pattern == "dobj" and not dobj:
                matches = False
            if pattern == "~dobj" and dobj:
                matches = False
            if pattern == "nsubj" and not nsubj:
                matches = False
            if pattern == "~nsubj" and nsubj:
                matches = False
            if pattern == "to-be-adj" and not to_be_adj:
                matches = False
            if pattern == "that-comp" and not that_comp:
                matches = False
            if pattern == "by-np" and not by_np:
                matches = False
            if pattern == "wh" and not wh:
                matches = False
            if pattern == "to-be" and not to_be:
                matches = False
            if pattern == "adv" and not adv:
                matches = False
            if pattern == "adj" and not adj:
                matches = False
            if pattern == "pro-vp" and not pro_vp:
                matches = False
            if pattern == "nmod" and not nmod:
                matches = False
            if pattern == "to-np" and not to_np:
                matches = False
            if pattern == "neg" and not neg:
                matches = False
            for p in PREPS:
                if pattern == p and not prep_results[p]:
                    matches = False

        if matches:
            res[verb + "-" + pattern_set] = line.strip() + ";;" + verb_node[0] + "\n"

    return res

def get_sentences(f, syn_patterns, n = 10):
    totals = {}
    keys = []
    for item in syn_patterns:
        for pattern in item[1]:
            keys.append(item[0] +"-"+ pattern)

    c = 0
    final_res = {k:[] for k in keys}
    totals.update(final_res)

    for line in open(f, encoding='utf-8').readlines():
        c += 1
        lemmas = [lemmatizer.lemmatize(w, "v") for w in line.split()]
        for verb, patterns in syn_patterns:
            if verb in lemmas:
                res = (match_line(patterns, line, verb))
                for verb_pattern in res:
                    final_res[verb_pattern].append(res[verb_pattern])

        if c % 100 == 0:
            print (c)
            for k in final_res:
                totals[k] += final_res[k]
            print({k: len(totals[k]) for k in totals})
#            if c == 100:
#                mode = "w"
#            else:
            mode = "a"
            for pattern in final_res:
                with open("C:/Users/Kevin/PycharmProjects/metaphor/corpora/additional_met/syntax/" + "_".join(pattern.split()), mode, encoding="utf-8") as output_file:
                    for r in final_res[pattern]:
                        output_file.write(r)

            final_res = {k:[] for k in keys}


def main():

    # this batch, lines done : 18700
    '''
    syn_patterns = [
        ("encourage", ["to", "dobj ~to"]),
        ("blow", ["over", "up", "away"]),
        ("conduct", ["pass", "~pass dobj"]),
        ("find", ["nsubj pro-vp", "out", "dead", "nsubj dobj to-be-adj"]),
        ("fall", ["nsubj adv", "nsubj ~dobj", "wh", "in", "to"]),
        ("hold", ["nsubj dobj by-np", "nsubj dobj out", "onto", "on", "out", "dobj that-comp", "dobj adj", "at", "down"]),
        ("bring", ["up", "about", "dobj to-np", "together", "in"]),
        ("allow", ["dobj to be", "which"])
                    ]
    '''
    syn_patterns = [
        ("spend", ["nsubj dobj on", "nsubj ~dobj", "time", "life"]),
        ("play", ["against dobj", "down dobj", "on dobj", "with", "nsubj ~dobj"]),
        ("meet", ["nsubj ~dobj", "for", "at", "to"]),
        ("suggest", ["neg"]),
        ("allow", ["neg"])
    ]
    for i in [5]:
        f = "../lexical_resources/wikipedia/cleaned/" + str(i) + ".clean"
        get_sentences(f, syn_patterns)

if __name__ == "__main__":
    main()