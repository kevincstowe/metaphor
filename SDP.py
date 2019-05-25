from nltk.parse.stanford import StanfordDependencyParser
from nltk.tag.stanford import StanfordPOSTagger

path_to_jar = '../stanford-parser-full-2018-10-17/stanford-parser.jar'
path_to_models_jar = '..//stanford-parser-full-2018-10-17/stanford-parser-3.9.2-models.jar'
path_to_pos_tagger = '../stanford-postagger-2018-10-16/models/english-bidirectional-distsim.tagger'
path_to_pos_jar = '../stanford-postagger-2018-10-16/stanford-postagger.jar'

dependency_parser = StanfordDependencyParser(path_to_jar=path_to_jar, path_to_models_jar=path_to_models_jar)
pos_tagger = StanfordPOSTagger(path_to_pos_tagger, path_to_pos_jar)

def tag_sentence(sentence):
    if type(sentence) != list:
        sentence = sentence.split()
    return pos_tagger.tag(sentence)


def parse_sentences(sentences):
    return [dependency_parser.parse(s) for s in sentences]

def parse_sentence(sentence):
    if type(sentence) != list:
        sentence = sentence.split()
    return dependency_parser.parse(sentence)

def trim_clauses(sentence):
    def find_deps(node, all_nodes, to_search=None, res=None):
        if not res:
            res = [node]
            to_search = [node]

        #base case : to_search is empty
        if not to_search:
            return res

        if node not in res:
            res.append(node)

        node_index = int(node[0])
        for node2 in all_nodes:
            if int(node2[6]) == node_index:
                to_search.append(node2)
        next_node = to_search.pop()
        return find_deps(next_node, all_nodes, to_search, res)

    try:
        deps = [d.split("\t") for d in list(parse_sentence(sentence))[0].to_conll(style=10).split("\n")][:-1]
    except StopIteration as e:
        print (e)
        deps = []

    res = []
    for i in range(len(deps)):
        d = deps[i]
        if len(d) and d[3][0] == "V":
            new_deps = find_deps(d, deps, [], [])
            res.append((d, new_deps))

    return res

if __name__ == "__main__":
    print (tag_sentence("can a can can a can"))
