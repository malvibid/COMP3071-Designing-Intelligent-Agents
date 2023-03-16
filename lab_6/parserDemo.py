import nltk
from pyStatParser.stat_parser import Parser
from pprint import pprint

parser = Parser()
sentence = "My street is Triumph Road"
t = parser.parse(sentence)
t.draw()
pprint(t)

print("#######################")
for st in t.subtrees():
    print(st.label())
    if st.label() == "VP":
        vpTree = st
for st in vpTree.subtrees():
    if st.label() == "NP":
        npTree = st
print(" ".join(npTree.leaves()))
