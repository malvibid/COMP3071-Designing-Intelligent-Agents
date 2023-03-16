from pprint import pprint
import random
import nltk

knowledge = {("person1", "name", "?"),
             ("person1", "town", "?"),
             ("person1", "street", "?")}

active = True
while active:
    unknowns = {(person, fact, value) for (person, fact, value)
                in knowledge if value == "?"}
    print("UNKNOWN:")
    pprint(unknowns)
    print("KNOWN:")
    pprint(knowledge - unknowns)
    if unknowns:  # is non-empty
        person, fact, value = random.choice(list(unknowns))
        question = f"What is your {fact}?"

        # remove current query from knowledge
        currentQuery = (person, fact, value)
        knowledge.remove(currentQuery)

        reply = input(question)

        #  process reply
        tokens = nltk.word_tokenize(reply)
        print("Tokens: \n" + str(tokens) + "\n")
        taggedTokens = nltk.pos_tag(tokens)
        print("Tagged tokens: \n" + str(taggedTokens) + "\n")

        # get proper noun
        filterNoun = [word for (word, pos) in taggedTokens if pos == "NNP"]
        # if not proper noun, get noun
        if not filterNoun:
            filterNoun = [word for (word, pos) in taggedTokens if pos == "NN"]
        print("Noun: \n" + str(filterNoun) + "\n")

        currentQuery = (person, fact, filterNoun[0])
        knowledge.add(currentQuery)

    else:
        question = "How can I help you? "
        helpRequest = input(question)
        # to fill in - process reply
    print()
