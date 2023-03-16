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

        # Task 3: optional exercise
        if fact == "want to buy" and filterNoun[0] == "book":
            knowledge.add(("person1", "book title", "?"))
            knowledge.add(("person1", "book author", "?"))

        if fact == "want to buy" and filterNoun[0] == "pen":
            knowledge.add(("person1", "pen colour", "?"))

    else:
        question = "How can I help you? "
        helpRequest = input(question)
        # process reply
        tokens = nltk.word_tokenize(helpRequest)
        print("Tokens: \n" + str(tokens) + "\n")
        taggedTokens = nltk.pos_tag(tokens)
        print("Tagged tokens: \n" + str(taggedTokens) + "\n")

        # get verb
        filterVerb = [word for (word, pos)
                      in taggedTokens if pos.startswith("VB")]
        print("Verb: \n" + str(filterVerb) + "\n")

        if any(item in ["buy", "order", "purchase"] for item in filterVerb):
            # if "buy" in filterVerb:
            knowledge.add(("person1", "want to buy", "?"))
        else:
            print("I cannot help you with that. Can I help you to buy something?")

    print()
