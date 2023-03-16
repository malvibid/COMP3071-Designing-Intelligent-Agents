from pprint import pprint
import random

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
        currentQuery = (person, fact, reply)
        knowledge.add(currentQuery)

    else:
        question = "How can I help you? "
        helpRequest = input(question)
        # to fill in - process reply
    print()
