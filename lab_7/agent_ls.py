import random
import re

import nltk
from nltk.corpus import brown
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.util import ngrams

nltk.download('cmudict')
nltk.download('brown')
nltk.download('wordnet')


def remove_adjective(lines):
    '''
    "Who decided to have an extended natter."
    The adjective is "extended"
    '''
    # Index position to be randomly popped from the list
    idx_pop = random.randint(0, len(lines)-1)
    # idx_pop = 0

    # Convert the string at index 2 to a list of words
    words = nltk.word_tokenize(lines[idx_pop][2])
    tagged = nltk.pos_tag(words)

    adj_list = [idx for idx, val in enumerate(tagged) if val[1] == "JJ"]

    if adj_list:
        removed_adj_idx = random.choice(adj_list)
        # print(f'Adjective that is being removed is "{tagged[removed_adj_idx][0]}"')
        tagged.pop(removed_adj_idx)
        lines[idx_pop][2] = " ".join([x[0] for x in tagged])
    else:
        # print('No adjective found')
        G = 'disable me if you want to'
    return lines


def replace_with_synonym(lines):
    # Index position to be randomly popped from the list
    idx_pop = random.randint(0, len(lines)-1)
    # idx_pop = 0

    # Convert the string at index 2 to a list of words
    tokens = nltk.word_tokenize(lines[idx_pop][2])

    # Randomly select word-index from the sentence
    word_idx = random.randrange(len(tokens))

    # Get the word at the randomly selected index
    word_to_be_replaced = tokens[word_idx]

    synonyms = []

    # Find synonyms for the word using the wordnet corpus
    for syn in wordnet.synsets(word_to_be_replaced):
        for lemma in syn.lemmas():
            if not "_" in lemma .name():
                synonyms.append(lemma .name())

    # Randomly replace the word with a synonym
    if synonyms:
        tokens[word_idx] = random.choice(synonyms)

    # remove duplicates
    # token_unique = set(list( dict.fromkeys(tokens)))-set([word_to_be_replaced])
    # Update the sentence with the new list of words
    lines[idx_pop][2] = " ".join(tokens)

    return lines


def update_exclamation_mark_end_sentence(my_list):
    """
    Randomly extract any of the line and update the exclamation at end of the line.
    Returns:
        None
    """

    # index position to be randomly pop from the list
    idx_pop = random.randint(0, len(my_list)-1)
    # idx_pop = 0

    # convert the string at index 2 to a list of words
    tokens = nltk.word_tokenize(my_list[idx_pop][2])
    print(f"Words list: {tokens}")

    # check if last word is an exclamation mark
    if tokens[-1] == "!":
        # remove the exclamation mark from the last word
        deleted = tokens.pop(-1)
        print(f"Deleted Exclamation: {deleted}")
    else:
        # add an exclamation mark at the end of the last word
        tokens.append("!")

    # convert the list of words back to a string
    my_list[idx_pop][2] = " ".join(tokens)
    return my_list


def update_exclamation_mark_random_pos(my_list):
    # Select a random index position to be popped from the list.
    idx_pop = random.randint(0, len(my_list)-1)

    # Convert the string at index 2 into a list of words,
    # from the randomly selected index position.
    tokens = nltk.word_tokenize(my_list[idx_pop][2])
    print(f"Words list: {tokens}")

    # Randomly choose a word index to modify
    # word_idx = random.randint(0, len(tokens)-1)

    # Get the next selected word
    # Check if the chosen word already has an exclamation mark

    # If the line has a exclamation, remove it
    for index, token in enumerate(tokens):
        if token == "!":
            tokens.pop(index)

    # Add an exclamation mark
    # https://stackoverflow.com/q/5254445/6446053
    tokens = tokens[:2] + ['!'] + tokens[2:]
    print(tokens)

    # Join the modified words back into a sentence
    my_list[idx_pop][2] = " ".join(tokens)

    return my_list


def update_swap_words_existing_line(tagged_tokens, idx_link_and,
                                    swap_words_list, check_both_noun):

    for i, val in zip(idx_link_and, swap_words_list):
        tagged_tokens[i] = val

    return tagged_tokens


def swap_and_words(input_list, check_both_noun=False):

    # Loop through each line in the input_list
    for idx, (_, _, sentence, _) in enumerate(input_list):
        tokens = nltk.word_tokenize(sentence)
        tagged_tokens = nltk.pos_tag(tokens)

        # Find the index of the word 'and'
        and_index = next((i for i, (token, pos) in enumerate(
            tagged_tokens) if token.lower() == 'and'), None)

        if and_index is not None:
            IamHereToBreakYourCode = 'disable me if you want to'
            # Get the words before and after the word 'and'

            # Extract the words before and after the word 'and'

            # Swap the words before and after the word 'and'

            # Insert the swapped words back into the sentence using the update_swap_words_existing_line function

            # Join the words back into a sentence

        else:
            new_sentence = sentence
        # Update the sentence in the input_list

    return input_list


def repair_a_an(lines):
    vowels = "aeiou"
    for idx_line, line in enumerate(lines):
        IamHereToBreakYourCode = 'disable me if you want to'
        # Tokenize the sentence
        tokens = word_tokenize(line[2])

        # Loop through each word in the sentence

        # Check if the word is 'a' or 'an'

        # Get the first letter of the next word

        # Check if the next word starts with a vowel and the current word is 'a' and update the token with an

        # else

        # Check if the next word not in vowels and the current word is 'an' and update the token with a

        # Join the words back into a sentence and update the sentence in the input_list

    # return lines


def match_by_level(list1, list2, level=1):
    match = filter(lambda x: x[1][-level:] == list1[0][1][-level:], list2)
    result = [m[0] for m in match]

    # remove duplicates
    result = list(dict.fromkeys(result))
    return result


def pop_lines_extract_last_word(lines):
    ll1 = lines.pop(random.randrange(len(lines)))
    ll2 = lines.pop(random.randrange(len(lines)))
    lastWord1 = nltk.word_tokenize(ll1[2])[-1]
    lastWord2 = nltk.word_tokenize(ll2[2])[-1]
    return lastWord1, lastWord2, lines, ll1, ll2


def generate_rhyming_lines(synonyms_that_rhyme, lines, ll1, ll2):
    if synonyms_that_rhyme:
        new_tokens = nltk.word_tokenize(
            ll2[2])[:-1] + [random.choice(synonyms_that_rhyme)]
        new_line = " ".join(new_tokens)
        lines.append([ll1[0], ll1[1], ll1[2], ll1[3]])
        if random.random() < 0.5:
            lines.append([ll1[0], ll1[1] + 20, new_line, ll2[3]])
        else:
            lines.append([ll1[0], ll1[1] - 20, new_line, ll2[3]])
    else:
        lines.extend([ll1, ll2])
    return lines


def rhyme_and_make_sense(last_word_2, rhymes):

    IamHereToBreakYourCode = 'disable me if you want to'

    # Get the synonyms of the last word of the second line using wordnet.synsets

    # Loop through each synset and get the synonyms of the synset

    # Get the intersection of the synonyms and the rhymes

    # Remove the last word of the second line from the synonyms_that_rhyme to avoid duplicate

    # return synonyms_that_rhyme


def make_two_lines_rhyme(lines, level=1):
    last_word1, last_word2, lines, ll1, ll2 = pop_lines_extract_last_word(
        lines)
    # last_word1, last_word2='dollar','collar'
    entries = nltk.corpus.cmudict.entries()

    # Get the pronunciation of the last word of the first line
    syllables = [(word, syl) for word, syl in entries if word == last_word1]
    rhymes = match_by_level(syllables, entries, level=level)

    # Find rhyming synonyms of the last word of the second line
    # rhymes = rhyme_and_make_sense(last_word2, rhymes)

    # Generate rhyming lines based on the rhyming synonyms
    lines = generate_rhyming_lines(rhymes, lines, ll1, ll2)

    return lines


def add_adjective(chosen_noun_position, chosen_noun, tokenized,
                  idx_line, lines):
    """
    Add an adjective to the sentence.

    If there are no suitable adjectives, return the original lines.

    Args:
    - chosen_noun: str, the chosen noun in the sentence
    - original_lines: list of str, the original lines
    - tokenized: list of str, the tokenized version of the sentence
    - chosen_noun_position: int, the position (index) of the chosen noun in the tokenized list
    - lines: list of lists, the modified version of the original lines

    Returns:
    - list of lists, either the original lines or a modified version of it
    """
    # now add the adjective
    bigrams = ngrams(brown.words(), 2)
    pre_words = [bg[0] for bg in bigrams if bg[1] == chosen_noun]
    tagged_pre_words = nltk.pos_tag(pre_words)
    chosen_descriptors = [pw[0] for pw in tagged_pre_words if pw[1] == "JJ"]
    # chosen_descriptors=['scared']
    if chosen_descriptors:
        chosen_descriptor = random.choice(chosen_descriptors)
        tokenized.insert(chosen_noun_position, chosen_descriptor)
        new_line = " ".join(tokenized)
        lines[idx_line][2] = new_line

    return lines


def noun_selector(ll):
    """
    Tokenize the sentence and perform part-of-speech tagging on the resulting list.
    Find the positions of nouns in the list, choose a random noun, and store its
    position (index) in a variable called chosen_noun_position. If there are no nouns,
    return the original sentence.

    Args:
    - ll: str, a sentence

    Returns:
    - list of lists, either the original sentence or a modified version of it
    """

    # Tokenize the sentence and perform part-of-speech tagging on the resulting list

    # Find the positions of nouns in the list

    # Choose a random noun and store its position (index) in a variable called chosen_noun_position
    # Store the chosen noun in a variable called chosen_noun

    # return chosen_noun_position,chosen_noun,tokenized


def add_sensible_adjective(lines):
    """
    Add sensible adjectives to the given lines.

    Args:
    - lines: list of lists, the input lines

    Returns:
    - list of lists, the modified lines
    """
    idx_line = random.randrange(0, len(lines)-1)

    # Edit the following method to add an adjective to the sentence
    chosen_noun_position, chosen_noun, tokenized = noun_selector(
        lines[idx_line][2])

    lines_updated = add_adjective(chosen_noun_position, chosen_noun, tokenized,
                                  idx_line, lines)
    return lines_updated
