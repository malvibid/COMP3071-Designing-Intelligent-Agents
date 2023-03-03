from bs4 import BeautifulSoup
import nltk
import numpy as np
import requests
# If running nltk for the first time, will need to download the following nltk datasets: (uncomment next line)
# nltk.download(['punkt', 'averaged_perceptron_tagger'])


class FigurativeLanguageGenerator:

    def __init__(self, url='http://bonnat.ucd.ie/jigsaw/index.jsp', fname='mexica.txt'):
        '''
        This method initializes the FigurativeLanguageGenerator class with two optional parameters:
        url and fname.

        :param url:
        url is the URL of a website that is used to retrieve figurative language terms.
        The default value of url is 'http://bonnat.ucd.ie/jigsaw/index.jsp'.

        :param fname:
        fname is the name of a file that contains the text to which the figurative language terms will be added.
        The default value of fname is 'mexica.txt'.
        '''
        self.url = url
        self.fname = fname

    def read_file(self):
        '''
        Reads the contents of a file and returns them as a string.

        Returns:
        str: The contents of the file specified in the 'fname' parameter of the constructor
        '''
        # Open the file and read its contents, and return the file contents
        return open(self.fname, "r").read()

    def tokenize_and_pos_tag(self, story):
        """
        Tokenizes and performs part-of-speech tagging on a given text.

        Args:
            story (str): The input text to be tokenized and tagged.

        Returns:
            list: A list of tuples, where each tuple contains a word from the input text and its corresponding part-of-speech tag.

        """
        # Tokenize the input text
        listOfWords = nltk.word_tokenize(story)
        print("Tokens: \n" + str(listOfWords) + "\n")

        # Perform part-of-speech tagging on the tokenized text
        taggedListOfWords = nltk.pos_tag(listOfWords)

        # Print the tagged list of words (for debugging purposes)
        print("Tagged tokens: \n" + str(taggedListOfWords) + "\n")

        # Return the tagged list of words
        return taggedListOfWords

    def add_figurative_language_terms_to_dict(self, story):
        """
        Adds figurative language terms to a given story and returns the updated story as a list of words.

        Args:
            story (str): The story to which figurative language terms will be added.

        Returns:
            list: A list of words representing the updated story, with some words replaced by figurative language terms.
        """

        # Extract all adjectives from the story and get their base forms
        pos_tag = self.tokenize_and_pos_tag(story)
        all_adj = self.filter_adjectives(pos_tag)
        print("All adjectives: \n" + str(all_adj) + "\n")

        # Get the figurative language terms corresponding to the adjectives
        query_params = [tup[0] for tup in all_adj]
        figurative_language_terms = self.get_figurative_language_terms(
            query_params)

        # Replace some words in the story with figurative language terms
        new_text = []
        for token in pos_tag:
            if token[-1] == 'JJ':
                # checking that the adjective has a list of expression. If its empty, its appended to new_text as it is (see else block).
                expressionList = figurative_language_terms[token[0]]
                if expressionList:
                    # Choose a random figurative language term for the word
                    position = self.rand_pos(len(expressionList))
                    # Append updated token to new_text
                    new_text.append(
                        f"{token[0]} as {figurative_language_terms[token[0]][position]}")
                # Keep the original word
                else:
                    new_text.append(token[0])
            # Append all non-adjective tokens to new_text
            else:
                new_text.append(token[0])

        print("New Text: \n" + str(new_text) + "\n")
        # Return the updated story as a list of words
        return new_text

    def get_figurative_language_term(self, param):
        '''
        This method takes a string as input and queries a website to retrieve a list of figurative
        language terms that are related to the input string. It returns a list of the retrieved terms.


        Args:
        param (str): The input string to query the website with.

        Returns:
         list: A list of figurative language terms that are related to the input string.
        '''

        # Set up the query parameters
        params = {'q': param}

        # Send a GET request to the website with the query parameters
        response = requests.get(self.url, params=params)
        # print(response.status_code)
        # print(response.text)

        # Raise an exception if the response status code is not 200. response.raise_for_status() will raise an HTTPError if the HTTP request returned an unsuccessful status code.
        response.raise_for_status()

        # Parse the HTML response and extract the table that contains the figurative language terms
        soup_html = BeautifulSoup(response.text, 'html.parser')

        # Extract the figurative language terms from the table and return them as a list
        aTags = soup_html.select(
            "body > table tr > td:nth-child(2) > table tr:nth-child(2) > td:nth-child(1) table a")

        all_figurative = []
        for aTag in aTags:
            # Getting the text from the <a> tags
            all_figurative.append(aTag.text)

        # print("Simple Elaborations: \n" + str(all_figurative) + "\n")

        return all_figurative

    def get_figurative_language_terms(self, query_params):
        '''

        ['influential', 'powerful', 'important', 'cacao', 'violent', 'medical', 'grateful']

        This method takes a list of query parameters and gets the figurative language terms for each query
        parameter by calling the get_figurative_language_term method for each parameter.

        The resulting figurative language terms are stored in a dictionary where the keys are the query parameters and
        the values are lists of figurative language terms.

        The dictionary is returned.
        :param query_params: a list of query parameters
        :return: a dictionary where the keys are the query parameters and the values are lists of figurative language terms.

        '''

        # create an empty dictionary to store the figurative language terms
        figurative_language_terms = {}
        print("Query params: \n" + str(query_params) + "\n")

        # iterate over each query parameter
        for param in query_params:
            # get the figurative language terms for the current parameter
            figurative_language_terms[param] = self.get_figurative_language_term(
                param)
        print("Figurative language terms: \n" +
              str(figurative_language_terms) + "\n")
        # return the dictionary of figurative language terms
        return figurative_language_terms

    def filter_adjectives(self, taggedListOfWords):
        '''

        This method filters the list of words that are adjectives from a list of words that are tagged
        with their parts of speech returns a list of tuples containing  only adjectives.
        :param taggedListOfWords:
            A list of tuples containing tagged words.
        :return:
            A filtered list of tuples containing only adjectives.
        '''
        # val = []
        # for tw in taggedListOfWords:
        #     if tw[-1] == 'JJ':
        #         print(tw)
        #         val.append(tw)

        return list(filter(lambda tlw: tlw[-1] == 'JJ', taggedListOfWords))

    def rand_pos(self, max_val):
        """
        Generates a random integer between 0 and max_val (exclusive) using numpy's randint method.

        Args:
            max_val (int): The upper limit (exclusive) of the range of possible integers to generate.

        Returns:
            A randomly generated integer between 0 and max_val (exclusive).
        """
        return np.random.randint(low=0, high=max_val, size=10)[-1]

    def generate_text_with_figurative_language(self):
        """
        Takes a text file and generates a new text file with figurative language terms added.
        Reads the text file specified in the constructor, then adds figurative language terms to the
        text using the add_figurative_language_terms_to_dict method. The resulting text is returned
        as a string.

        :return: A string containing the text with figurative language terms added.
        """

        # Read the contents of the file into a string
        story = self.read_file()
        print("\nStory: \n" + story + "\n")

        # Add figurative language terms to the string
        self.add_figurative_language_terms_to_dict(story)

        # Convert the list of words to a single string

        # return new_text


fig_lang_gen = FigurativeLanguageGenerator()
generated_text = fig_lang_gen.generate_text_with_figurative_language()
# print(generated_text)
