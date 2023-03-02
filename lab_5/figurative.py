import nltk
import numpy as np
import requests
# Will need to download the following nltk dataset if running nltk for the first time.
# nltk.download(['punkt'])


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

        # Print the tagged list of words (for debugging purposes)

        # Return the tagged list of words
        # return taggedListOfWords

    def add_figurative_language_terms_to_dict(self, story):
        """
        Adds figurative language terms to a given story and returns the updated story as a list of words.

        Args:
            story (str): The story to which figurative language terms will be added.

        Returns:
            list: A list of words representing the updated story, with some words replaced by figurative language terms.
        """

        # Extract all adjectives from the story and get their base forms

        # Get the figurative language terms corresponding to the adjectives

        # Replace some words in the story with figurative language terms

        # Choose a random figurative language term for the word

        # Keep the original word

        # Return the updated story as a list of words
        # return new_text

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

        # Send a GET request to the website with the query parameters

        # Raise an exception if the response status code is not 200

        # Parse the HTML response and extract the table that contains the figurative language terms

        # Extract the figurative language terms from the table and return them as a list

        # return all_figurative

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

        # iterate over each query parameter

        # get the figurative language terms for the current parameter

        # return the dictionary of figurative language terms
        # return figurative_language_terms

    def filter_adjectives(self, taggedListOfWords):
        '''

        This method filters the list of words that are adjectives from a list of words that are tagged
        with their parts of speech returns a list of tuples containing  only adjectives.
        :param taggedListOfWords:
            A list of tuples containing tagged words.
        :return:
            A filtered list of tuples containing only adjectives.
        '''
        # return

    def rand_pos(self, max_val):
        """
        Generates a random integer between 0 and max_val (exclusive) using numpy's randint method.

        Args:
            max_val (int): The upper limit (exclusive) of the range of possible integers to generate.

        Returns:
            A randomly generated integer between 0 and max_val (exclusive).
        """
        # return

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
        print(story)

        # Add figurative language terms to the string
        self.tokenize_and_pos_tag(story)

        # Convert the list of words to a single string

        # return new_text


fig_lang_gen = FigurativeLanguageGenerator()
generated_text = fig_lang_gen.generate_text_with_figurative_language()
# print(generated_text)
