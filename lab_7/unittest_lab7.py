from agent_ls import (update_exclamation_mark_end_sentence,
                      update_exclamation_mark_random_pos, swap_and_words,
                      repair_a_an)
import random
# random.seed(0)
import unittest


class TestCase(unittest.TestCase):
    def test_remove_adjective(self):
        from agent_ls import remove_adjective
        lines = [["1", "John", "Who decided to have an extended natter."]]
        expected_result = [["1", "John", "Who decided to have an natter ."]]
        result = remove_adjective(lines)
        self.assertEqual(result, expected_result)

    def test_replace_with_synonym(self):
        from agent_ls import replace_with_synonym
        input_lines = [['id1', 'author1', "Who decided to have an extended natter", 'source1'],
                       ['id1', 'author1', "Once there was a dog and a cat", 'source1']]
        random.seed(0)
        output_lines = replace_with_synonym(input_lines)
        self.assertEqual(output_lines[1][2],
                         'Once there was a dog and angstrom cat')

    def test_update_exclamation_mark(self):
        # create a list of tuples for testing
        my_list = [
            [1, 4, 'This is a sentence.', 'line1'],
            [1, 5, 'This is another sentence!', 'line2'],
            [1, 6, 'Yet another sentence.', 'line3']
        ]
        random.seed(0)
        # call the function
        output = update_exclamation_mark_end_sentence(my_list)
        print(output)
        # check that the length of the list hasn't changed
        self.assertEqual(len(output), 3)

        # check that the exclamation mark has been updated for a random sentence
        self.assertEqual(output[1][2], 'This is another sentence')
        print('Success: test_update_exclamation_mark')

    def test_update_exclamation_mark_random_pos(self):
        random.seed(1000)
        print('Running test: update_exclamation_mark_random_pos')
        # test with list of tuples containing a sentence without an exclamation mark
        my_list = [[1, 4, 'This is a sentence.', 'line1']]

        modified_sentence = update_exclamation_mark_random_pos(my_list)
        print(modified_sentence)
        self.assertIn('!', modified_sentence[0][2])

        # test with list of list containing a sentence with an exclamation mark
        my_list = [[1, 4, 'This is another ! sentence', 'line2']]
        modified_sentence = update_exclamation_mark_random_pos(my_list)
        print(modified_sentence)
        self.assertNotIn('!', modified_sentence)
        print('Success: test_update_exclamation_mark_random_pos')

    def test_swap_and(self):
        # create a list of tuples for testing
        my_list = [[1, 5, 'once there was a dog and a cat', 'line1']]
        modified_sentence = swap_and_words(my_list)
        self.assertEqual(
            modified_sentence[0][2], 'once there was a cat and a dog')
        print('Success: test_update_exclamation_mark')

    def test_swap_and_noun(self):
        # create a list of tuples for testing
        my_list = [[1, 5, 'once there was a funny and a cat', 'line1']]
        modified_sentence = swap_and_words(my_list, check_both_noun=True)
        self.assertEqual(
            modified_sentence[0][2], 'once there was a funny and a cat')

        my_list = [[1, 5, 'once there was funny cat and silly dog', 'line2']]
        modified_sentence = swap_and_words(my_list, check_both_noun=True)
        self.assertEqual(
            modified_sentence[0][2], 'once there was silly dog and funny cat')

        print('Success: test_update_exclamation_mark')

    def test_repair_a_an(self):
        my_list = [[1, 5, 'a astronaut walked into a bar', 'line1'],
                   [1, 6, 'a fox jumped over an lazy dog and an old tree', 'line2'],
                   [1, 7, 'an apple an day keeps a evil doctor away', 'line3']]

        expected_output = [[1, 5, 'an astronaut walked into a bar', 'line1'],
                           [1, 6, 'a fox jumped over a lazy dog and an old tree', 'line2'],
                           [1, 7, 'an apple a day keeps an evil doctor away', 'line3']]
        repaired_output = repair_a_an(my_list)
        self.assertEqual(expected_output, repaired_output)

    def test_match_by_level(self):
        '''
        http://www.speech.cs.cmu.edu/cgi-bin/cmudict
        :return:
        '''
        from agent_ls import match_by_level

        list2 = [('dissidence', ['D', 'IH1', 'S', 'AH0', 'D', 'AH0', 'N', 'S']),
                 ('dollar', ['D', 'AA1', 'L', 'ER0']),
                 ('blossom', ['B', 'L', 'AA1', 'S', 'AH0', 'M'])]

        result_lev_1 = match_by_level([('time', [' T', 'AY1', 'M'])],
                                      list2,
                                      level=1)
        self.assertEqual(['blossom'], result_lev_1)

        result_lev_2 = match_by_level([('collar', ['K', 'AA1', 'L', 'ER0'])],
                                      list2, level=2)
        self.assertEqual(['dollar'], result_lev_2)

        result_lev_3 = match_by_level([('persistence', ['P', 'ER0', 'S', 'IH1', 'S', 'T', 'AH0', 'N', 'S'])],
                                      list2, level=3)
        self.assertEqual(['dissidence'], result_lev_3)

    def test_rhyme_and_make_sense(self):
        from agent_ls import rhyme_and_make_sense
        last_word2 = 'man'

        # The following based on 1 level match cmudict entries
        rhymes = ['aachen', 'gentleman', 'serviceman',
                  'aaron', 'aaronson', 'aasen', 'man', 'human']
        expected_output = ['gentleman', 'human', 'serviceman']
        output = rhyme_and_make_sense(last_word2, rhymes)

        # Sort the output and expected output to make sure they are the same
        output.sort()
        self.assertEqual(output, expected_output)

    def test_noun_selector(self):
        from agent_ls import noun_selector
        my_list = [[1, 5, 'Once there was a dog and a cat', 'line1']]

        result = noun_selector(my_list[0][2])
        self.assertIn(result[1], ["dog", "cat"])

    def test_add_sensible_adjective(self):
        from agent_ls import add_sensible_adjective
        my_list = [[1, 5, 'Once there was a dog and a cat', 'line1'],
                   [10, 50, 'Who decided to have an extended natter', 'line50']]
        random.seed(100)
        result = add_sensible_adjective(my_list)
        self.assertEqual(result[0][2], 'Once there was a dog and a scared cat')


if __name__ == '__main__':
    unittest.main()
