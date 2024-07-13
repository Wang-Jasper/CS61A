"""Typing test implementation"""
import math

from utils import (
    lower,
    split,
    remove_punctuation,
    lines_from_file,
    count,
    deep_convert_to_tuple,
)
from ucb import main, interact, trace
from datetime import datetime


###########
# Phase 1 #
###########


def pick(paragraphs, select, k):
    """Return the Kth paragraph from PARAGRAPHS for which the SELECT returns True.
    If there are fewer than K such paragraphs, return an empty string.

    Arguments:
        paragraphs: a list of strings representing paragraphs
        select: a function that returns True for paragraphs that meet its criteria
        k: an integer

    >>> ps = ['hi', 'how are you', 'fine']
    >>> s = lambda p: len(p) <= 4
    >>> pick(ps, s, 0)
    'hi'
    >>> pick(ps, s, 1)
    'fine'
    >>> pick(ps, s, 2)
    ''
    """
    # BEGIN PROBLEM 1
    "*** YOUR CODE HERE ***"
    for i in range(len(paragraphs)):
        if select(paragraphs[i]):
            if k == 0:
                return paragraphs[i]
            else:
                k -= 1
    return ''
    # END PROBLEM 1


def about(subject):
    """Return a function that takes in a paragraph and returns whether
    that paragraph contains one of the words in SUBJECT.

    Arguments:
        subject: a list of words related to a subject

    >>> about_dogs = about(['dog', 'dogs', 'pup', 'puppy'])
    >>> pick(['Cute Dog!', 'That is a cat.', 'Nice pup!'], about_dogs, 0)
    'Cute Dog!'
    >>> pick(['Cute Dog!', 'That is a cat.', 'Nice pup.'], about_dogs, 1)
    'Nice pup.'
    """
    assert all([lower(x) == x for x in subject]), "subjects should be lowercase."

    # BEGIN PROBLEM 2
    "*** YOUR CODE HERE ***"
    def check_words(paragraph):
        for i in range(len(subject)):
            if subject[i].lower() in remove_punctuation(paragraph.lower()).split():
                return True
        return False
    return check_words
    # END PROBLEM 2


def accuracy(typed, source):
    """Return the accuracy (percentage of words typed correctly) of TYPED
    compared to the corresponding words in SOURCE.

    Arguments:
        typed: a string that may contain typos
        source: a model string without errors

    >>> accuracy('Cute Dog!', 'Cute Dog.')
    50.0
    >>> accuracy('A Cute Dog!', 'Cute Dog.')
    0.0
    >>> accuracy('cute Dog.', 'Cute Dog.')
    50.0
    >>> accuracy('Cute Dog. I say!', 'Cute Dog.')
    50.0
    >>> accuracy('Cute', 'Cute Dog.')
    100.0
    >>> accuracy('', 'Cute Dog.')
    0.0
    >>> accuracy('', '')
    100.0
    """
    typed_words = split(typed)
    source_words = split(source)
    # BEGIN PROBLEM 3
    "*** YOUR CODE HERE ***"
    if len(source_words) == len(typed_words) == 0:
        return 100.0
    correct_words = 0
    for i in range(min(len(typed_words), len(source_words))):
        if typed_words[i] == source_words[i]:
            correct_words += 1
    return correct_words / len(typed_words) * 100.0 if len(typed_words) != 0 else 0.0
    # END PROBLEM 3


def wpm(typed, elapsed):
    """Return the words-per-minute (WPM) of the TYPED string.

    Arguments:
        typed: an entered string
        elapsed: an amount of time in seconds

    >>> wpm('hello friend hello buddy hello', 15)
    24.0
    >>> wpm('0123456789',60)
    2.0
    """
    assert elapsed > 0, "Elapsed time must be positive"
    # BEGIN PROBLEM 4
    "*** YOUR CODE HERE ***"
    return len(typed) * 12.0 / elapsed
    # END PROBLEM 4


################
# Phase 4 (EC) #
################


def memo(f):
    """A general memoization decorator."""
    cache = {}

    def memoized(*args):
        immutable_args = deep_convert_to_tuple(args)  # convert *args into a tuple representation
        if immutable_args not in cache:
            result = f(*immutable_args)
            cache[immutable_args] = result
            return result
        return cache[immutable_args]

    return memoized


def memo_diff(diff_function):
    """A memoization function."""
    cache = {}

    def memoized(typed, source, limit):
        # BEGIN PROBLEM EC
        "*** YOUR CODE HERE ***"
        # END PROBLEM EC

    return memoized


###########
# Phase 2 #
###########


def autocorrect(typed_word, word_list, diff_function, limit):
    """Returns the element of WORD_LIST that has the smallest difference
    from TYPED_WORD based on DIFF_FUNCTION. If multiple words are tied for the smallest difference,
    return the one that appears closest to the front of WORD_LIST. If the
    difference is greater than LIMIT, return TYPED_WORD instead.

    Arguments:
        typed_word: a string representing a word that may contain typos
        word_list: a list of strings representing source words
        diff_function: a function quantifying the difference between two words
        limit: a number

    >>> ten_diff = lambda w1, w2, limit: 10 # Always returns 10
    >>> autocorrect("hwllo", ["butter", "hello", "potato"], ten_diff, 20)
    'butter'
    >>> first_diff = lambda w1, w2, limit: (1 if w1[0] != w2[0] else 0) # Checks for matching first char
    >>> autocorrect("tosting", ["testing", "asking", "fasting"], first_diff, 10)
    'testing'
    """
    # BEGIN PROBLEM 5
    "*** YOUR CODE HERE ***"
    if typed_word in word_list:
        return typed_word
    min_diff = limit + 1
    for i in range(len(word_list)):
        if diff_function(typed_word, word_list[i], limit) < min_diff:
            min_diff = diff_function(typed_word, word_list[i], limit)
            min_diff_word = word_list[i]
    if min_diff != limit + 1 and min_diff <= limit:
        return min_diff_word
    else:
        return typed_word
    # END PROBLEM 5


def furry_fixes(typed, source, limit):
    """A diff function for autocorrect that determines how many letters
    in TYPED need to be substituted to create SOURCE, then adds the difference in
    their lengths and returns the result.

    Arguments:
        typed: a starting word
        source: a string representing a desired goal word
        limit: a number representing an upper bound on the number of chars that must change

    >>> big_limit = 10
    >>> furry_fixes("nice", "rice", big_limit)    # Substitute: n -> r
    1
    >>> furry_fixes("range", "rungs", big_limit)  # Substitute: a -> u, e -> s
    2
    >>> furry_fixes("pill", "pillage", big_limit) # Don't substitute anything, length difference of 3.
    3
    >>> furry_fixes("roses", "arose", big_limit)  # Substitute: r -> a, o -> r, s -> o, e -> s, s -> e
    5
    >>> furry_fixes("rose", "hello", big_limit)   # Substitute: r->h, o->e, s->l, e->l, length difference of 1.
    5
    """
    # BEGIN PROBLEM 6
    # assert False, 'Remove this line'
    "*** YOUR CODE HERE ***"
    if limit <= 0:
        return int(not(typed == source))
    if typed == "" or source == "":
        return len(typed) + len(source)
    if(typed[0] == source[0]):
        return furry_fixes(typed[1:], source[1:], limit)
    else:
        return 1 + furry_fixes(typed[1:], source[1:], limit - 1)
    # END PROBLEM 6


def minimum_mewtations(typed, source, limit):
    """A diff function for autocorrect that computes the edit distance from TYPED to SOURCE.
    This function takes in a string TYPED, a string SOURCE, and a number LIMIT.

    Arguments:
        typed: a starting word
        source: a string representing a desired goal word
        limit: a number representing an upper bound on the number of edits

    >>> big_limit = 10
    >>> minimum_mewtations("cats", "scat", big_limit)       # cats -> scats -> scat
    2
    >>> minimum_mewtations("purng", "purring", big_limit)   # purng -> purrng -> purring
    2
    >>> minimum_mewtations("ckiteus", "kittens", big_limit) # ckiteus -> kiteus -> kitteus -> kittens
    3
    """
    # assert False, 'Remove this line'
    # BEGIN
    # add; remove; substitute
    # Trying to sense the operation each time but turns out just doing it violently:D
    # also learned the dp from chatgpt
    "*** YOUR CODE HERE ***"
    '''
    if typed == source:
        return 0
    if typed == "" or source == "":
        return len(typed) + len(source)
    if limit <= 0:
        return 1
    if typed[0] == source[0]:
        return minimum_mewtations(typed[1:], source[1:], limit)

    add = 1 + minimum_mewtations(typed, source[1:], limit - 1)
    remove = 1 + minimum_mewtations(typed[1:], source, limit - 1)
    substitute = 1 + minimum_mewtations(typed[1:], source[1:], limit - 1)
    return min(add, min(remove, substitute))
    '''
    # Create a table to store the minimum number of edits
    dp = [[0] * (len(source) + 1) for _ in range(len(typed) + 1)]

    # Initialize the base cases
    for i in range(len(typed) + 1):
        dp[i][0] = i  # Deleting all characters from `typed`
    for j in range(len(source) + 1):
        dp[0][j] = j  # Adding all characters from `source`

    # Fill the table using the recursive relationships
    for i in range(1, len(typed) + 1):
        for j in range(1, len(source) + 1):
            if typed[i - 1] == source[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]  # Characters match, no edit needed
            else:
                # Minimum of substitute, add, or remove
                dp[i][j] = min(
                    dp[i - 1][j - 1] + 1,  # Substitute (the minimum operation of previous character + 1)
                    dp[i][j - 1] + 1,  # Add (the minimum operation of previous character using typed + 1)
                    dp[i - 1][j] + 1  # Remove (the minimum operation of previous existing source character + 1)
                )

    # The value at dp[len(typed)][len(source)] is the answer
    result = dp[len(typed)][len(source)]

    if result > limit:
        return limit + 1
    return result

# Ignore the line below
minimum_mewtations = count(minimum_mewtations)


key_board = [
    ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
    ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', ','],
    ['Z', 'X', 'C', 'V', 'B', 'N', 'M', '<', '>', '?']
]

def final_diff(typed, source, limit):
    """A diff function that takes in a string TYPED, a string SOURCE, and a number LIMIT.
    If you implement this function, it will be used."""
    # assert False, "Remove this line to use your final_diff function."

    # switch adjacent characters, seen as the same
    # rest delete, add, switch the same with minimum_mewtations()
    def get_index(c):
        for index in range(len(key_board)):
            if c.upper() in key_board[index]:
                return index, key_board[index].index(c.upper())
            return 100, 100


    if typed == source:
        return 0
    if typed == "" or source == "":
        return len(typed) + len(source)
    if limit <= 0:
        return limit + 1

    # Create a table to store the minimum number of edits
    dp = [[0] * (len(source) + 1) for _ in range(len(typed) + 1)]

    for i in range(len(typed) + 1):
        dp[i][0] = i  # Deleting all characters from `typed`
    for j in range(len(source) + 1):
        dp[0][j] = j  # Adding all characters from `source`

    for i in range(1, len(typed) + 1):
        for j in range(1, len(source) + 1):
            # adjacent keyboard distance seen as one character
            if math.dist(get_index(typed[i - 1]), get_index(source[j - 1])) < 2 and get_index(typed[i - 1])[0] ==  get_index(source[j - 1])[0]:
               dp[i][j] = dp[i - 1][j - 1]
           # if typed[i - 1] == source[j - 1]:
           #     dp[i][j] = dp[i - 1][j - 1]  # Characters match, no edit needed
            else:
                dp[i][j] = min(
                    dp[i - 1][j - 1] + 1,  # Substitute (the minimum operation of previous character + 1)
                    dp[i][j - 1] + 1,  # Add (the minimum operation of previous character using typed + 1)
                    dp[i - 1][j] + 1  # Remove (the minimum operation of previous existing source character + 1)
                )

    result = dp[len(typed)][len(source)]

    if result > limit:
        return limit + 1
    return result


FINAL_DIFF_LIMIT = 6  # REPLACE THIS WITH YOUR LIMIT


###########
# Phase 3 #
###########


def report_progress(typed, source, user_id, upload):
    """Upload a report of your id and progress so far to the multiplayer server.
    Returns the progress so far.

    Arguments:
        typed: a list of the words typed so far
        source: a list of the words in the typing source
        user_id: a number representing the id of the current user
        upload: a function used to upload progress to the multiplayer server

    >>> print_progress = lambda d: print('ID:', d['id'], 'Progress:', d['progress'])
    >>> # The above function displays progress in the format ID: __, Progress: __
    >>> print_progress({'id': 1, 'progress': 0.6})
    ID: 1 Progress: 0.6
    >>> typed = ['how', 'are', 'you']
    >>> source = ['how', 'are', 'you', 'doing', 'today']
    >>> report_progress(typed, source, 2, print_progress)
    ID: 2 Progress: 0.6
    0.6
    >>> report_progress(['how', 'aree'], source, 3, print_progress)
    ID: 3 Progress: 0.2
    0.2
    """
    # BEGIN PROBLEM 8
    "*** YOUR CODE HERE ***"
    progress_num = 0
    for i in range(len(typed)):
        if typed[i] == source[i]:
            progress_num += 1
        else:
            break
    progress_num = progress_num / len(source)
    upload({'id': user_id, 'progress': progress_num})

    return progress_num
    # END PROBLEM 8


def time_per_word(words, timestamps_per_player):
    """Return two values: the list of words that the players are typing and
    a list of lists that stores the durations it took each player to type each word.

    Arguments:
        words: a list of words, in the order they are typed.
        TIMESTAMPS_PER_PLAYER: A list of lists of timestamps including the time
                          the player started typing, followed by the time
                          the player finished typing each word.


    >>> p = [[75, 81, 84, 90, 92], [19, 29, 35, 36, 38]]
    >>> words, times = time_per_word(['collar', 'plush', 'blush', 'repute'], p)
    >>> words
    ['collar', 'plush', 'blush', 'repute']
    >>> times
    [[6, 3, 6, 2], [10, 6, 1, 2]]
    """
    # BEGIN PROBLEM 9
    "*** YOUR CODE HERE ***"
    times_list = []
    for i in range(len(timestamps_per_player)):
        person_times_list = []
        for j in range(1, len(timestamps_per_player[i])):
            person_times_list.append(timestamps_per_player[i][j] - timestamps_per_player[i][j - 1])
        times_list.append(person_times_list)

    return words, times_list
    # END PROBLEM 9


def time_per_word_match(words, timestamps_per_player):
    """Return a match object containing the words typed and the time it took each player to type each word.

    Arguments:
        words: a list of words, in the order they are typed.
        timestamps_per_player: A list of lists of timestamps including the time
                          the player started typing, followed by the time
                          the player finished typing each word.

    >>> p = [[75, 81, 84, 90, 92], [19, 29, 35, 36, 38]]
    >>> match_object = time_per_word_match(['collar', 'plush', 'blush', 'repute'], p)
    >>> get_all_words(match_object)    # Notice how we now use the selector functions to access the data
    ['collar', 'plush', 'blush', 'repute']
    >>> get_all_times(match_object)
    [[6, 3, 6, 2], [10, 6, 1, 2]]
    """
    # BEGIN PROBLEM 10
    "*** YOUR CODE HERE ***"
    words, times = time_per_word(words, timestamps_per_player)
    return match(words, times)
    # END PROBLEM 10


def fastest_words(match_object):
    """Return a list of lists indicating which words each player typed the fastest.

    Arguments:
        match_object: a match data abstraction created by the match constructor

    >>> p0 = [5, 1, 3]
    >>> p1 = [4, 1, 6]
    >>> fastest_words(match(['Just', 'have', 'fun'], [p0, p1]))
    [['have', 'fun'], ['Just']]
    >>> p0  # input lists should not be mutated
    [5, 1, 3]
    >>> p1
    [4, 1, 6]
    """
    player_indices = range(len(get_all_times(match_object)))  # contains an *index* for each player
    word_indices = range(len(get_all_words(match_object)))  # contains an *index* for each word
    # BEGIN PROBLEM 11
    "*** YOUR CODE HERE ***"
    fastest_list = [[] for _ in player_indices]
    for i in word_indices:
        times = [get_time(match_object, p, i) for p in player_indices]
        fastest_list[times.index(min(times))].append(get_word(match_object, i))
    return fastest_list
    # END PROBLEM 11


def match(words, times):
    """Creates a data abstraction containing all words typed and their times.

    Arguments:
        words: A list of strings, each string representing a word typed.
        times: A list of lists for how long it took for each player to type
            each word.
            times[i][j] = time it took for player i to type words[j].

    Example input:
        words: ['Hello', 'world']
        times: [[5, 1], [4, 2]]
    """
    assert all([type(w) == str for w in words]), "words should be a list of strings"
    assert all([type(t) == list for t in times]), "times should be a list of lists"
    assert all([isinstance(i, (int, float)) for t in times for i in t]), "times lists should contain numbers"
    assert all([len(t) == len(words) for t in times]), "There should be one word per time."
    return {"words": words, "times": times}


def get_word(match, word_index):
    """A utility function that gets the word with index word_index"""
    assert (0 <= word_index < len(get_all_words(match))), "word_index out of range of words"
    return get_all_words(match)[word_index]


def get_time(match, player_num, word_index):
    """A utility function for the time it took player_num to type the word at word_index"""
    assert word_index < len(get_all_words(match)), "word_index out of range of words"
    assert player_num < len(get_all_times(match)), "player_num out of range of players"
    return get_all_times(match)[player_num][word_index]


def get_all_words(match):
    """A selector function for all the words in the match"""
    return match["words"]


def get_all_times(match):
    """A selector function for all typing times for all players"""
    return match["times"]


def match_string(match):
    """A helper function that takes in a match data abstraction and returns a string representation of it"""
    return f"match({get_all_words(match)}, {get_all_times(match)})"


enable_multiplayer = False  # Change to True when you're ready to race.

##########################
# Command Line Interface #
##########################


def run_typing_test(topics):
    """Measure typing speed and accuracy on the command line."""
    paragraphs = lines_from_file("data/sample_paragraphs.txt")
    select = lambda p: True
    if topics:
        select = about(topics)
    i = 0
    while True:
        source = pick(paragraphs, select, i)
        if not source:
            print("No more paragraphs about", topics, "are available.")
            return
        print("Type the following paragraph and then press enter/return.")
        print("If you only type part of it, you will be scored only on that part.\n")
        print(source)
        print()

        start = datetime.now()
        typed = input()
        if not typed:
            print("Goodbye.")
            return
        print()

        elapsed = (datetime.now() - start).total_seconds()
        print("Nice work!")
        print("Words per minute:", wpm(typed, elapsed))
        print("Accuracy:        ", accuracy(typed, source))

        print("\nPress enter/return for the next paragraph or type q to quit.")
        if input().strip() == "q":
            return
        i += 1


@main
def run(*args):
    """Read in the command-line argument and calls corresponding functions."""
    import argparse

    parser = argparse.ArgumentParser(description="Typing Test")
    parser.add_argument("topic", help="Topic word", nargs="*")
    parser.add_argument("-t", help="Run typing test", action="store_true")

    args = parser.parse_args()
    if args.t:
        run_typing_test(args.topic)