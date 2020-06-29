
import os
import pandas as pd
import numpy as np
import requests
import time
import re

# ---------------------------------------------------------------------
# Question #1
# ---------------------------------------------------------------------


def get_book(url):
    """
    get_book that takes in the url of a 'Plain Text UTF-8' book and 
    returns a string containing the contents of the book.

    The function should satisfy the following conditions:
        - The contents of the book consist of everything between 
        Project Gutenberg's START and END comments.
        - The contents will include title/author/table of contents.
        - You should also transform any Windows new-lines (\r\n) with 
        standard new-lines (\n).
        - If the function is called twice in succession, it should not 
        violate the robots.txt policy.

    :Example: (note '\n' don't need to be escaped in notebooks!)
    >>> url = 'http://www.gutenberg.org/files/57988/57988-0.txt'
    >>> book_string = get_book(url)
    >>> book_string[:20] == '\\n\\n\\n\\n\\nProduced by Chu'
    True
    """

    resp = requests.get(url)
    
    text_split = [resp.text[:5000], resp.text[1000:-20000], resp.text[-25000:]]
    text1 = re.sub('[\\r]' , '', text_split[0])
    text1 = re.sub('(.|\\n)*\*\*\*( START OF.+)\*\*\*' , '', text1)

    text2 = re.sub('[\\r]' , '', text_split[1])

    text3 = re.sub('[\\r]' , '', text_split[2])
    text3 = re.sub('(\*\*\* END OF)(.|\\n)*', '', text3)
    
    return text1 + text2 + text3

    
# ---------------------------------------------------------------------
# Question #2
# ---------------------------------------------------------------------


def tokenize(book_string):
    """
    tokenize takes in book_string and outputs a list of tokens 
    satisfying the following conditions:
        - The start of any paragraph should be represented in the 
        list with the single character \x02 (standing for START).
        - The end of any paragraph should be represented in the list 
        with the single character \x03 (standing for STOP).
        - Tokens in the sequence of words are split 
        apart at 'word boundaries' (see the regex lecture).
        - Tokens should include no whitespace.

    :Example:
    >>> shakespeare_fp = os.path.join('data', 'shakespeare.txt')
    >>> shakespeare = open(shakespeare_fp, encoding='utf-8').read()
    >>> tokens = tokenize(shakespeare)
    >>> tokens[0] == '\x02'
    True
    >>> (tokens[1] == '\x03') and (tokens[-1] == '\x03')
    True
    >>> tokens[10] == 'Shakespeare'
    True
    """
    book_string = '\x02 ' + book_string + ' \x03'
    test = re.sub('\n\n', ' \x03 \x02 ', book_string)
    test = re.sub('\n', ' ', test)

    split_by = r'\w+|[^\w\s]+'

    return re.findall(split_by , test)


# ---------------------------------------------------------------------
# Question #3
# ---------------------------------------------------------------------


class UniformLM(object):
    """
    Uniform Language Model class.
    """

    def __init__(self, tokens):
        """
        Initializes a Uniform languange model using a
        list of tokens. It trains the language model
        using `train` and saves it to an attribute
        self.mdl.
        """
        self.mdl = self.train(tokens)
        
    def train(self, tokens):
        """
        Trains a uniform language model given a list of tokens.
        The output is a series indexed on distinct tokens, and
        values giving the (uniform) probability of a token occuring
        in the language.

        :Example:
        >>> tokens = tuple('one one two three one two four'.split())
        >>> unif = UniformLM(tokens)
        >>> isinstance(unif.mdl, pd.Series)
        True
        >>> set(unif.mdl.index) == set('one two three four'.split())
        True
        >>> (unif.mdl == 0.25).all()
        True
        """

        uniform_prob = 1 / pd.Series(tokens).nunique()
        index = pd.Series(tokens).unique()
        
        return pd.Series(uniform_prob, index=index)

    def probability(self, words):
        """
        probability gives the probabiliy a sequence of words
        appears under the language model.
        :param: words: a tuple of tokens
        :returns: the probability `words` appears under the language
        model.

        :Example:
        >>> tokens = tuple('one one two three one two four'.split())
        >>> unif = UniformLM(tokens)
        >>> unif.probability(('five',))
        0
        >>> unif.probability(('one', 'two')) == 0.0625
        True
        """
        prob = 1
        for token in words:
            if token not in self.mdl.index:
                prob = 0
                break
            prob *= self.mdl[token]
        return prob

    def sample(self, M):
        """
        sample selects tokens from the language model of length M, returning
        a string of tokens.

        :Example:
        >>> tokens = tuple('one one two three one two four'.split())
        >>> unif = UniformLM(tokens)
        >>> samp = unif.sample(1000)
        >>> isinstance(samp, str)
        True
        >>> len(samp.split()) == 1000
        True
        >>> s = pd.Series(samp.split()).value_counts(normalize=True)
        >>> np.isclose(s, 0.25, atol=0.05).all()
        True
        """
        tokens = ''

        for _ in np.arange(M):
            tokens += np.random.choice(self.mdl.index, p = self.mdl.values) + ' '
        return tokens.strip()

            
# ---------------------------------------------------------------------
# Question #4
# ---------------------------------------------------------------------


class UnigramLM(object):
    
    def __init__(self, tokens):
        """
        Initializes a Unigram languange model using a
        list of tokens. It trains the language model
        using `train` and saves it to an attribute
        self.mdl.
        """
        self.mdl = self.train(tokens)
    
    def train(self, tokens):
        """
        Trains a unigram language model given a list of tokens.
        The output is a series indexed on distinct tokens, and
        values giving the probability of a token occuring
        in the language.

        :Example:
        >>> tokens = tuple('one one two three one two four'.split())
        >>> unig = UnigramLM(tokens)
        >>> isinstance(unig.mdl, pd.Series)
        True
        >>> set(unig.mdl.index) == set('one two three four'.split())
        True
        >>> unig.mdl.loc['one'] == 3 / 7
        True
        """

        return pd.Series(tokens).value_counts(normalize = True)
    
    def probability(self, words):
        """
        probability gives the probabiliy a sequence of words
        appears under the language model.
        :param: words: a tuple of tokens
        :returns: the probability `words` appears under the language
        model.

        :Example:
        >>> tokens = tuple('one one two three one two four'.split())
        >>> unig = UnigramLM(tokens)
        >>> unig.probability(('five',))
        0
        >>> p = unig.probability(('one', 'two'))
        >>> np.isclose(p, 0.12244897959, atol=0.0001)
        True
        """
        prob = 1
        for token in words:
            if token not in self.mdl.index:
                prob = 0
                break
            prob *= self.mdl[token]
        return prob
        
    def sample(self, M):
        """
        sample selects tokens from the language model of length M, returning
        a string of tokens.

        >>> tokens = tuple('one one two three one two four'.split())
        >>> unig = UnigramLM(tokens)
        >>> samp = unig.sample(1000)
        >>> isinstance(samp, str)
        True
        >>> len(samp.split()) == 1000
        True
        >>> s = pd.Series(samp.split()).value_counts(normalize=True).loc['one']
        >>> np.isclose(s, 0.41, atol=0.05).all()
        True
        """

        tokens = ''

        for _ in np.arange(M):
            tokens += np.random.choice(self.mdl.index, p = self.mdl.values) + ' '
        return tokens.strip()
        
    
# ---------------------------------------------------------------------
# Question #5,6,7,8
# ---------------------------------------------------------------------

class NGramLM(object):
    
    def __init__(self, N, tokens):
        """
        Initializes a N-gram languange model using a
        list of tokens. It trains the language model
        using `train` and saves it to an attribute
        self.mdl.
        """

        self.N = N
        ngrams = self.create_ngrams(tokens)

        self.ngrams = ngrams
        self.mdl = self.train(ngrams)

    def create_ngrams(self, tokens):
        """
        create_ngrams takes in a list of tokens and returns a list of N-grams. 
        The START/STOP tokens in the N-grams should be handled as 
        explained in the notebook.

        :Example:
        >>> tokens = tuple('\x02 one two three one four \x03'.split())
        >>> bigrams = NGramLM(2, [])
        >>> out = bigrams.create_ngrams(tokens)
        >>> isinstance(out[0], tuple)
        True
        >>> out[0]
        ('\\x02', '\\x02')
        >>> out[2]
        ('one', 'two')
        """
        ngrams = []
        num_preceding = 0
        start_dist = 1
        for i in np.arange(len(tokens)):

            if tokens[i] == '\x02':
                gram = (tokens[i],) * self.N
                ngrams.append(gram)
                num_preceding = 0
                start_dist = 1
                continue        

            if tokens[i] == '\x03':
                k = 0
                while k < self.N:
                    gram = ()
                    for j in np.arange(self.N - 1 - k, 0, -1):
                        gram += (tokens[i - j],)
                    gram += ('\x03',) * (k + 1)
                    ngrams.append(gram)
                    k += 1
            elif num_preceding < self.N - 1:
                gram = (tokens[i - start_dist],) * (self.N - 1 - num_preceding)
                for j in np.arange(num_preceding, -1, -1):
                    gram += (tokens[i - j],)
                ngrams.append(gram)
                    
            else:
                gram = ()
                for j in np.arange(self.N - 1, -1, -1):
                    gram += (tokens[i - j],)
                ngrams.append(gram)

            num_preceding += 1
            start_dist += 1

        return ngrams

    
    def train(self, ngrams):
        """
        Trains a n-gram language model given a list of tokens.
        The output is a series indexed on distinct tokens, and
        values giving the probability of a token occuring
        in the language.

        :Example:
        >>> tokens = tuple('\x02 one two three one four \x03'.split())
        >>> bigrams = NGramLM(2, tokens)
        >>> set(bigrams.mdl.columns) == set('ngram n1gram prob'.split())
        True
        >>> bigrams.mdl.shape == (8, 3)
        True
        >>> bigrams.mdl['prob'].min() == 0.5
        True
        """
      
        # Create ngram counts C(w_1, ..., w_n)
        ngram = pd.Series(self.ngrams)
        ngram_unigram = UnigramLM(ngram)
        
        # Create n-1 gram counts C(w_1, ..., w_(n-1))
        n1gram = ngram.apply(lambda x: x[:-1])
        n1gram_unigram = UnigramLM(n1gram)
        
        # Create the conditional probabilities
        prob = ngram.apply(lambda x: ngram_unigram.mdl[x]) / n1gram.apply(lambda y: n1gram_unigram.mdl[y])
        
        # Put it all together
        
        return pd.DataFrame({'ngram': ngram, 'n1gram': n1gram, 'prob': prob})
    
    def probability(self, words):
        """
        probability gives the probabiliy a sequence of words
        appears under the language model.
        :param: words: a tuple of tokens
        :returns: the probability `words` appears under the language
        model.

        :Example:
        >>> tokens = tuple('\x02 one two three one four \x03'.split())
        >>> bigrams = NGramLM(2, tokens)
        >>> bigrams.probability('one two three'.split()) == 0.5
        True
        >>> bigrams.probability('one two five'.split()) == 0
        True
        """
        prob = 1
        words_ngram = NGramLM(self.N, words).ngrams[1:]
        for ngram in words_ngram:
            if ngram not in self.mdl['ngram'].to_list():
                prob = 0
                break
            prob *= self.mdl.loc[self.mdl['ngram'] == ngram, 'prob'].max()

        return prob

    def sample(self, length):
        """
        sample selects tokens from the language model of length M, returning
        a string of tokens.

        :Example:
        >>> tokens = tuple('\x02 one two three one four \x03'.split())
        >>> bigrams = NGramLM(2, tokens)
        >>> samp = bigrams.sample(3)
        >>> len(samp.split()) == 4  # don't count the initial N-1 START token(s).
        True
        >>> samp[:2] == '\\x02 '
        True
        >>> set(samp.split()) <= {'\\x02', '\\x03', 'one', 'two', 'three', 'four'}
        True
        """
        
        # Use a helper function to generate sample tokens of length `length`
        def pick_sample(n1gram, M, num_picked):
        
            if M > 0 and num_picked > self.mdl.shape[0] - (self.N-1)*2:
                return '\x03 ' * M

            if n1gram[-1] == '\x03':
                if M > 1:
                    return '\x02 ' + pick_sample(('\x02',)*(self.N-1), M-1, num_picked + 1)
                else:
                    return ''

            temp = self.mdl.loc[self.mdl['n1gram'] == n1gram].drop_duplicates()
            choices = temp['ngram'].apply(lambda x: x[-1]).tolist()
            p = temp['prob'].tolist()
            if M == 1:
                return np.random.choice(choices, p = p)

            else:
                w = np.random.choice(choices, p = p)
                n1gram = (n1gram + (w,))[1:]
                return w + ' ' + pick_sample(n1gram, M-1, num_picked+1)
        
        # Tranform the tokens to strings
        sampled_tokens = '\x02 '
        sampled_tokens += pick_sample(('\x02',)*(self.N-1), length, 0)
        return sampled_tokens.strip()

# ---------------------------------------------------------------------
# Question # 9
# ---------------------------------------------------------------------
    

def predict_next(lm, tokens):
    """
    predict_next that takes in an instantiated NGramLM object 
    and a list of tokens, and predicts the most likely token to 
    follow the list of tokens in the input (according to NGramLM).

    :Example:
    >>> tokens = tuple('\x02 one two three one four \x03'.split())
    >>> bigrams = NGramLM(2, tokens)
    >>> predict_next(bigrams, ('one', 'two')) == 'three'
    True
    """
    
    t = NGramLM(lm.N - 1, tokens)
    f = lm.mdl.loc[lm.mdl['n1gram'] == t.ngrams[-1]]
    
    return f.loc[f['prob'].idxmax()].ngram[-1]


# ---------------------------------------------------------------------
# Question # 10
# ---------------------------------------------------------------------
    

def evaluate_models(tokens):
    """
    Extra-Credit (see notebook)
    """

    return ...

# ---------------------------------------------------------------------
# DO NOT TOUCH BELOW THIS LINE
# IT'S FOR YOUR OWN BENEFIT!
# ---------------------------------------------------------------------


# Graded functions names! DO NOT CHANGE!
# This dictionary provides your doctests with
# a check that all of the questions being graded
# exist in your code!

GRADED_FUNCTIONS = {
    'q01': ['get_book'],
    'q02': ['tokenize'],
    'q03': ['UniformLM'],
    'q04': ['UnigramLM'],
    'q05': ['NGramLM'],
    'q09': ['predict_next'],
    'q10': ['evaluate_models']
}


def check_for_graded_elements():
    """
    >>> check_for_graded_elements()
    True
    """
    
    for q, elts in GRADED_FUNCTIONS.items():
        for elt in elts:
            if elt not in globals():
                stmt = "YOU CHANGED A QUESTION THAT SHOULDN'T CHANGE! \
                In %s, part %s is missing" %(q, elt)
                raise Exception(stmt)

    return True
