import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer

STOP_WORDS = set(stopwords.words("english"))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def tokenize(text):
    return word_tokenize(text)

def normalize(tokens):
    return [token.lower() for token in tokens]

def remove_stopwords(tokens):
    #   .isalnum() removes punctuation-only tokens
    return [t for t in tokens if t not in STOP_WORDS and t.isalnum()]

def stem(tokens):
    return [stemmer.stem(t) for t in tokens]

def _get_wordnet_pos(treebank_tag: str):
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    if treebank_tag.startswith("V"):
        return wordnet.VERB
    if treebank_tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN

def lemmatize(tokens, use_pos=True):
    if not use_pos:
        return [lemmatizer.lemmatize(t) for t in tokens]

    tagged = nltk.pos_tag(tokens)
    return [lemmatizer.lemmatize(word, _get_wordnet_pos(tag)) for word, tag in tagged]


def process_text(text, use_stopwords=True, use_stemming=True, use_lemmatization=True, lemmatize_with_pos=True):
    tokens = tokenize(text)
    tokens = normalize(tokens)

    if use_stopwords:
        tokens = remove_stopwords(tokens)

    if use_stemming:
        tokens = stem(tokens)

    if use_lemmatization:
        tokens = lemmatize(tokens, use_pos= lemmatize_with_pos)

    return tokens
