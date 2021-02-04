from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from bs4 import BeautifulSoup
import nltk
from nltk.stem.snowball import SnowballStemmer
import re
import numpy as np

# globals
toy_documents = ["This is a silly example",
                 "A better example",
                 "Nothing to see here",
                 "This is a great and long example",
                 "Raining mining housing adding nearest up"]


d = {"and": "&", "AND": "&",
     "or": "|", "OR": "|",
     "not": "1 -", "NOT": "1 -",
     "(": "(", ")": ")"}

documents = []
terms = []
sparse_td_matrix = []
t2i = []
gv = []
g_matrix = []
gv_stem = []
g_matrix_stem = []
stemmer = SnowballStemmer("english")


def tokenize(text):
    tokens = [word for word in nltk.word_tokenize(text) if len(word) > 1]
    stems = [stemmer.stem(item) for item in tokens]
    return stems


def initialize():
    # to save changes globally
    global documents
    global terms
    global sparse_td_matrix
    global t2i
    global gv
    global g_matrix
    global gv_stem
    global g_matrix_stem

    # ready documents
    with open('text_data.txt', encoding="utf8") as file:
        contents = file.read()

    soup = BeautifulSoup(contents, 'html.parser')
    articles = soup.find_all('article')
    documents = [t.get_text().replace('\n', ' ') for t in articles]

    # initialize boolean search tools
    cv = CountVectorizer(lowercase=True, binary=True,
                         token_pattern=r"(?u)\b\w+\b")
    sparse_matrix = cv.fit_transform(documents)
    dense_matrix = sparse_matrix.todense()
    td_matrix = dense_matrix.T
    terms = cv.get_feature_names()
    t2i = cv.vocabulary_
    sparse_td_matrix = sparse_matrix.T.tocsr()

    # initialize relevance search tools
    # no stemming
    gv = TfidfVectorizer(lowercase=True,
                         sublinear_tf=True, use_idf=True, norm="l2")
    g_matrix = gv.fit_transform(documents).T.tocsr()

    # stemming
    gv_stem = TfidfVectorizer(tokenizer=tokenize, lowercase=True,
                              sublinear_tf=True, use_idf=True, norm="l2")
    g_matrix_stem = gv_stem.fit_transform(documents).T.tocsr()


def rewrite_token(t):
    # for boolean search
    if t not in terms and t not in d:
        return "np.zeros((1, len(documents)), dtype=int)"
    else:
        return d.get(t, 'sparse_td_matrix[t2i["{:s}"]].todense()'.format(t))


def rewrite_query(query):
    # rewrite every token in the query for boolean search
    return " ".join(rewrite_token(t) for t in query.split())


def test_query(query):
    print("Query: '" + query + "'")
    print("Rewritten:", rewrite_query(query))
    print("Matching:", eval(rewrite_query(query)))
    print()


def boolean_search(query):
    try:
        if rewrite_query(query) is None:
            print("No matches.")
        else:
            hits_matrix = eval(rewrite_query(query))
            hits_list = list(hits_matrix.nonzero()[1])
            print('Results:')
            print("Matched", len(hits_list), "documents.")

            for doc_idx in hits_list[:10]:
                print(
                    f"Matching doc: [{doc_idx}] {documents[doc_idx][:50]}...")
    except:
        print('Bad query, could not perform a search.')
    print()


def relevance_search(query_string):
    # Vectorize query string
    words = query_string.split()

    if '"' in query_string:  # exact search  <"searchword">
        query_string = query_string.replace('"', '')
        words = query_string.split()

        vocab = gv.get_feature_names()
        final_words = [w for w in words if w in vocab]
        if not final_words:
            print("No matches")
            return
        new_query_string = " ".join(final_words)
        query_vec = gv.transform([new_query_string]).tocsc()
        # Cosine similarity
        hits = np.dot(query_vec, g_matrix)

    else:  # stemming can be used
        vocab = gv_stem.get_feature_names()
        final_words = [stemmer.stem(w)
                       for w in words if stemmer.stem(w) in vocab]
        if not final_words:
            print("No matches")
            return
        new_query_string = " ".join(final_words)
        query_vec = gv_stem.transform([new_query_string]).tocsc()
        # Cosine similarity
        hits = np.dot(query_vec, g_matrix_stem)

    # Rank hits
    ranked_scores_and_doc_ids = \
        sorted(zip(np.array(hits[hits.nonzero()])[0], hits.nonzero()[1]),
               reverse=True)

    # Output result
    print("Your query '{:s}' matches the following documents:".format(
        query_string))
    for i, (score, doc_idx) in enumerate(ranked_scores_and_doc_ids):
        print("Doc #{:d} (score: {:.4f}): {:s}".format(
            i, score, documents[doc_idx][:50]))
    print()


if __name__ == "__main__":
    initialize()
    print("Welcome to English Wikipedia search engine!")
    while True:
        engine = input(
            "\nChoose search engine (0 for boolean, 1 for relevance, enter to quit): ")
        if engine == "":
            break
        if engine == '0':
            while True:
                print('\n-BOOLEAN SEARCH-')
                print('(press enter to return to engine menu)')
                query = input(
                    ">Add a query: ").lower().strip()
                if query == "":
                    break
                boolean_search(query)
        elif engine == "1":
            while True:
                print('\n-RELEVANCE SEARCH-')
                print('(press enter to return to engine menu, use "" for exact match)')
                query = input(
                    ">Add a query: ").lower().strip()
                if query == "":
                    break
                relevance_search(query)
        else:
            print("No such engine.")

    print('Goodbye!')
