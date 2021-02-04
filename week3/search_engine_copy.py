from sklearn.feature_extraction.text import CountVectorizer
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import numpy as np

toy_documents = ["This is a silly example",
                 "A better example",
                 "Nothing to see here",
                 "This is a great and long example"]

d = {"and": "&", "AND": "&",
     "or": "|", "OR": "|",
     "not": "1 -", "NOT": "1 -",
     "(": "(", ")": ")"}

with open('text_data.txt', encoding="utf8") as file:
    contents = file.read()

soup = BeautifulSoup(contents, 'html.parser')

articles = soup.find_all('article')
documents = [t.get_text().replace('\n', ' ') for t in articles]

cv = CountVectorizer(lowercase=True, binary=True, token_pattern=r"(?u)\b\w+\b")
sparse_matrix = cv.fit_transform(documents)
dense_matrix = sparse_matrix.todense()
td_matrix = dense_matrix.T  # .T transposes the matrix
terms = cv.get_feature_names()


def rewrite_token(t):
    if t not in terms and t not in d:
        return "np.zeros((1, len(documents)), dtype=int)"
    else:
        return d.get(t, 'sparse_td_matrix[t2i["{:s}"]].todense()'.format(t))


def rewrite_query(query):  # rewrite every token in the query
    return " ".join(rewrite_token(t) for t in query.split())


def test_query(query):
    print("Query: '" + query + "'")
    print("Rewritten:", rewrite_query(query))
    print("Matching:", eval(rewrite_query(query)))
    print()


t2i = cv.vocabulary_
sparse_td_matrix = sparse_matrix.T.tocsr()

# RELEVANCE RANKING SEARCH

gv = TfidfVectorizer(lowercase=True, sublinear_tf=True, use_idf=True, norm="l2")
g_matrix = gv.fit_transform(documents).T.tocsr()


def search_wikipedia(query_string):

    # Vectorize query string
    words = query_string.split()
    vocab = gv.get_feature_names()
    final_words = []
    for w in words:
        if w in vocab:
            final_words.append(w)
    if not final_words:
        print("No matches")
    else:
        new_query_string = " ".join(final_words)
        query_vec = gv.transform([ new_query_string ]).tocsc()

        # Cosine similarity
        hits = np.dot(query_vec, g_matrix)

        # Rank hits
        ranked_scores_and_doc_ids = \
            sorted(zip(np.array(hits[hits.nonzero()])[0], hits.nonzero()[1]),
                   reverse=True)

        # Output result
        print("Your query '{:s}' matches the following documents:".format(query_string))
        for i, (score, doc_idx) in enumerate(ranked_scores_and_doc_ids):
            print("Doc #{:d} (score: {:.4f}): {:s}".format(i, score, documents[doc_idx][:50]))
        print()


if __name__ == "__main__":
    print("Welcome to English Wikipedia search engine!")
    while True:
        engine = input("Choose search engine (0 for boolean, 1 for relevance): ")
        query = input("Add a query (press enter to quit): ").lower().strip()
        if query == "":
            break

        if engine == "0":
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
            except:  # (KeyError, SyntaxError):
                print('Bad query, could not perform a search.')
            print()
        elif engine == "1":
            search_wikipedia(query)
        else:
            print("Bad engine chosen, no search performed")
            continue

    # refactoring, error handling
