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
doc_names = []
terms = []
sparse_td_matrix = []
t2i = []
gv = TfidfVectorizer()
g_matrix = []
gv_stem = TfidfVectorizer()
g_matrix_stem = []
stemmer = SnowballStemmer("english")


def tokenize(text):
    tokens = [word for word in nltk.word_tokenize(text) if len(word) > 1]
    stems = [stemmer.stem(item) for item in tokens]
    return stems


def initialize():
    # to save changes globally
    global documents
    global doc_names
    global terms
    global sparse_td_matrix
    global t2i
    global gv_stem
    global g_matrix_stem

    # ready documents
    with open('static/text_data.txt', encoding="utf8") as file:
        contents = file.read()

    soup = BeautifulSoup(contents, 'html.parser')
    articles = soup.find_all('article')
    documents = [t.get_text().replace('\n', ' ') for t in articles]

    # Store document names to list:
    doc_names = [n.get('name') for n in articles]

    # initialize boolean search tools
    cv = CountVectorizer(lowercase=True, binary=True,
                         token_pattern=r"(?u)\b\w+\b")
    sparse_matrix = cv.fit_transform(documents)
    dense_matrix = sparse_matrix.todense()
    td_matrix = dense_matrix.T
    terms = cv.get_feature_names()
    t2i = cv.vocabulary_
    sparse_td_matrix = sparse_matrix.T.tocsr()

    # initialize relevance search tools for stemming
    gv_stem = TfidfVectorizer(tokenizer=tokenize, lowercase=True,
                              sublinear_tf=True, use_idf=True, norm="l2")
    g_matrix_stem = gv_stem.fit_transform(documents).T.tocsr()


def init_exact_search(n):
    # Exact search tools are initialized independently, as number of query words is needed and that is not known
    # until later. This initialization works for both single-word and multi-word searches.
    global gv
    global g_matrix
    gv = TfidfVectorizer(lowercase=True,
                         sublinear_tf=True, use_idf=True, norm="l2", ngram_range=(n, n))
    g_matrix = gv.fit_transform(documents).T.tocsr()
    return gv, g_matrix


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
            # print('Results:')
            # print("Matched", len(hits_list), "documents.")
            matches = []
            for doc_idx in hits_list:
                matches.append(
                    {"name": doc_names[doc_idx], "content": documents[doc_idx], "id": doc_idx})
                # print(
                #    f"Matching doc: [{doc_idx}] {documents[doc_idx][:50]}...")
            return matches
    except:
        print('Bad query, could not perform a search.')
    return []


def match_stems(words):
    vocab = gv_stem.get_feature_names()
    final_words = [stemmer.stem(w)
                   for w in words if stemmer.stem(w) in vocab]

    if final_words:
        new_query_string = " ".join(final_words)
        stem_query_vec = gv_stem.transform([new_query_string]).tocsc()
    else:
        print(f"No matches for stemmed search '{words}'")
        print()
        return None

    return stem_query_vec


def match_exact(words):
    exact_query_words = words.split()
    if len(exact_query_words) > 1:
        print(f"Searching for n-gram of length {len(exact_query_words)}...")
    gv, g_matrix = init_exact_search(len(exact_query_words))
    vocab = gv.get_feature_names()
    # Because of the n-gram parameter given to vectorizer, the query string is in its entirety one token, so
    # no need to remove unknown words here.
    if words in vocab:
        exact_query_vec = gv.transform([words]).tocsc()
    else:
        print(f"No matches for exact query '{words}'.")
        print()
        return None

    return exact_query_vec


def match_wildcard(words):
    # This works similarly to exact searches: the no. of words in the query determines the n-gram length
    # that is used for searches. Combined searches with word stems are not supported yet.
    ngram_len = len(words.split())
    gv, g_matrix = init_exact_search(ngram_len)
    vocab = gv.get_feature_names()

    # Replace the wildcard with a regex pattern matching 0 or more characters:
    word_no_wc = words.replace("*", ".*")

    # Compile a regex pattern based on the query word:
    wc_pattern = re.compile(word_no_wc)

    # Find all matching words in the vocabulary and form a new query word list:
    query_words = [w for w in vocab if re.fullmatch(wc_pattern, w)]
    if query_words:
        if ngram_len > 1:
            print("Looking for n-grams:", ", ".join(query_words))
        else:
            print("Looking for words:", ", ".join(query_words))
        new_query_string = " ".join(query_words)
        query_vec = gv.transform([new_query_string]).tocsc()
    else:
        print(f"No matches for wildcard search '{words}'")
        print()
        return None

    return query_vec


def ranked_scores_and_doc_ids(hits):
    return sorted(zip(np.array(hits[hits.nonzero()])[0], hits.nonzero()[1]),
                  reverse=True)


def relevance_search(query_string):
    # Vectorize query string
    words = query_string.split()
    matches = []

    if '"' in query_string:  # exact search  <"searchword"> or <"searchword1 searchword2...">
        # check if search structure is: 'searchword "searchword1 searchword2"'
        # separate them into different searches

        exact_words = re.findall(r'"[^"]+"', query_string)
        stem_query = query_string
        for phrase in exact_words:
            stem_query = stem_query.replace(phrase, "")
        stem_words = stem_query.split()
        exact_query = ' '.join(exact_words).replace('"', '')

        # Check if search contained stemmable search terms and continue search from there
        if len(stem_words) != 0:
            stem_query_vec = match_stems(stem_words)
            exact_query_vec = match_exact(exact_query)

            # Output result
            print("Your query '{:s}' matches the following documents:".format(
                query_string))

            # Check if word was found in stem search
            if stem_query_vec is not None:
                # Cosine similarity
                stem_hits = np.dot(stem_query_vec, g_matrix_stem)

                # Rank hits for stemmed
                stem_rank_hits = ranked_scores_and_doc_ids(stem_hits)

                # print("Stemmed search term results: ")
                for i, (score, doc_idx) in enumerate(stem_rank_hits):
                    matches.append(
                        {"name": doc_names[doc_idx], "content": documents[doc_idx], "id": doc_idx})
                #     print("Doc #{:d} (score: {:.4f}): {:s}...".format(
                #         i, score, documents[doc_idx][:50]))
                return matches

            # Check if word was found in exact search
            if exact_query_vec is not None:
                # Cosine similarity
                exact_hits = np.dot(exact_query_vec, g_matrix)

                # Rank hits for exact
                exact_rank_hits = ranked_scores_and_doc_ids(exact_hits)

                # print("Exact seach term results: ")
                for i, (score, doc_idx) in enumerate(exact_rank_hits):
                    matches.append(
                        {"name": doc_names[doc_idx], "content": documents[doc_idx], "id": doc_idx})
                #    print("Doc #{:d} (score: {:.4f}): {:s}...".format(
                #        i, score, documents[doc_idx][:50]))
                return matches

        else:
            query_string = query_string.replace('"', '')
            query_vec = match_exact(query_string)

            if query_vec is not None:
                # Cosine similarity
                hits = np.dot(query_vec, g_matrix)

                # Rank hits
                rank_hits = ranked_scores_and_doc_ids(hits)

                # Output result
                print("Your query '{:s}' matches the following documents:".format(
                    query_string))

                for i, (score, doc_idx) in enumerate(rank_hits):
                    matches.append(
                        {"name": doc_names[doc_idx], "content": documents[doc_idx], "id": doc_idx})
                #    print("Doc #{:d} (score: {:.4f}): {:s}...".format(
                #        i, score, documents[doc_idx][:50]))
                return matches

    elif "*" in query_string:
        query_vec = match_wildcard(query_string)

        # Cosine similarity
        hits = np.dot(query_vec, g_matrix)

        # Rank hits
        rank_hits = ranked_scores_and_doc_ids(hits)

        # Output result
        if query_vec is not None:
            # print("Your query '{:s}' matches the following documents:".format(
            #     query_string))
            for i, (score, doc_idx) in enumerate(rank_hits):
                matches.append(
                    {"name": doc_names[doc_idx], "content": documents[doc_idx], "id": doc_idx})
            #     print("Doc #{:d} (score: {:.4f}): {:s}...".format(
            #         i, score, documents[doc_idx][:50]))
            return matches

    else:  # stemming can be used
        query_vec = match_stems(words)
        # Output result
        # print("Your query '{:s}' matches the following documents:".format(
        #     query_string))
        if query_vec is not None:
            # Cosine similarity
            hits = np.dot(query_vec, g_matrix_stem)

            # Rank hits
            rank_hits = ranked_scores_and_doc_ids(hits)
            for i, (score, doc_idx) in enumerate(rank_hits):
                matches.append(
                    {"name": doc_names[doc_idx], "content": documents[doc_idx], "id": doc_idx})
            #     print("Doc #{:d} (score: {:.4f}): {:s}...".format(
            #         i, score, documents[doc_idx][:50]))
            return matches

    return matches


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
