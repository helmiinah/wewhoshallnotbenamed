from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import nltk
from nltk.stem.snowball import SnowballStemmer
import re
import numpy as np
import pandas as pd

# globals

d = {"and": "&", "AND": "&",
     "or": "|", "OR": "|",
     "not": "1 -", "NOT": "1 -",
     "(": "(", ")": ")"}

reviews = pd.DataFrame()
country_codes = {}
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
    global reviews
    global country_codes
    global terms
    global sparse_td_matrix
    global t2i
    global gv_stem
    global g_matrix_stem

    reviews = pd.read_csv("./static/10k-winemag-reviews.csv", sep=",", usecols=range(14))
    reviews = reviews.rename(columns={'Unnamed: 0': 'id'})
    reviews["country"] = reviews["country"].fillna("Unknown")
    reviews["price"] = pd.to_numeric(reviews["price"], downcast="float")

    country_codes = {"Luxembourg": "lu", "Spain": "es", "Australia": "au", "South Africa": "za", "Czech Republic": "cz", "Slovenia": "si", 
                     "France": "fr", "Moldova": "md", "Serbia": "rs", "Argentina": "ar", "Mexico": "mx", "Croatia": "hr", "England": "gb", "Germany": "de", 
                     "Georgia": "ge", "Lebanon": "lb", "Chile": "cl", "Canada": "ca", "Morocco": "ma", "Uruguay": "uy", "Cyprus": "cy", "New Zealand": "nz", 
                     "Turkey": "tr", "Greece": "gr", "Romania": "ro", "Brazil": "br", "Portugal": "pt", "Bulgaria": "bg", "Austria": "at", "Hungary": "hu", 
                     "Italy": "it", "Armenia": "am", "Peru": "pe", "India": "in", "US": "us", "Israel": "il", "Unknown": "Unknown"}

    reviews["search_data"] = reviews['description'] + ' ' + reviews['variety'] + ' ' + reviews['title']
    reviews["price-quality"] = round(reviews['points']/reviews["price"], 1)
    

    # initialize boolean search tools
    cv = CountVectorizer(lowercase=True, binary=True,
                         token_pattern=r"(?u)\b\w+\b")
    sparse_matrix = cv.fit_transform(reviews['search_data'].tolist())
    dense_matrix = sparse_matrix.todense()
    td_matrix = dense_matrix.T
    terms = cv.get_feature_names()
    t2i = cv.vocabulary_
    sparse_td_matrix = sparse_matrix.T.tocsr()

    # initialize relevance search tools for stemming
    gv_stem = TfidfVectorizer(tokenizer=tokenize, lowercase=True,
                              sublinear_tf=True, use_idf=True, norm="l2")
    g_matrix_stem = gv_stem.fit_transform(reviews['search_data'].tolist()).T.tocsr()


def init_exact_search(n):
    # Exact search tools are initialized independently, as number of query words is needed and that is not known
    # until later. This initialization works for both single-word and multi-word searches.
    global gv
    global g_matrix
    gv = TfidfVectorizer(lowercase=True,
                         sublinear_tf=True, use_idf=True, norm="l2", ngram_range=(n, n))
    g_matrix = gv.fit_transform(reviews['search_data'].tolist()).T.tocsr()
    return gv, g_matrix


def rewrite_token(t):
    # for boolean search
    if t not in terms and t not in d:
        return "np.zeros((1, len(reviews['description'].tolist())), dtype=int)"
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
            matches = []
            for doc_idx in hits_list:
                matches.append(reviews.iloc[doc_idx])
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
        return None

    return stem_query_vec


def match_exact(words):
    exact_query_words = words.split()
    gv, g_matrix = init_exact_search(len(exact_query_words))
    vocab = gv.get_feature_names()
    # Because of the n-gram parameter given to vectorizer, the query string is in its entirety one token, so
    # no need to remove unknown words here.
    if words in vocab:
        exact_query_vec = gv.transform([words]).tocsc()
    else:
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
        new_query_string = " ".join(query_words)
        query_vec = gv.transform([new_query_string]).tocsc()
    else:
        return None

    # Also return a string of the query words in order to show them in UI:
    return query_vec, ", ".join(query_words)


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

            # Check if word was found in stem search
            if stem_query_vec is not None:
                # Cosine similarity
                stem_hits = np.dot(stem_query_vec, g_matrix_stem)

                # Rank hits for stemmed
                stem_rank_hits = ranked_scores_and_doc_ids(stem_hits)

                for i, (score, doc_idx) in enumerate(stem_rank_hits):
                    matches.append(reviews.iloc[doc_idx])
                return matches

            # Check if word was found in exact search
            if exact_query_vec is not None:
                # Cosine similarity
                exact_hits = np.dot(exact_query_vec, g_matrix)

                # Rank hits for exact
                exact_rank_hits = ranked_scores_and_doc_ids(exact_hits)

                for i, (score, doc_idx) in enumerate(exact_rank_hits):
                    matches.append(reviews.iloc[doc_idx])
                return matches

        else:
            query_string = query_string.replace('"', '')
            query_vec = match_exact(query_string)

            if query_vec is not None:
                # Cosine similarity
                hits = np.dot(query_vec, g_matrix)

                # Rank hits
                rank_hits = ranked_scores_and_doc_ids(hits)

                for i, (score, doc_idx) in enumerate(rank_hits):
                    matches.append(reviews.iloc[doc_idx])
                return matches

    elif "*" in query_string:
        query_vec, words = match_wildcard(query_string)

        # Cosine similarity
        hits = np.dot(query_vec, g_matrix)

        # Rank hits
        rank_hits = ranked_scores_and_doc_ids(hits)
        
        if query_vec is not None:
            for i, (score, doc_idx) in enumerate(rank_hits):
                matches.append(reviews.iloc[doc_idx])
            return matches, words

    else:  # stemming can be used
        query_vec = match_stems(words)
        if query_vec is not None:
            # Cosine similarity
            hits = np.dot(query_vec, g_matrix_stem)
            # Rank hits
            rank_hits = ranked_scores_and_doc_ids(hits)
            for i, (score, doc_idx) in enumerate(rank_hits):
                matches.append(reviews.iloc[doc_idx])
            return matches

    return matches


if __name__ == "__main__":
    initialize()
