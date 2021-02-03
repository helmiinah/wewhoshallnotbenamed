from sklearn.feature_extraction.text import CountVectorizer
from bs4 import BeautifulSoup
import re
import numpy as np

toy_documents = ["This is a silly example",
                 "A better example",
                 "Nothing to see here",
                 "This is a great and long example"]

d = {"and": "&", "AND": "&",
     "or": "|", "OR": "|",
     "not": "1 -", "NOT": "1 -",
     "(": "(", ")": ")"}  # operator replacements

with open('text_data.txt', encoding="utf8") as file:
    contents = file.read()

soup = BeautifulSoup(contents, 'html.parser')

articles = soup.find_all('article')
documents = [t.get_text().replace('\n', ' ') for t in articles]
# print('num of articles', len(articles))
# print(cleaned_documents[0])

cv = CountVectorizer(lowercase=True, binary=True, token_pattern=r"(?u)\b\w+\b")
sparse_matrix = cv.fit_transform(documents)
dense_matrix = sparse_matrix.todense()
td_matrix = dense_matrix.T  # .T transposes the matrix

terms = cv.get_feature_names()
# print(terms)


def handle_oov_parenthesis(query):
    # query = word AND/OR (word AND/OR word), ( word AND/OR word ) AND/OR word
    # with an OOV somewhere and possible NOTs, except before parenthesis
    query_parts = query.split()

    if query.startswith('('):  # ( word AND/OR word ) AND/OR word
        parenthesis = query[2:query.index(')')]
        rest = query[query.index(')')+2:].strip()
        rewritten_par = rewrite_query(parenthesis)
        rewritten_rest = rewrite_query(re.sub('(and|or) ', '', rest))

        if rest.startswith('and'):  # () AND <word>
            if not rewritten_par or not rewritten_rest:  # either part is not valid
                return None
            else:
                return '({}) & {}'.format(rewritten_par, rewritten_rest)

        else:  # () OR <word>
            rewritten_query = ''
            if rewritten_par:
                rewritten_query += '({}) | '.format(rewritten_par)
            if rewritten_rest:
                rewritten_query += rewritten_rest
            return rewritten_query

    else:  # word AND/OR (word AND/OR word)
        start = query[:query.index('(')].strip()
        parenthesis = query[query.index('(')+2:len(query)-2]
        rewritten_start = rewrite_query(re.sub(' (and|or)', '', start))
        rewritten_par = rewrite_query(parenthesis)

        if start.endswith('and'):
            if not rewritten_par or not rewritten_start:  # either part is not valid
                return None
            else:
                return '{} & ({})'.format(rewritten_start, rewritten_par)
        else:  # <word> OR ()
            rewritten_query = ''
            if rewritten_start:
                rewritten_query = rewritten_start
            if rewritten_par:
                rewritten_query += ' | ({})'.format(rewritten_par)

            return rewritten_query

    return None


def rewrite_token(t):
    return d.get(t, 'sparse_td_matrix[t2i["{:s}"]].todense()'.format(t))


def rewrite_query(query):  # rewrite every token in the query
    query_parts = query.split()
    for i in range(len(query_parts)):
        token = query_parts[i]
        if token not in terms and token not in d:
            # Token is an OOV word
            if len(query_parts) == 1:
                # 1. case: token is only element of query
                # query = "unknown"
                return None

            elif i == 0:
                # 2. case: token is first element of query
                next_token = query_parts[i + 1].lower()
                if next_token == "and":
                    # query = "unknown AND ... "
                    return None
                elif next_token == "or":
                    # query = "unknown OR ... "
                    # return documents matching word after OR
                    return rewrite_token(query_parts[i + 2])

            elif len(query_parts) == 2 and query_parts[i-1] == "not":
                # query = "NOT unknown"
                # return a 1x100 numpy array filled with ones -> matches all documents
                return "np.ones((1,len(documents)), dtype=int)"

            elif len(query_parts) == 4 and query_parts[i-1] == "not":
                # queary: two words, AND/OR, and NOT
                if i-2 > 0:
                    if query_parts[i-2] == "or":
                        # query = "<known> OR NOT <unknown>"
                        # -> match all documents
                        return "np.ones((1,len(documents)), dtype=int)"
                    if query_parts[i-2] == "and":
                        # query = "<known> AND NOT <unknown>"
                        # -> return matching documents for the first search term
                        return rewrite_query(query_parts[i-3])
                else:
                    if query_parts[i+1] == "or":
                        # query = "NOT <unknown> OR <known>"
                        # -> match all documents
                        return "np.ones((1,len(documents)), dtype=int)"
                    if query_parts[i+1] == "and":
                        # query = "<NOT unknown> AND <known>"
                        # -> return matching documents for the last search term
                        return rewrite_query(query_parts[i+2])

            elif '(' in query:
                # bonus case: OOV query with parenthesis. Assumes query is grammatically correct
                return handle_oov_parenthesis(query)

            elif i == len(query_parts) - 1:
                # 3. case: token is last element of query
                prev_token = query_parts[i - 1].lower()
                if prev_token == "and":
                    # query = " ... AND unknown"
                    return None
                elif prev_token == "or":
                    # query = " ... OR unknown"
                    # return documents matching word before OR
                    new_query = query.split()
                    new_query = new_query[:-2]
                    return rewrite_query(" ".join(new_query))
                # elif prev_token == "not":
                    # query = " ... NOT unknown"
                    # return documents matching word before NOT
                #    return rewrite_token(query.split()[i - 2])

            elif i != 0 and i < len(query_parts) - 1:
                # 4. case: token is not first nor last element of query
                prev_token = query_parts[i - 1].lower()
                next_token = query_parts[i + 1].lower()
                # if there are two other words in the query in addition to unknown and both operator are "and":
                if len(query_parts) == 5 and prev_token == "and" and next_token == "and":
                    return None
                elif prev_token == "or" and next_token == "or":
                    # remove the current unknown word and the next operator "or" and recursively call function again:
                    new_query = query.split()
                    new_query.pop(i)
                    new_query.pop(i)
                    # print(" ".join(new_query))  <- print to inspect new query
                    return rewrite_query(" ".join(new_query))
                elif prev_token == "or" and next_token == "and":
                    # remove current unknown word and the previous operator "or" and recursively call function again:
                    new_query = query.split()
                    new_query.pop(i)
                    new_query.pop(i - 1)
                    # print(" ".join(new_query))  <- print to inspect new query
                    return rewrite_query(" ".join(new_query))
                elif prev_token == "and" and next_token == "or":
                    # remove current unknown word, the next operator "or" and the word before "or"
                    # and recursively call function again:
                    new_query = query.split()
                    new_query.pop(i)
                    new_query.pop(i)
                    new_query.pop(i - 1)
                    new_query.pop(i - 2)
                    # print(" ".join(new_query))  <- print to inspect new query
                    return rewrite_query(" ".join(new_query))

    return " ".join(rewrite_token(t) for t in query_parts)


def test_query(query):
    print("Query: '" + query + "'")
    print("Rewritten:", rewrite_query(query))
    # Eval runs the string as a Python command
    print("Matching:", eval(rewrite_query(query)))
    print()


t2i = cv.vocabulary_  # shorter notation: t2i = term-to-index
# print("Query: example")
# print(td_matrix[t2i["example"]])

sparse_td_matrix = sparse_matrix.T.tocsr()
# print(sparse_td_matrix)

while True:
    query = input("Add a query (press enter to quit): ").lower().strip()
    # test_query(query)
    if query == "":
        break

    try:
        if rewrite_query(query) is None:
            print("No matches.")
        else:
            # print(rewrite_query(query))
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
# print("Matching documents as vector (it is actually a matrix with one single row):", hits_matrix)
# print("The coordinates of the non-zero elements:", hits_matrix.nonzero())


# refactoring, error handling
