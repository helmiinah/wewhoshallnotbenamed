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

test = soup.find_all('article')
documents = [t.get_text().replace('\n', ' ') for t in test]
# print('num of articles', len(articles))
# print(cleaned_documents[0])


cv = CountVectorizer(lowercase=True, binary=True, token_pattern=r"(?u)\b\w+\b")
sparse_matrix = cv.fit_transform(documents)

# print("Term-document matrix: (?)\n")
# print(sparse_matrix)

# print("Term-document matrix: (?)\n")
dense_matrix = sparse_matrix.todense()

# print("Term-document matrix:\n")
td_matrix = dense_matrix.T  # .T transposes the matrix

terms = cv.get_feature_names()


# print(terms)


def rewrite_token(t):
    # Can you figure out what happens here?

    return d.get(t, 'sparse_td_matrix[t2i["{:s}"]].todense()'.format(t))


def rewrite_query(query):  # rewrite every token in the query
    for i in range(len(query.split())):
        token = query.split()[i]
        if token not in terms and token not in d:
            if len(query.split()) == 1:
                # 1. case: token is only element of query
                # query = "unknown"
                return None
            elif i == 0:
                # 2. case: token is first element of query
                next_token = query.split()[i + 1].lower()
                if next_token == "and":
                    # query = "unknown AND ... "
                    return None
                elif next_token == "or":
                    # query = "unknown OR ... "
                    # return documents matching word after OR
                    return rewrite_token(query.split()[i + 2])
            elif len(query.split()) == 2 and query.split()[i-1] == "not":
                # query = "NOT unknown"
                # return a 1x100 numpy array filled with ones -> matches all documents
                return "np.ones((1,len(documents)), dtype=int)"
            elif i == len(query.split()) - 1:
                # 3. case: token is last element of query
                prev_token = query.split()[i - 1].lower()
                if prev_token == "and":
                    # query = " ... AND unknown"
                    return None
                elif prev_token == "or":
                    # query = " ... OR unknown"
                    # return documents matching word before OR
                    return rewrite_token(query.split()[i - 2])
                #elif prev_token == "not":
                    # query = " ... NOT unknown"
                    # return documents matching word before NOT
                #    return rewrite_token(query.split()[i - 2])
            elif i != 0 and i < len(query.split()) - 1:
                # 4. case: token is not first nor last element of query
                prev_token = query.split()[i - 1].lower()
                next_token = query.split()[i + 1].lower()
                # if there are two other words in the query in addition to unknown and both operator are "and":
                if len(query.split()) == 5 and prev_token == "and" and next_token == "and":
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

    return " ".join(rewrite_token(t) for t in query.split())


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
    # test_query(query)
    query = input("Add a query (press enter to quit): ").lower()
    if query == "":
        break
    print('Results')

    try:
        if rewrite_query(query) is None:
            print("Unknown word in query.")
        else:
            hits_matrix = eval(rewrite_query(query))
            hits_list = list(hits_matrix.nonzero()[1])
            print("Matched", len(hits_list), "documents.")

            for doc_idx in hits_list[:10]:
                print(f"Matching doc: [{doc_idx}] {documents[doc_idx][:50]}...")
    except KeyError:
        print('Bad query')
    print()
# print("Matching documents as vector (it is actually a matrix with one single row):", hits_matrix)
# print("The coordinates of the non-zero elements:", hits_matrix.nonzero())
