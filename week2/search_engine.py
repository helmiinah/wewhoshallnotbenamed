from sklearn.feature_extraction.text import CountVectorizer
from bs4 import BeautifulSoup
import re

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
        if not token in terms and not token in d:
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
                elif prev_token == "not":
                    # query = " ... NOT unknown"
                    # return documents matching word before NOT
                    return rewrite_token(query.split()[i - 2])
            elif i != 0 and i < len(query.split()) - 1:
                # 4. case: token is not first nor last element of query
                continue
            #    prev = query.split()[i-1]
            #    next = query.split()[i+1]
            #    if prev in ["AND", "and"]:

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

            for doc_idx in hits_list[:10]:
                print(f"Matching doc: [{doc_idx}] {documents[doc_idx][:50]}...")
    except KeyError:
        print('Bad query')
    print()
# print("Matching documents as vector (it is actually a matrix with one single row):", hits_matrix)
# print("The coordinates of the non-zero elements:", hits_matrix.nonzero())
