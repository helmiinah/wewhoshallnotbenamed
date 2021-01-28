from sklearn.feature_extraction.text import CountVectorizer


documents = ["This is a silly example",
             "A better example",
             "Nothing to see here",
             "This is a great and long example"]


d = {"and": "&", "AND": "&",
     "or": "|", "OR": "|",
     "not": "1 -", "NOT": "1 -",
     "(": "(", ")": ")"}          # operator replacements


def rewrite_token(t):
    # Can you figure out what happens here?
    return d.get(t, 'sparse_td_matrix[t2i["{:s}"]].todense()'.format(t))


def rewrite_query(query):  # rewrite every token in the query
    return " ".join(rewrite_token(t) for t in query.split())


def test_query(query):
    print("Query: '" + query + "'")
    print("Rewritten:", rewrite_query(query))
    # Eval runs the string as a Python command
    print("Matching:", eval(rewrite_query(query)))
    print()


cv = CountVectorizer(lowercase=True, binary=True)
sparse_matrix = cv.fit_transform(documents)

print("Term-document matrix: (?)\n")
print(sparse_matrix)

#print("Term-document matrix: (?)\n")
dense_matrix = sparse_matrix.todense()

#print("Term-document matrix:\n")
td_matrix = dense_matrix.T   # .T transposes the matrix

terms = cv.get_feature_names()

t2i = cv.vocabulary_  # shorter notation: t2i = term-to-index
#print("Query: example")
# print(td_matrix[t2i["example"]])

sparse_td_matrix = sparse_matrix.T.tocsr()
print(sparse_td_matrix)


hits_matrix = eval(rewrite_query("NOT example OR great"))
#print("Matching documents as vector (it is actually a matrix with one single row):", hits_matrix)
#print("The coordinates of the non-zero elements:", hits_matrix.nonzero())

hits_list = list(hits_matrix.nonzero()[1])
print(hits_list)

for doc_idx in hits_list:
    print("Matching doc:", documents[doc_idx])
