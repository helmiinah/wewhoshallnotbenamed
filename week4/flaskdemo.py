from flask import Flask, render_template, request, url_for, redirect
import search_engine as engine
import matplotlib.pyplot as plt
import spacy
import pke
import matplotlib
import en_core_web_sm


# Initialize Flask instance
app = Flask(__name__)
engine.initialize()

matplotlib.use('Agg')

example_data = [
    {'name': 'Cat sleeping on a bed', 'source': 'cat.jpg'},
    {'name': 'Misty forest', 'source': 'forest.jpg'},
    {'name': 'Bonfire burning', 'source': 'fire.jpg'},
    {'name': 'Old library', 'source': 'library.jpg'},
    {'name': 'Sliced orange', 'source': 'orange.jpg'}
]

# These globals exist to allow saving previous query and matches and such,
# so that after opening a document, the user can return to the original result list
engine_choice = 'boolean'
previous_query = ''
previous_matches = ''
back_to_list = False


# Function search() is associated with the address base URL + "/search"
@app.route('/search')
def search():
    global engine_choice, previous_matches
    global previous_query, back_to_list

    if back_to_list:  # We return from a document and wish to see previous results
        back_to_list = False
        return render_template('index.html', matches=previous_matches, number=len(previous_matches), query=previous_query, engine_choice=engine_choice)

    # Get query from URL variable
    query = request.args.get('query')
    # Initialize list of matches
    matches = []

    # If query exists (i.e. is not None)
    if query:
        query = query.lower().strip()
        previous_query = query
        engine_choice = request.args.get("engine")

        if engine_choice == 'boolean':
            matches = engine.boolean_search(query)

        else:
            matches = engine.relevance_search(query)
        previous_matches = matches

    # Render index.html with matches variable
    return render_template('index.html', matches=matches, number=len(matches), query=query, engine_choice=engine_choice)


def generate_plot(idx, document):
    extractor = pke.unsupervised.TopicRank()
    extractor.load_document(input=document, language='en')
    extractor.candidate_selection()
    extractor.candidate_weighting()
    keyphrases = extractor.get_n_best(n=10)
    phrases = [p[0] for p in keyphrases]
    scores = [p[1] for p in keyphrases]
    plt.figure()
    plt.title('Themes') # add a title 
    plt.xlabel('Keyphrases') # name the x-axis
    plt.ylabel('Scores') # name of the y-axis
    plt.plot(phrases, scores)
    plt.savefig('static/' + str(idx) + '_plt.png')


@app.route('/search/<id>')
def show_document(id):
    query = request.args.get('query')
    query = query.lower()

    engine_choice = request.args.get('engine')

    docs = engine.documents
    names = engine.doc_names
    idx = int(id)

    # Count word matches inside document
    doc_matches = 0
    for word in docs[idx].split():
        if engine_choice == "boolean":
            query_splitted = query.split()

            if "and" in query_splitted:
                query_splitted.remove("and")
            elif "or" in query_splitted:
                query_splitted.remove("or")
            elif "not" in query_splitted:
                query_splitted.remove("not")
            
            if word.lower() in query_splitted:
                doc_matches += 1
                
        elif engine_choice == "relevance":
            # Check if query was exact match search
            if '"' in query:
                if query.strip('""') == word:
                    doc_matches += 1
            else:
                if query == word.lower() or query in word.lower():
                    doc_matches += 1

    generate_plot(idx, docs[idx]) 

    return render_template('document.html', idx=str(idx), name=names[idx], content=docs[idx], query=query, num_matches=doc_matches, engine=engine_choice)


@app.route('/search/')
def return_to_results():
    # helper method to ensure original results render when returning to search page
    global back_to_list
    back_to_list = True
    return redirect(url_for('search'))