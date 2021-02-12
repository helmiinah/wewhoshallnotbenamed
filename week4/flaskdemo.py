from flask import Flask, render_template, request, url_for, redirect
import search_engine as engine

# Initialize Flask instance
app = Flask(__name__)
engine.initialize()

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


@app.route('/search/<id>')
def show_document(id):
    docs = engine.documents
    names = engine.doc_names
    idx = int(id)
    return render_template('document.html', name=names[idx], content=docs[idx])


@app.route('/search/')
def return_to_results():
    # helper method to ensure original results render when returning to search page
    global back_to_list
    back_to_list = True
    return redirect(url_for('search'))
