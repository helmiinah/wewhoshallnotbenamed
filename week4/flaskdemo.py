from flask import Flask, render_template, request
import search_engine as engine 


#Initialize Flask instance
app = Flask(__name__)
engine.initialize()

example_data = [
    {'name': 'Cat sleeping on a bed', 'source': 'cat.jpg'},
    {'name': 'Misty forest', 'source': 'forest.jpg'},
    {'name': 'Bonfire burning', 'source': 'fire.jpg'},
    {'name': 'Old library', 'source': 'library.jpg'},
    {'name': 'Sliced orange', 'source': 'orange.jpg'}
]

#Function search() is associated with the address base URL + "/search"
@app.route('/search')
def search():

    #Get query from URL variable
    query = request.args.get('query')

    #Initialize list of matches
    matches = []
    

    #If query exists (i.e. is not None)
    if query:
        query=query.lower().strip()
        engine_choice=request.args.get("engine")

        if engine_choice=='boolean':
            matches=engine.boolean_search(query)

        else:
            matches=engine.relevance_search(query)
        # #Look at each entry in the example data
        # for entry in example_data:
        #     #If an entry name contains the query, add the entry to matches
        #     if query.lower() in entry['name'].lower():
        #         matches.append(entry)

    #Render index.html with matches variable
    return render_template('index.html', matches=matches)

