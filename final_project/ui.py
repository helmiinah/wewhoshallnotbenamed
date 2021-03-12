from flask import Flask, render_template, request, url_for, redirect
import search_engine as engine
import matplotlib.pyplot as plt
import spacy
import pke
import matplotlib
import en_core_web_sm
import os
import os.path
from wordcloud import WordCloud
from wordcloud import STOPWORDS
import re
import math
import pandas as pd
from random import randint
import requests
import numpy as np
from PIL import Image

# Initialize Flask instance
app = Flask(__name__)
engine.initialize()

matplotlib.use('Agg')
engine_choice = 'boolean' # to help ui remember engine choice

@app.route('/')
def redirect_to_search():
    return redirect('/search', code=302)


# Function search() is associated with the address base URL + "/search"
@app.route('/search')
def search():
    global engine_choice 

    # Get query from URL variable
    query = request.args.get('query')

    # Get the price range set by the user
    price_range = request.args.get('price_range')

    wildcard_query_words = ""

    # Extract minimum & maximum price
    if price_range:
        range_limits = re.match(r"\$(\d+) - \$(\d+)", price_range)
        min_price = float(range_limits[1])
        max_price = float(range_limits[2])

    # Get the minimum rating set by the user
    min_rating = request.args.get('min_rating')
    if min_rating:
        min_rating = float(min_rating)

    # Initialize list of matches
    matches = []

    # If query exists (i.e. is not None)
    if query:
        query = query.lower().strip()
        engine_choice = request.args.get("engine")

        if engine_choice == 'boolean':
            matches = engine.boolean_search(query)

        else:
            matches = engine.relevance_search(query)
            if len(matches) == 2:
                matches, wildcard_query_words = matches

    # Filter the matched wines based on price
    if price_range:
        matches = [wine for wine in matches if min_price <= wine["price"] <= max_price]

    # Filter the matched wines based on rating
    if min_rating:
        matches = [wine for wine in matches if wine["points"] >= min_rating]

    if matches:
        # Generate a random id for the country plot image
        random_id = str(randint(0, 1000000))

        country_plot_path = 'static/plots/country_plot_' + random_id + '.png'
        wordcloud_plot_path = 'static/plots/wordcloud_plot_' + random_id + '.png'

        generate_country_plot(matches, country_plot_path)
        generate_wordcloud_matches(matches, wordcloud_plot_path)
    else:
        country_plot_path=''
        wordcloud_plot_path = ''

    if wildcard_query_words != "":
        return render_template('index.html', matches=matches, number=len(matches), query=query, engine_choice=engine_choice,
                               country_plot_path=country_plot_path, wordcloud_plot_path=wordcloud_plot_path,
                               query_words=wildcard_query_words)
    else:
        return render_template('index.html', matches=matches, number=len(matches), query=query, engine_choice=engine_choice,
                               country_plot_path=country_plot_path, wordcloud_plot_path=wordcloud_plot_path)


@app.route('/search/<id>')
def show_document(id):
    query = request.args.get('query')
    query = query.lower()

    engine_choice = request.args.get('engine')

    idx = int(id)
    wine = engine.reviews.iloc[idx]
    country_codes = engine.country_codes

    # Count word matches inside document
    doc_matches = 0
    for word in wine["description"].split() + wine["variety"].split():
        if engine_choice == "boolean":
            query_splitted = query.split()

            if "and" in query_splitted:
                query_splitted.remove("and")
            if "or" in query_splitted:
                query_splitted.remove("or")
            if "not" in query_splitted:
                query_splitted.remove("not")

            if word.lower().strip(",.;:!?") in query_splitted:
                doc_matches += 1

        elif engine_choice == "relevance":
            # Check if query was exact match search
            if '"' in query:
                if query.strip('""') == word.lower().strip(",.;:!?"):
                    doc_matches += 1
            elif '*' in query:
                if query.strip('*') in word.lower().strip(",.;:!?"):
                    doc_matches += 1
            else:
                for q_word in query.split():
                    if q_word == word.lower().strip(",.;:!?") or q_word in word.lower().strip(",.;:!?"):
                        doc_matches += 1

    #if wine specific plots do not already exist, create them
    if not os.path.exists('static/plots/' + str(idx) + '_plt.png'):
      keyphrases = get_keyphrases(wine["description"])
      generate_plot(keyphrases, idx, wine["title"])
      generate_wordcloud(keyphrases, idx)

    wiki_path = re.sub(r"\s", r"_", wine["variety"])

    # Make a request to the possible Wikipedia article of the variety
    try:
        response = requests.get(f"https://en.wikipedia.org/wiki/{wiki_path}")
        # If successful, the article exists and can be shown to user:
        if str(response.status_code).startswith('2'):
            return render_template('wine.html', idx=str(idx), query=query, flag=country_codes[wine['country']],
                                   num_matches=doc_matches, engine=engine_choice, wine=wine, wiki_path=wiki_path)
        else:
            return render_template('wine.html', idx=str(idx), query=query, flag=country_codes[wine['country']],
                                num_matches=doc_matches, engine=engine_choice, wine=wine)
    except:
        return render_template('wine.html', idx=str(idx), query=query, flag=country_codes[wine['country']],
                                num_matches=doc_matches, engine=engine_choice, wine=wine)


def get_keyphrases(document):
    extractor = pke.unsupervised.TopicRank()
    extractor.load_document(input=document, language='en')
    extractor.candidate_selection()
    extractor.candidate_weighting()
    keyphrases = extractor.get_n_best(n=10)
    return keyphrases


def generate_plot(keyphrases, idx, name):
    phrases = [p[0] for p in keyphrases]
    scores = [p[1] for p in keyphrases]
    plt.figure()
    plt.title(f'Themes in "{name}"')  # add a title
    plt.xlabel('Keyphrases')  # name the x-axis
    plt.ylabel('Scores')  # name of the y-axis
    ax = plt.gca()
    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
    plt.plot(phrases, scores)
    plt.tight_layout()
    plot_path = 'static/plots/' + str(idx) + '_plt.png'
    plt.savefig(plot_path)
    plt.close()


def red_color_func(word, font_size, font_path, position, orientation, random_state=None):
    # Function source: https://github.com/amueller/word_cloud/issues/52
    return "hsl(10, 100%%, %d%%)" % randint(40, 100)


def generate_wordcloud(keyphrases, idx):
    keyphrases = dict(keyphrases)
    wordcloud = WordCloud(width=800, height=800,
                          background_color='white',
                          min_font_size=10,
                          color_func=red_color_func).generate_from_frequencies(keyphrases)
    plt.figure()
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plot_path = 'static/plots/' + str(idx) + '_wordcloud.png'
    plt.savefig(plot_path)
    plt.close()


def generate_wordcloud_matches(matches, plot_path):
    matches = pd.DataFrame(matches)
    descriptions = " ".join([desc for desc in matches["description"]])
    stopwords = set(list(STOPWORDS) + ["wine", "drink", "flavor", "flavors", "finish", "aroma", "aromas", "palate"])
    mask = np.array(Image.open("static/winebottle.png"))
    wordcloud = WordCloud(width=1500, height=1500,
                          background_color='#c5b68b',
                          min_font_size=10,
                          stopwords=stopwords,
                          mask=mask,
                          color_func=red_color_func).generate(descriptions)
    plt.figure()
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(plot_path, facecolor='#c5b68b')
    plt.close()


def get_top10_labels(sizes, labels):
    new_labels = []
    for size, label in zip(sizes, labels):
        if size in sorted(sizes)[-10:] and len(new_labels) < 10:
            new_labels.append(label)
        else:
            new_labels.append("")
    return new_labels


def generate_country_plot(matches, plot_path):
    matches = pd.DataFrame(matches)
    counts = matches["country"].value_counts()
    values = counts.values
    labels = get_top10_labels(values, counts.keys())
    fig1, ax = plt.subplots()
    l = ax.pie(values, startangle=-90)  # autopct='%1.1f%%' <- Percentage

    # Add the labels so that their angle aligns with the slice:
    for label, t in zip(labels, l[1]):
        x, y = t.get_position()
        angle = int(math.degrees(math.atan2(y, x)))
        ha = "left"
        if x < 0:
            angle -= 180
            ha = "right"
        plt.annotate(label, xy=(x, y), rotation=angle, ha=ha, va="center", rotation_mode="anchor", size=8)

    fig1.set_facecolor('#c5b68b')
    plt.tight_layout(pad=0)
    plt.savefig(plot_path)
    plt.close()

