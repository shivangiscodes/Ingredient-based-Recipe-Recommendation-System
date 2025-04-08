import streamlit as st
import json
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load Recipes
def load_recipes(files):
    recipes = {}
    for file in files:
        category = os.path.splitext(file)[0]
        with open(file, "r") as f:
            text = f.read()
            # Extract recipe lists
            recipe_list = re.findall(r"\[(.*?)\]", text, re.DOTALL)
            recipes[category] = recipe_list
    return recipes

# Initialize Data
files = [
    "DataFiles\\Salad.txt", 
    "DataFiles\\Sides.txt", 
    "DataFiles\\Drinks.txt", 
    "DataFiles\\Desserts.txt", 
    "DataFiles\\Breakfast and Brunch.txt", 
    "DataFiles\\Bread.txt"
]
recipes = load_recipes(files)

# Flatten Recipe Data
recipe_texts = []
categories = []
for category, recipe_list in recipes.items():
    for recipe in recipe_list:
        recipe_texts.append(recipe)
        categories.append(category)

# TF-IDF Vectorization
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(recipe_texts)

# Cosine Similarity for Recommendations
def recommend_recipe(query, top_n=5):
    query_vector = tfidf.transform([query])
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    recommendations = sorted(enumerate(similarities), key=lambda x: -x[1])[:top_n]
    return [(recipe_texts[i], categories[i], similarities[i]) for i, _ in recommendations]

# Streamlit UI
st.title("Recipe Recommendation System")
st.write("Enter a keyword or ingredient to get recipe recommendations.")

# User Input
query = st.text_input("Search for recipes:", "")

if query:
    # Get Recommendations
    recommendations = recommend_recipe(query)
    st.write("### Recommendations:")
    for rec in recommendations:
        st.write(f"**Recipe:** {rec[0]}")
        st.write(f"**Category:** {rec[1]}")
        st.write(f"**Similarity Score:** {rec[2]:.2f}")
        st.write("---")
