import os

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import streamlit as st
import pickle


# Define the list of possible categories and items
categories = ['Appetizers', 'Beverages', 'Desserts', 'Main Course']
items = [
        "Iced Tea", "New York Cheesecake", "Tiramisu", "Soda", "Caprese Salad", 
        "Coffee", "Vegetable Stir-Fry", "Spinach Artichoke Dip", "Bruschetta", 
        "Fruit Tart", "Stuffed Mushrooms", "Lemonade", "Grilled Steak", 
        "Chocolate Lava Cake", "Shrimp Scampi", "Chicken Alfredo"
    ]
all_ingredients = [
    'Alfredo Sauce',
    'Basil',
    'Butter',
    'Chicken',
    'Chocolate',
    'Eggs',
    'Fettuccine',
    'Garlic',
    'Olive Oil',
    'Parmesan',
    'Sugar',
    'Tomatoes',
    'confidential'
]

# define preprocessing function
def preprocess_inputs(price, category, item, ingredients):
    # Create a DataFrame for one-hot encoding
    df = pd.DataFrame({
        'Price': [price],
        'MenuCategory': [category],
        'MenuItem': [item],
        'Ingredients': [ingredients]
    })
    
    # StandardScaler for numerical features
    scaler = StandardScaler()
    df['Price'] = scaler.fit_transform(df[['Price']])
    
    le = LabelEncoder()
    df['MenuCategory'] = le.fit_transform(df['MenuCategory'])
    df['MenuItem'] = le.fit_transform(df['MenuItem'])
    
    # OneHotEncoder for categorical features
    encoder = OneHotEncoder(categories=[categories, items], handle_unknown='ignore')
    encoded_features = encoder.fit_transform(df[['MenuCategory', 'MenuItem']])
    
    # Prepare encoded feature names
    feature_names = encoder.get_feature_names_out(['MenuCategory', 'MenuItem'])
    
    # Combine encoded features and the numerical feature
    encoded_df = pd.DataFrame(encoded_features, columns=feature_names)
    
    # Example encoding for ingredients (one-hot encoding)
    ingredients_list = ingredients.split(', ')
    ingredients_encoded = [1 if ing in ingredients_list else 0 for ing in all_ingredients]
    
    # Combine all features
    final_df = pd.concat([df[['Price']].reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)
    for ingredient in all_ingredients:
        final_df[ingredient] = ingredients_encoded[all_ingredients.index(ingredient)]
    
    return final_df

models_path = {
    "DecisionTree": "../Models/DecisionTree.pkl",
    "DNN_model": "../Models/DNN_model.pkl",
    "RandomForest": "../Models/RandomForest.pkl",
    "XGBoost": "../Models/XGBoost.pkl"
}

def load_model(model_path):

    if not os.path.exists(model_path):
        st.error(f"Model file not found    :   {model_path}")
        return None

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    return model

# Streamlit app layout
st.set_page_config(page_title="Menu Items Profitability Predictor", layout="wide")
st.sidebar.title("Model Selection")

# Sidebar for model selection
model_name = st.sidebar.selectbox("Select a Model for prediction", list(models_path.keys()))
model_path = models_path[model_name]
model = load_model(model_path)

if model is None:
    st.stop()

# Main Content
st.title("Restaurant Menu Items Profitability Prediction")
st.markdown("Welcome to the Restaurant Menu Items Profitability Predictor! Use this app to predict the profitability of menu items using different models.")

# Input features
st.header("Input Features")

price = st.number_input('Price', min_value=0.0, step=0.01)

# Categorical inputs
category = st.selectbox('Menu Category', categories)
item = st.selectbox('Menu Item', items)

# Ingredients input
ingredients = st.text_input('Ingredients [coma-separated]', 'Tomato, Basil, Garlic')

# Predict button
if st.button('Predict Profitability'):

    # prepare the input data
    input_data = preprocess_inputs(price, category, item, ingredients)

    # Make prediction
    prediction = model.predict(input_data)

    # Display  the prediction
    st.header("Prediction Result")
    st.write('Predicted Profitability   :   ', prediction[0])