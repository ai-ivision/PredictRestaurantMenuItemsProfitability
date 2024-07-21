# -*- coding: utf-8 -*-

# -- Sheet --

# # **Predict Restaurant Menu Items Profitability**
# 
# ## **Background:**
# In the highly competitive restaurant industry, understanding the profitability of menu items is crucial for maintaining a successful business. Profitability can be influenced by various factors such as the ingredients used, the price of the item, the restaurant category, and customer preferences. Efficiently predicting which menu items are likely to be more profitable can help restaurant managers make informed decisions about menu design, pricing strategies, and inventory management.
# 
# ## **Objective:**
# The objective of this project is to develop a predictive model that can classify the profitability of restaurant menu items into categories such as **Low, Medium, and High**. This model will leverage historical data on menu items, including their prices, ingredients, and other relevant attributes, to make accurate profitability predictions.
# 
# ## **Data:**
# The dataset consists of 1000 entries, each representing a menu item from various restaurants. The features of the dataset are as follows:
# - **RestaurantID**: Unique identifier for the restaurant.
# - **MenuCategory**: Category of the menu item (e.g., Appetizers, Main Course, Desserts).
# - **MenuItem**: Name of the menu item.
# - **Ingredients**: List of ingredients used in the menu item.
# - **Price**: Price of the menu item.
# - **Profitability**: Profitability category of the menu item (Low, Medium, High).
# 
# ## **Tasks:**
# 
# ### **1. Data Exploration and Preprocessing:**
# - Conduct exploratory data analysis (EDA) to understand the distribution and relationships within the data.
# - Handle missing values, if any, and encode categorical features appropriately.
# - Engineer new features that may help in improving the model's performance, such as the number of ingredients used or specific ingredient indicators.
# 
# ### **2. Model Development:**
# - Develop several machine learning models (e.g., RandomForestClassifier, DecisionTreeClassifier, XGBClassifier) to predict the profitability of menu items.
# - Perform hyperparameter tuning using GridSearchCV to find the best parameters for each model.
# 
# ### **3. Model Evaluation:**
# - Evaluate the performance of each model using metrics such as accuracy and F1 score.
# - Compare the models and select the best-performing one based on the evaluation metrics.
# 
# ### **4. Deep Learning Model:**
# - Develop a Deep Neural Network (DNN) model with multiple layers to predict profitability.
# - Train and evaluate the DNN model, and compare its performance with traditional machine learning models.
# 
# ### **5. Model Saving and Deployment:**
# - Save the best models (with accuracy greater than 80%) in the `models` directory for future use.
# - Document the model training and evaluation process, and provide recommendations for deploying the model in a production environment.
# 
# ## **Expected Outcome:**
# By the end of this project, we aim to have a robust predictive model that accurately classifies the profitability of restaurant menu items. This model can be used by restaurant managers to optimize their menus, set competitive prices, and ultimately enhance their profitability.
# 
# ## **Impact:**
# The implementation of this predictive model will enable restaurants to make data-driven decisions, leading to improved financial performance and customer satisfaction. By understanding which menu items are more profitable, restaurants can focus on promoting and improving these items, thereby maximizing their overall profitability.
# 
# ## **Acknowledgements:**
# We would like to extend our gratitude to the following resources and tools, which made this project possible:
# 
# - **[TensorFlow](https://www.tensorflow.org/)**: For providing a robust framework for developing and training deep learning models.
# - **[Scikit-learn (sklearn)](https://scikit-learn.org/stable/)**: For offering comprehensive tools for data analysis, preprocessing, and machine learning.
# - **[Keras](https://keras.io/)**: For its easy-to-use API that facilitated the creation and training of our deep neural network models.
# - **[Kaggle](https://www.kaggle.com/datasets/rabieelkharoua/predict-restaurant-menu-items-profitability)**: For providing the dataset that was crucial for training and evaluating our models.
# 
# Their contributions have been invaluable in achieving the objectives of this project.


# ## Importing Libraries


import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score

from xgboost import XGBClassifier

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler

import pickle

# ## Load and EDA


dataset = pd.read_csv("./Data/restaurant_menu_optimization_data.csv")

dataset.shape

dataset.head(10)

dataset.info()

dataset.isnull().sum()

# Analyze the distribution of the target variable (Profitability).
sns.countplot(x='Profitability', data=dataset)
plt.show()

dataset['Profitability'].value_counts()

for col in ['MenuCategory', 'MenuItem']:
    print(f"The Value Counts for {col}  :   \n{dataset[col].value_counts(())}\n")
    print(f"-"*75)
    sns.countplot(x=col, data=dataset)
    plt.show()

sns.histplot(dataset['Price'], kde=True)
plt.show()

cat_cols = [x for x in dataset.columns if dataset[x].dtypes == 'object']
cat_cols.remove('Profitability')

cat_cols

for col in cat_cols:
    contgnc_tbl = pd.crosstab(dataset[col], dataset['Profitability'])
    contgnc_tbl = contgnc_tbl.loc[:, ['Low', 'Medium', 'High']]
    plt.figure(figsize=(10, 6))
    plt.title(col)
    sns.heatmap(contgnc_tbl, annot=True, cmap='coolwarm', fmt='d')
    plt.show()

plt.figure(figsize=(10, 8))
plt.title("Price & Profitability")
sns.boxplot(data=dataset, x='Profitability', y='Price')
plt.show()

plt.figure(figsize=(10, 8))
plt.title("Price & Profitability")
sns.boxplot(data=dataset, x='MenuCategory', y='Price')
plt.show()

plt.figure(figsize=(10, 8))
plt.title("Price & Profitability")
sns.boxplot(data=dataset, x='RestaurantID', y='Price')
plt.show()

plt.figure(figsize=(10, 8))
plt.title("Price & Profitability")
sns.boxplot(data=dataset, x='MenuItem', y='Price')
plt.show()

Q1 = dataset['Price'].quantile(0.25)
Q3 = dataset['Price'].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 *IQR
upper_bound = Q3 + 1.5*IQR

outliers = dataset[(dataset['Price'] < lower_bound) | (dataset['Price'] > upper_bound)]
print(outliers)

# ## Feature Engineering


# Splitting number of Ingredients
dataset['NumIngredientsUsed'] = dataset['Ingredients'].apply(lambda x: len(x.split(',')))

dataset.head()

# Encoding categorical cols
LE = LabelEncoder()

# Label Encoding for Target col
dataset['Profitability'] = LE.fit_transform(dataset['Profitability'])

# One-Hot Encoding for other cat cols
dataset = pd.get_dummies(dataset, columns=['RestaurantID', 'MenuCategory', 'MenuItem', 'Ingredients'])

dataset.shape

dataset.head()

dataset.columns = [col.replace('[', '').replace(']', '').replace("'", '').replace(" ", "_") for col in dataset.columns]

dataset.head()

bool_feat = []

for col in dataset.columns:
    if dataset[col].dtypes == 'bool':
         bool_feat.append(col)

bool_feat

for col in bool_feat:
    dataset[col] = dataset[col].replace({True: 1, False: 0})

dataset.head()

X = dataset.drop('Profitability', axis=1)
y = dataset['Profitability']

print(f"The shape of X  :   {X.shape}")
print(f"The shape of y  :   {y.shape}")

# ## Build and Evaluate Model


trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"The shape of the trainX    :    {trainX.shape}")
print(f"The shape of the testX    :    {testX.shape}")
print(f"The shape of the trainY    :    {trainY.shape}")
print(f"The shape of the testY    :    {testY.shape}")

# Function to evaluate model
def evaluate_model(trueY, predY):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

    # Compute accuracy
    accuracy = accuracy_score(trueY, predY)
    
    # Compute precision
    precision = precision_score(trueY, predY, average='weighted')
    
    # Compute recall
    recall = recall_score(trueY, predY, average='weighted')
    
    # Compute F1 score
    f1 = f1_score(trueY, predY, average='weighted')
    
    # Compute confusion matrix
    conf_matrix = confusion_matrix(trueY, predY)
    
    # Generate classification report
    class_report = classification_report(trueY, predY)
    
    # Print the metrics
    print("Model Evaluation Metrics:")
    print("-" * 30)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(class_report)
    
    # Return the metrics as a dictionary
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report
    }
    
    return metrics

# Function to fit model using GridSearchCV
def fitModel(trainX, testX, trainY, testY, model_name, model_algo, params, CV):
    """
    Fits a machine learning model using GridSearchCV, refits it with the best parameters,
    evaluates it on both training and test data, and stores the evaluation metrics.

    Parameters:
    - trainX: Training features.
    - testX: Test features.
    - trainY: Training labels.
    - testY: Test labels.
    - model_name: Name of the model (string).
    - model_algo: Machine learning algorithm (estimator).
    - params: Dictionary of hyperparameters to tune.
    - CV: Number of cross-validation folds.

    Returns:
    - best_model: The model fitted with the best parameters.
    - best_params: Best hyperparameters found by GridSearchCV.
    - test_metrics: Evaluation metrics on the test set.
    """

    np.random.seed(10)

    print(f"{'-'*75}")
    print("Information")
    print(f"{'-'*75}")
    print(f"Fitting for Model: {model_name}")
    
    grid = GridSearchCV(
        estimator=model_algo,
        param_grid=params,
        scoring='accuracy',
        n_jobs=-1,
        cv=CV,
        verbose=1
    )

    res = grid.fit(trainX, trainY)
    print("Model fitting completed.")
    print(f"{'-'*75}")

    best_params = res.best_params_
    print(f"Found best parameters for model {model_name}: {best_params}")
    print(f"{'-'*75}")

    # Refit the model with the best parameters
    model_algo.set_params(**best_params)
    model_algo.fit(trainX, trainY)
    print(f"Completed refitting the model {model_name} with best parameters.")
    print(f"{'-'*75}")

    # Evaluate on the training data
    print(f"Evaluating {model_name} on the training data.")
    trainY_pred = model_algo.predict(trainX)
    print(f"Evaluation metrics for {model_name} on the training data:")
    train_metrics = evaluate_model(trainY, trainY_pred)
    print(f"{'-'*75}")

    # Evaluate on the test data
    print(f"Evaluating {model_name} on the test data.")
    testY_pred = model_algo.predict(testX)
    print(f"Evaluation metrics for {model_name} on the test data:")
    test_metrics = evaluate_model(testY, testY_pred)
    print(f"{'-'*75}")

    test_accuracy = test_metrics['accuracy']
    if test_accuracy > 0.80:
        model_filename = f"./Models/{model_name}.pkl"
        with open(model_filename, 'wb') as f:
            pickle.dump(model_algo, f)
        print(f"Model {model_name} saved as '{model_name}.pkl'")

    return model_algo, best_params, test_metrics

# Define models and their hyperparameters
models = {
    'RandomForest': {
        'model': RandomForestClassifier(),
        'params': {
            'n_estimators': [10, 50, 100, 500],
            'max_depth': [None, 10, 20, 30]
        }
    },
    'DecisionTree': {
        'model': DecisionTreeClassifier(),
        'params': {
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 10, 20]
        }
    },
    'XGBoost': {
        'model': XGBClassifier(),
        'params': {
            'n_estimators': [10, 50, 100],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2]
        }
    }
}

model_results = {}

for model_name, model_info in models.items():
    model_algo = model_info['model']
    params = model_info['params']

    best_model, best_params, test_metrics = fitModel(trainX,testX,
                                                     trainY,testY, 
                                                     model_name, model_algo, 
                                                     params, CV=5)
    model_results[model_name] = {
        'best_model': best_model,
        'best_params': best_params,
        'test_metrics': test_metrics
    }

for model_name, result in model_results.items():
    print(f"\nModel :  {model_name}")
    print(f"Best Parameters:   :   {result['best_params']}")
    print(f"Test Metrics   :   {result['test_metrics']}")

def createDNNmodel(input_dim, output_dim):

    np.random.seed(10)

    model = Sequential()

    # Hidden Layers
    model = Sequential()
    model.add(Dense(256, input_dim=input_dim, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))

    # Output Layer
    model.add(Dense(output_dim, activation='softmax'))

    # Model Compile
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

input_dim = trainX.shape[1]
output_dim = 3

model = createDNNmodel(input_dim, output_dim)

model.summary()

trainY_encoded = to_categorical(trainY, num_classes=3)
testY_encoded = to_categorical(testY, num_classes=3)

# Callbacks
early_stoping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss')

# Define learning rate sheduler
def lr_shedule(epoch, lr):
    if epoch > 150:
        lr = lr *0.1
    elif epoch > 100:
        lr = lr * 0.5
    
    return lr

lr_sheduler = LearningRateScheduler(lr_shedule)

history = model.fit(trainX, trainY_encoded,
                    epochs=150, batch_size=16,
                    validation_split=0.2, 
                    verbose=1,
                    callbacks=[early_stoping, model_checkpoint, lr_sheduler])

test_loss, test_accuracy = model.evaluate(testX, testY_encoded, verbose=1)

print(f"Expanded Model Test Loss: {test_loss*100:.2f}%")
print(f"Expanded Model Test Accuracy: {test_accuracy*100:.2f}%")

if test_accuracy > 0.80:
    dnn_modelName = './Models/DNN_model.pkl'
    with open (dnn_modelName, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"DNN model saved as DNN_model.pkl")

# Plot training & val accuracy values for the expanded model
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Expanded Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["Train", "Validation"], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Expanded Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["Train", "Validation"], loc="upper left")
plt.show()

