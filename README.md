
# **Predict Restaurant Menu Items Profitability**

[!Restaurant Menu Items Profitability](./Designer.png)

## **Background:**
In the highly competitive restaurant industry, understanding the profitability of menu items is crucial for maintaining a successful business. Profitability can be influenced by various factors such as the ingredients used, the price of the item, the restaurant category, and customer preferences. Efficiently predicting which menu items are likely to be more profitable can help restaurant managers make informed decisions about menu design, pricing strategies, and inventory management.

## **Objective:**
The objective of this project is to develop a predictive model that can classify the profitability of restaurant menu items into categories such as **Low, Medium, and High**. This model will leverage historical data on menu items, including their prices, ingredients, and other relevant attributes, to make accurate profitability predictions.

## **Data:**
The dataset consists of 1000 entries, each representing a menu item from various restaurants. The features of the dataset are as follows:
- **RestaurantID**: Unique identifier for the restaurant.
- **MenuCategory**: Category of the menu item (e.g., Appetizers, Main Course, Desserts).
- **MenuItem**: Name of the menu item.
- **Ingredients**: List of ingredients used in the menu item.
- **Price**: Price of the menu item.
- **Profitability**: Profitability category of the menu item (Low, Medium, High).

## **Tasks:**

### **1. Data Exploration and Preprocessing:**
- Conduct exploratory data analysis (EDA) to understand the distribution and relationships within the data.
- Handle missing values, if any, and encode categorical features appropriately.
- Engineer new features that may help in improving the model's performance, such as the number of ingredients used or specific ingredient indicators.

### **2. Model Development:**
- Develop several machine learning models (e.g., RandomForestClassifier, DecisionTreeClassifier, XGBClassifier) to predict the profitability of menu items.
- Perform hyperparameter tuning using GridSearchCV to find the best parameters for each model.

### **3. Model Evaluation:**
- Evaluate the performance of each model using metrics such as accuracy and F1 score.
- Compare the models and select the best-performing one based on the evaluation metrics.

### **4. Deep Learning Model:**
- Develop a Deep Neural Network (DNN) model with multiple layers to predict profitability.
- Train and evaluate the DNN model, and compare its performance with traditional machine learning models.

### **5. Model Saving and Deployment:**
- Save the best models (with accuracy greater than 80%) in the `models` directory for future use.
- Document the model training and evaluation process, and provide recommendations for deploying the model in a production environment.

## **Expected Outcome:**
By the end of this project, we aim to have a robust predictive model that accurately classifies the profitability of restaurant menu items. This model can be used by restaurant managers to optimize their menus, set competitive prices, and ultimately enhance their profitability.

## **Impact:**
The implementation of this predictive model will enable restaurants to make data-driven decisions, leading to improved financial performance and customer satisfaction. By understanding which menu items are more profitable, restaurants can focus on promoting and improving these items, thereby maximizing their overall profitability.

## **Acknowledgements:**
We would like to extend our gratitude to the following resources and tools, which made this project possible:

- **[TensorFlow](https://www.tensorflow.org/)**: For providing a robust framework for developing and training deep learning models.
- **[Scikit-learn (sklearn)](https://scikit-learn.org/stable/)**: For offering comprehensive tools for data analysis, preprocessing, and machine learning.
- **[Keras](https://keras.io/)**: For its easy-to-use API that facilitated the creation and training of our deep neural network models.
- **[Kaggle](https://www.kaggle.com/datasets/rabieelkharoua/predict-restaurant-menu-items-profitability)**: For providing the dataset that was crucial for training and evaluating our models.

Their contributions have been invaluable in achieving the objectives of this project.
