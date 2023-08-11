# Heart Disease Prediction App using Streamlit

## Overview
This code creates a web-based application using Streamlit to predict and analyze heart disease using a RandomForestClassifier model. The application allows users to explore the dataset, evaluate model performance, and even get predictions for individual patients. Users can also provide feedback on model predictions and contribute doctor evaluations.

## Libraries Used
- `streamlit`: A Python library used for creating interactive web applications.
- `sklearn`: Scikit-learn is used for machine learning functionalities, including model training, evaluation, and metrics.
- `joblib`: Used for loading a pre-trained model.
- `pandas`: For data manipulation and analysis.
- `numpy`: For numerical computations.
- `matplotlib`: For data visualization.
- `scikitplot`: Provides additional plotting functions for scikit-learn.
- `lime`: Used for explaining machine learning models.

## Data Loading and Splitting
1. The heart disease dataset is loaded from the 'Heart_Disease_Prediction.csv' file.
2. Features (`X`) are selected and target (`Y`) is extracted.
3. The dataset is split into training and testing sets using the `train_test_split` function.

## Model Training and Prediction
1. A RandomForestClassifier model with 100 estimators is created and trained on the training data.
2. The model's predictions are made on the test data.
3. The model's predictions are added to the dataset as a new column.

## Streamlit User Interface
1. The Streamlit UI consists of three main pages accessible from the sidebar: "Data Preview", "Model Performance", and "Model Evaluation & Feedback".

### Data Preview Page
- Displays a preview of the dataset.
- Allows users to select specific heart disease categories to filter the data.

### Model Performance Page
- Displays a normalized confusion matrix to evaluate the model's performance.
- Shows a bar plot of feature importances.
- Displays the classification report with precision, recall, F1-score, and support.

### Model Evaluation & Feedback Page
- Allows users to select predicted disease labels and doctor labels for filtering data.
- Displays a filtered dataset based on the selections.
- Provides sliders for adjusting patient data attributes.
- Displays the model's prediction and confidence percentage.
- Utilizes LIME (Local Interpretable Model-Agnostic Explanations) to explain model predictions.
- Allows doctors to evaluate and provide remarks on patient predictions.

## Usage Instructions
1. Install required libraries using: `pip install streamlit scikit-learn joblib pandas numpy matplotlib scikitplot lime`
2. Place the 'Heart_Disease_Prediction.csv' file in the same directory as the script.
3. Run the script using: `streamlit run script_name.py` (replace 'script_name.py' with your script's filename).

## Note
- Ensure the 'Heart_Disease_Prediction.csv' file is correctly formatted with columns matching those used in the script.
- This code assumes that the dataset file, model, and predictions are consistent with the provided code.

This application provides an interactive interface to explore heart disease data, assess model performance, and understand the basis of model predictions while also allowing for medical evaluation and feedback.
