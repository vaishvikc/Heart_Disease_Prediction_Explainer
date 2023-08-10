import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from joblib import load

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scikitplot as skplt
from lime import lime_tabular

st.set_page_config(layout="wide")

# Load Data
data = pd.read_csv('Heart_Disease_Prediction.csv')
Tcolumns=['Age', 'Sex', 'Chest pain type', 'BP', 'Cholesterol', 'FBS over 120',
       'EKG results', 'Max HR', 'Exercise angina', 'ST depression',
       'Slope of ST', 'Number of vessels fluro', 'Thallium']

X_train, X_test, Y_train, Y_test = train_test_split(data[Tcolumns], data['Heart Disease'], train_size=0.8, random_state=123)

# Load Model
rf_classif = RandomForestClassifier(n_estimators=100, random_state=100)
rf_classif.fit(X_train, Y_train)

Y_test_preds = rf_classif.predict(X_test)
data['Heart Disease']=rf_classif.predict(data[Tcolumns])
# Sidebar for navigation
st.sidebar.title("Heart Disease :red[Prediction]: :anatomical_heart: ")
st.sidebar.markdown("Predictions for Heart disease using RandomForestClassifier ")

menu = ["Data Preview", "Model Performance", "Model Evaluation & Feedback"]
choice = st.sidebar.radio("Select a page", menu)

if choice == "Data Preview":
    st.header("Dataset Preview")
    selected_categories = st.multiselect("Select Heart Disease Categories", data['Heart Disease'].unique())
    filtered_data = data[data['Heart Disease'].isin(selected_categories)] if selected_categories else data
    
    st.write(filtered_data)

elif choice == "Model Performance":
    st.header("Confusion Matrix | Feature Importances")
    col1, col2 = st.columns(2)
    with col1:
        conf_mat_fig = plt.figure(figsize=(6, 6))
        ax1 = conf_mat_fig.add_subplot(111)
        skplt.metrics.plot_confusion_matrix(Y_test, Y_test_preds, ax=ax1, normalize=True)
        st.pyplot(conf_mat_fig, use_container_width=True)

    with col2:
        feat_imp_fig = plt.figure(figsize=(6, 6))
        ax1 = feat_imp_fig.add_subplot(111)
        skplt.estimators.plot_feature_importances(rf_classif, feature_names=Tcolumns, ax=ax1, x_tick_rotation=90)
        st.pyplot(feat_imp_fig, use_container_width=True)

    st.divider()
    st.header("Classification Report")
    st.code(classification_report(Y_test, Y_test_preds))

elif choice == "Model Evaluation & Feedback":
    st.header("Model Prediction and Explanation")

    col1, col2 = st.columns(2)
    with col1:
        selected_disease_labels = st.multiselect("Select Heart Disease Labels", data['Heart Disease'].unique())
    with col2:
        selected_doctor_labels = st.multiselect("Select Doctor Labels", ['Absence', 'Presence','not Evaluted'])
    st.write('Please select both')
    filtered_data = data[
        (data['Heart Disease'].isin(selected_disease_labels)) &
        (data['Doctors label'].isin(selected_doctor_labels))
    ] if selected_disease_labels or selected_doctor_labels else data

    st.dataframe(filtered_data, height=200)
    
    st.divider()
    selected_id = st.selectbox("Select a Patient ID", data['Patient_ID'])
    
    selected_index = data[data['Patient_ID'] == selected_id].index[0]

    sliders = []
    col1, col2 = st.columns(2)
    with col1:
        for ingredient in Tcolumns:
            ing_slider = st.slider(label=ingredient, min_value=float(data[ingredient].min()), max_value=float(data[ingredient].max()),value=data.loc[selected_index, ingredient].astype(float),step=0.1)
            sliders.append(ing_slider)

    with col2:
        col1, col2 = st.columns(2, gap="medium")

        tmap = {'Absence': 0, 'Presence': 1}

        prediction = rf_classif.predict([sliders])

        with col1:
            st.markdown("### Model Prediction: **{}**".format(prediction[0]))

        probs = rf_classif.predict_proba([sliders])
        probability = probs[0][tmap[prediction[0]]]

        with col2:
            st.metric(label="Model Confidence", value="{:.2f} %".format(probability * 100), delta="{:.2f} %".format((probability - 0.5) * 100))

        explainer = lime_tabular.LimeTabularExplainer(np.array(X_train), mode="classification", class_names=list(data['Heart Disease'].unique())[::-1], feature_names=X_train.columns)
        explanation = explainer.explain_instance(np.array(sliders), rf_classif.predict_proba, num_features=len(Tcolumns), top_labels=3)
        interpretation_fig = explanation.as_pyplot_figure(label=tmap[prediction[0]])
        st.pyplot(interpretation_fig, use_container_width=True)


        with st.form("Doctor's Evalution for Patient ID {}".format(selected_index)):
            doctor_opinion = st.selectbox("Doctor's Opinion", ['Absence', 'Presence','not Evaluted'])
            doctor_remark = st.text_area("Doctor's Remark")

            submit_button = st.form_submit_button("Submit")

        if submit_button:
            data.loc[data['Patient_ID'] == selected_index, 'Doctors label'] = doctor_opinion
            if len(doctor_remark)==0:
                data.loc[data['Patient_ID'] == selected_index, 'Doctors Remark'] = 'No Remark'
            else: 
                data.loc[data['Patient_ID'] == selected_index, 'Doctors Remark'] = doctor_remark
            data.to_csv('Heart_Disease_Prediction.csv',index= False)
            st.toast('Updated!!')
            st.experimental_rerun()
