import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.image import imread
from src.machine_learning.evaluate_clf import load_test_evaluation


def page_ml_performance_metrics():
    version = 'v5'
    st.info(
        f"This page displays informaiton on how and why the data was split "
        f"for ML training, as well as a review of the ML performance.\n\n"
        f"An explanation is provided for each step."
    )
    st.write("### Images distribution per set and label ")

    labels_distribution = plt.imread(f"outputs/{version}/labels_distribution.png")
    st.image(labels_distribution, caption='Labels Distribution on Train, Validation and Test Sets')

    labels_distribution = plt.imread(f"outputs/{version}/sets_distribution_pie.png")
    st.image(labels_distribution, caption='Sets distribution')

    st.warning(
        f"The cherry_leaves dataset was divided into three subsets:\n"
        f"* **Train Set** (70% of the dataset): This subset is used to train "
        f"the model. The model learns to generalize and make predictions on "
        f"new, unseen data by fitting to this set.\n"
        f"* **Validation Set** (10% of the dataset): This subset helps "
        f"fine-tune the model during training. After each epoch (a full "
        f"pass of the training data through the model), the validation "
        f"set is used to adjust and improve performance.\n"
        f"* **Test Set** (20% of the dataset): This subset is used to "
        f"evaluate the final accuracy of the model once training is complete. "
        f"It consists of data the model has never encountered, providing "
        f"an unbiased assessment of its performance."
        )

    st.write("---")

    st.write("### Model Performance")

    model_clf = plt.imread(f"outputs/{version}/clf_report.png")
    st.image(model_clf, caption='Classification Report')  

    st.warning(
        f"**Classification Report**\n\n"
        f"* **Precision**: The percentage of correct positive predictions, "
        f"calculated as the ratio of true positives to the sum of true "
        f"positives and false positives. It measures the accuracy of the "
        f"positive predictions made by the model.\n"
        f"* **Recall**: The percentage of actual positive cases that the "
        f"model successfully detects, calculated as the ratio of true "
        f"positives to the sum of true positives and false negatives. It "
        f"reflects the model's ability to identify all relevant cases.\n"
        f"* **F1 Score**: A balanced measure of a model's precision and "
        f"recall, represented as the weighted harmonic mean of the two. "
        f"The F1 score ranges from 0.0 to 1.0, with 1.0 being the best "
        f"performance.\n"
        f"* **Support**: The total number of actual occurrences of a class "
        f"in the dataset, providing context for the model's performance "
        f"relative to the size of each class."
        )

    model_roc = plt.imread(f"outputs/{version}/roccurve.png")
    st.image(model_roc, caption='ROC Curve')

    st.warning(
        f"**ROC Curve**\n\n"
        f"The ROC curve (Receiver Operating Characteristic curve) is a "
        f"performance evaluation metric that assesses a model's ability "
        f"to distinguish between classes through accurate predictions. "
        f"It is created by plotting the **True Positive Rate (TPR)** "
        f"against the **False Positive Rate (FPR)**.\n"
        f"* **True Positive Rate (TPR)**: The proportion of correctly "
        f"predicted positive observations (e.g., the model predicts a "
        f"leaf is healthy, and it is indeed healthy).\n"
        f"* **False Positive Rate (FPR)**: The proportion of incorrectly "
        f"predicted positive observations (e.g., the model predicts a leaf "
        f"is healthy, but it is actually affected).\n\n"
        f"A higher TPR and lower FPR indicate a model's stronger capability "
        f"to separate classes effectively."
        )

    model_cm = plt.imread(f"outputs/{version}/confusion_matrix.png")
    st.image(model_cm, caption='Confusion Matrix')

    st.warning(
        f"**Confusion Matrix**\n\n"
        f"A Confusion Matrix is a performance evaluation tool for classifiers,"
        f" presented as a table that displays four distinct combinations "
        f"of predicted and actual outcomes.\n"
        f"* **True Positive (TP) / True Negative (TN)**: The model's "
        f"predictions align with the actual values.\n"
        f"* **False Positive (FP) / False Negative (FN)**: The model's "
        f"predictions contradict the actual values (e.g., predicting "
        f"a leaf is infected when it's actually healthy).\n\n"
        f"An effective model aims to achieve high TP and TN rates while "
        f"minimizing FP and FN occurrences."
        )

    model_perf = plt.imread(f"outputs/{version}/model_history.png")
    st.image(model_perf, caption='Model Performance')  

    st.warning(
        f"**Model Performance**\n\n"
        f"* **Loss** represents the cumulative errors made for each example "
        f"in the training set (referred to as loss) or the validation set "
        f"(referred to as val_loss). \nThe loss value indicates the model's "
        f"performance after each optimization iteration, revealing how well "
        f"or poorly the model is learning.\n"
        f"* **Accuracy** measures the correctness of the model's predictions "
        f"(accuracy) in relation to the actual data (val_acc)."
        f"A good model demonstrates strong performance on unseen data, "
        f"signifying its ability to generalize rather than being overly "
        f"fitted to the training dataset."
        )

    st.write("### Generalised Performance on Test Set")
    st.dataframe(pd.DataFrame(load_test_evaluation(version), index=['Loss', 'Accuracy']))
    
    st.write(
        f"For additional information, please view the "
        f"[Project README file]"
        f"(https://github.com/bengilbertcodes/"
        f"mildew-detection-in-cherry-leaves#readme)."
        )
    