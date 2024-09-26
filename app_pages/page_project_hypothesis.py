import streamlit as st
import matplotlib.pyplot as plt


def page_project_hypothesis_body():
    st.write("## Hypothesis")

    st.info(
        f"1. Infected leaves display distinct visual features, "
        f" such as white or grayish powdery spots and curled edges, "
        f"that differentiate them from healthy leaves.\n\n"
        f"2. Due to these distinct visual features the Machine "
        f"Learning Model with convolutional neural network (CNN) "
        f"should be able to accurately classify unseen data (an image of a "
        f"healthy or infected cherry leaf)."
    )

    st.write("### Conclusion")

    st.success(
        f"1. Infected leaves were found to exhibit clear visual differences "
        f"from healthy ones, with distinct white or grayish powdery spots "
        f"that spread, altering the leaves' colour and texture. "
        f"In contrast, healthy leaves maintained a uniform green "
        f"colour and smooth surface, confirming that the two conditions "
        f"are easily distinguishable for classification. \n\n"
        f"2. The Machine Learning Model is able to predict whether a "
        f"Cherry leaf was healthy or displaying symptoms of powdery mildew "
        f"to an accuracy level above the 97% required by the client."
    )
    st.write("### Validation")
     
    st.warning(
        f"The model successfully identified these differences and learned "
        f"to distinguish between classes, generalizing well to make accurate "
        f"predictions. An effective model trains on a batch of data while "
        f"avoiding overfitting to that specific dataset. \n\n"
        f"By doing so, it can generalize and make reliable predictions on "
        f"new data, as it learns the overall patterns between features "
        f"and labels rather than memorizing the relationships "
        f"from the training set.\n\n"
    )


    st.write(
        f"For additional information, please view the "
        f"[Project README file](https://github.com/bengilbertcodes/"
        f"mildew-detection-in-cherry-leaves#readme).")