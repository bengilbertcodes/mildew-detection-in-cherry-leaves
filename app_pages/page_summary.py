import streamlit as st
import matplotlib.pyplot as plt


def page_summary_body():

    st.write("### Quick Project Summary")

    st.info(
        f"The cherry plantation crop from Farmy & Foods is facing a challenge "
        f"where their cherry plantations have been presenting powdery mildew."
        f"Currently, the process is manual verification if a given cherry "
        f"tree contains powdery mildew."
        f"An employee spends around 30 minutes in each tree, taking a few "
        f"samples of tree leaves and verifying visually if the leaf tree is "
        f"healthy or has powdery mildew." 
        f"If there is powdery mildew, the employee applies a specific compound"
        f" to kill the fungus."
        f"The time spent applying this compound is 1 minute." 
        f"The company has thousands of cherry trees located on multiple farms "
        f"across the country."
        f"As a result, this manual process is not scalable due to the time "
        f"spent in the manual process inspection.\n\n"

        f"To save time in this process, the IT team suggested an ML system "
        f"that detects instantly, using a leaf tree image, if it is healthy "
        f"or has powdery mildew."
        f"A similar manual process is in place for other crops for detecting "
        f"pests," 
        f" and if this initiative is successful, there is a realistic chance "
        f"to replicate this project for all other crops."
        f"The dataset is a collection of cherry leaf images provided by "
        f"Farmy & Foods, taken from their crops."
    )

    st.info(
        f"**Powdery Mildews**\n\n"
        f"Powdery mildew is a parasitic fungal disease caused by 'Podosphaera "
        f"clandestina' that affects cherry trees."
        f"As the fungus spreads, it forms a layer of mildew composed of "
        f"numerous spores on the upper surfaces of the leaves.\n\n"
        f"The disease is particularly harmful to new growth, stunting the "
        f"plant's development and potentially infecting the fruit, leading "
        f"to significant crop losses.\n\n"
        f"To assess the presence of the disease, both healthy and infected "
        f"leaves were collected for examination."
        f"\nThe key visual indicators of infection include:\n\n"
        f"* Light-green/yellow, circular lesions on either side of the leaf.\n"
        f"* A subtle white, cotton-like growth that later develops in the "
        f"affected areas, spreading across the leaves and fruit, "
        f"ultimately reducing yield and quality.\n"
        f"* Leaf edges may curl upwards, exposing the powdery growth."
        f" \n\n")
    
    st.info(
        f"**Links for Further Information About Powdery Mildew**\n\n"
        f"[Royal Horticultural Society]"
        f"(https://www.rhs.org.uk/disease/powdery-mildews)"
        f"\n"
        f"\n[Britanica](https://www.britannica.com/science/powdery-mildew)"
    )

    st.warning(
        f"**Project Dataset**\n\n"
        f"The [cherry_leaves dataset]"
        f"(https://www.kaggle.com/datasets/codeinstitute/cherry-leaves) "
        f"from Kaggle contains 4208 images (photos of cherry tree leaves)."
        f"\n"
        f"2104 images show healthy cherry leaves, 2104 images show "
        f"leaves containing powdery mildew."
        )

    st.success(
        f"**Business Requirements**\n\n"
        f"1 - The client is interested in conducting a study to "
        f"visually differentiate a healthy cherry leaf from one with "
        f"powdery mildew.\n\n"
        f"2 - The client is interested in predicting if a cherry leaf "
        f"is healthy or contains powdery mildew. \n\n"
        f"3 - The client is interested in generating and downloading reports."
        )

    st.write(
        f"For additional information, please view the "
        f"[Project README file]"
        f"(https://github.com/bengilbertcodes/"
        f"mildew-detection-in-cherry-leaves#readme).")