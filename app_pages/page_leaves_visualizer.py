import streamlit as st
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread

import itertools
import random

def page_leaves_visualizer_body():
    st.write("### Cherry Leaf Visualizer")
    st.info(
        f"A study that visually differentiates a cherry leaf affected by powdery mildew from a healthy one.")

    st.write(
        f"For additional information, please view the "
        f"[Project README file](https://github.com/bengilbertcodes/mildew-detection-in-cherry-leaves#readme).")

    st.warning(
        f"We suspect that cherry leaves affected by powdery mildew display distinct visual signs, with the first symptom typically being a light-green/yellow, "
        f"circular lesion on either side of the leaf, followed by a subtle white, cotton-like growth in the infected areas.\n"
        f"To translate these characteristics into machine learning terms, the images must be preprocessed "
        f"before being fed into the model for optimal feature extraction and training.\n\n"
        f"When working with an image dataset, it's crucial to normalize the images before training a neural network."
        f" This process involves adjusting the pixel values based on the dataset's mean and standard deviation, "
        f"which are calculated using a formula that accounts for the image's properties. "
        f"Normalization ensures consistent input for the model, improving training efficiency and accuracy."
    )
    
    version = 'v5'
    if st.checkbox("Difference between average and variability image"):
      
      avg_powdery_mildew = plt.imread(f"outputs/{version}/avg_var_powdery_mildew.png")
      avg_uninfected = plt.imread(f"outputs/{version}/avg_var_healthy.png")

      st.warning(
        f"**Observations** \n"
        f"* Clear patterns that could allow for an intuitive differentiation  "
        f"between healthy and affected leaves are not apparent in the average and variability images.\n" 
        f"* Subtle changes in colour are visible, the average image for powdery mildew"
        f"having a slightly lighter colour pallete\n"
        f"* The edges of healthy leaves produce a stronger variablity image."
      )

      st.image(avg_powdery_mildew, caption='Affected leaf - Average and Variability')
      st.image(avg_uninfected, caption='healthy leaf - Average and Variability')
      st.write("---")

    if st.checkbox("Differences between average infected and average healthy leaves"):
          diff_between_avgs = plt.imread(f"outputs/{version}/avg_diff.png")

          st.image(diff_between_avgs, caption='Difference between average images')

    if st.checkbox("Image Montage"): 
      st.write("To refresh the montage, select label and click on the 'Create Montage' button")
      my_data_dir = 'inputs/cherry-leaves'
      labels = os.listdir(my_data_dir+ '/validation')
      label_to_display = st.selectbox(label="Select label", options=labels, index=0)
      if st.button("Create Montage"):      
        image_montage(dir_path= my_data_dir + '/validation',
                      label_to_display=label_to_display,
                      nrows=8, ncols=3, figsize=(10,25))
      st.write("---")


def image_montage(dir_path, label_to_display, nrows, ncols, figsize=(15,10)):
  sns.set_style("white")
  labels = os.listdir(dir_path)

  # subset the class you are interested to display
  if label_to_display in labels:

    # checks if your montage space is greater than subset size
    # how many images in that folder
    images_list = os.listdir(dir_path+'/'+ label_to_display)
    if nrows * ncols < len(images_list):
      img_idx = random.sample(images_list, nrows * ncols)
    else:
      print(
          f"Decrease nrows or ncols to create your montage. \n"
          f"There are {len(images_list)} in your subset. "
          f"You requested a montage with {nrows * ncols} spaces")
      return
    

    # create list of axes indices based on nrows and ncols
    list_rows= range(0,nrows)
    list_cols= range(0,ncols)
    plot_idx = list(itertools.product(list_rows,list_cols))


    # create a Figure and display images
    fig, axes = plt.subplots(nrows=nrows,ncols=ncols, figsize=figsize)
    for x in range(0,nrows*ncols):
      img = imread(dir_path + '/' + label_to_display + '/' + img_idx[x])
      img_shape = img.shape
      axes[plot_idx[x][0], plot_idx[x][1]].imshow(img)
      axes[plot_idx[x][0], plot_idx[x][1]].set_title(f"Width {img_shape[1]}px x Height {img_shape[0]}px")
      axes[plot_idx[x][0], plot_idx[x][1]].set_xticks([])
      axes[plot_idx[x][0], plot_idx[x][1]].set_yticks([])
    plt.tight_layout()
    
    st.pyplot(fig=fig)
    # plt.show()


  else:
    print("The label you selected doesn't exist.")
    print(f"The existing options are: {labels}")