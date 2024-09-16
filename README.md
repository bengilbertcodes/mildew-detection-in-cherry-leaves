## How to use this repo

1. Use this template to create your GitHub project repo

1. Log into your cloud IDE with your GitHub account.

1. On your Dashboard, click on the New Workspace button

1. Paste in the URL you copied from GitHub earlier

1. Click Create

1. Wait for the workspace to open. This can take a few minutes.

1. Open a new terminal and `pip3 install -r requirements.txt`

1. Open the jupyter_notebooks directory, and click on the notebook you want to open.

1. Click the kernel button and choose Python Environments.

Note that the kernel says Python 3.8.18 as it inherits from the workspace, so it will be Python-3.8.18 as installed by our template. To confirm this, you can use `! python --version` in a notebook code cell.

## Cloud IDE Reminders

To log into the Heroku toolbelt CLI:

1. Log in to your Heroku account and go to _Account Settings_ in the menu under your avatar.
2. Scroll down to the _API Key_ and click _Reveal_
3. Copy the key
4. In the terminal, run `heroku_config`
5. Paste in your API key when asked

You can now use the `heroku` CLI program - try running `heroku apps` to confirm it works. This API key is unique and private to you, so do not share it. If you accidentally make it public, then you can create a new one with _Regenerate API Key_.

## Dataset Content

- The dataset is sourced from [Kaggle](https://www.kaggle.com/codeinstitute/cherry-leaves). We then created a fictitious user story where predictive analytics can be applied in a real project in the workplace.
- The dataset contains +4 thousand images taken from the client's crop fields. The images show healthy cherry leaves and cherry leaves that have powdery mildew, a fungal disease that affects many plant species. The cherry plantation crop is one of the finest products in their portfolio, and the company is concerned about supplying the market with a compromised quality product.

## Business Requirements

The cherry plantation crop from Farmy & Foods is facing a challenge where their cherry plantations have been presenting powdery mildew. Currently, the process is manual verification if a given cherry tree contains powdery mildew. An employee spends around 30 minutes in each tree, taking a few samples of tree leaves and verifying visually if the leaf tree is healthy or has powdery mildew. If there is powdery mildew, the employee applies a specific compound to kill the fungus. The time spent applying this compound is 1 minute. The company has thousands of cherry trees located on multiple farms across the country. As a result, this manual process is not scalable due to the time spent in the manual process inspection.

To save time in this process, the IT team suggested an ML system that detects instantly, using a leaf tree image, if it is healthy or has powdery mildew. A similar manual process is in place for other crops for detecting pests, and if this initiative is successful, there is a realistic chance to replicate this project for all other crops. The dataset is a collection of cherry leaf images provided by Farmy & Foods, taken from their crops.

- 1 - The client is interested in conducting a study to visually differentiate a healthy cherry leaf from one with powdery mildew.
- 2 - The client is interested in predicting if a cherry leaf is healthy or contains powdery mildew.

## Hypothesis and how to validate?

* We suspect the Cherry Leaves affected by the powdery mildew disease display clear signs/symptoms - usually a pale, powdery growth on the surface of the infected leaves of the plant.
    - Validation will be acheived by conducting an average image study.

## The rationale to map the business requirements to the Data Visualisations and ML tasks

* **Business Requirement 1:** Data Visualisation

    - We will display the "mean" and "standard deviation" images for infected and uninfected cherry leaves.
    - We will display the difference between average infected and uninfected cherry leaves.
    - We will display an image montage for either infecetd or uninfected leaves.

* **Business Requirement 2:** Classification

    - We want to predict if a particular cherry leaf is is or is not infected by powdery mildew disease.
    - We want to build a binary classifier and generate reports.

## ML Business Case

### Cherry Leaf Powdery Mildew Classifier (Clf)
* We want a machine learning (ML) model to predict whether a leaf is infected with powdery mildew or not based on historical image data. This is a binary classification problem where there are two possible outcomes for each image:
    * **Infected:** Leaf displays signs of powdery mildew infection.
    * **Not infected:** Leaf shows no sign of powdery mildew infection.
* It will be a supervised model, and a 2-class, single-label classification model. This means there are two classes (Infected and Not Infected) and each image can only belong to one class; either Infected or Not Infected, but not both at the same time.
* The ideal outcome would be to provide the client with fast and reliable method of detecting powdery mildew infection.
* The model sucess metrics are:
    * Accuracy of 65% or above on the test set.
* The model output is designed as a flag, indicating if the cherry leaf is healthy or contains powdery mildew. Staff at the clients facilities will upload pictures to the app and the prediction will be made on the fly.
* Heuristics: The current diagnostic requires staff to conduct a detailed inspection of each tree, taking samples and visually verifying whether signs of powdery mildew are present. This is time consuming and leaves room for diagnostic inaccuracies due to human error.
* The training data to fit the model comes from [Kaggle](https://www.kaggle.com/datasets/codeinstitute/cherry-leaves). The dataset contains 4208 images - 2104 show healthy cherry leaves, 2104 show leaves containing powdery mildew.

## Dashboard Design
### **Page 1: Project Summary**

- Project Summary

    - General Information

        - Powdery Mildew is a plant disease that causes a powdery growth on the surface of leaves, buds, young shoots, flowers and fruits. 
        - Leaves commonly turn yellow and wither, flowers are distorted or fewer in number, and fruit yield and quality are reduced. [Britanica](https://www.britannica.com/science/powdery-mildew)

    - Project Dataset

        - The cherry_leaves dataset [Kaggle](https://www.kaggle.com/datasets/codeinstitute/cherry-leaves) contains 4208 images (photographs of cherry tree leaves). 2104 images show healthy cherry leaves, 2104 images show leaves containing powdery mildew.

    - Link to Additional Information

        - [Royal Horticultural Society](https://www.rhs.org.uk/disease/powdery-mildews)

    - Business Requirements

        - The client is interested in conducting a study to visually differentiate a healthy cherry leaf from one with powdery mildew.
        - The client is interested in predicting if a cherry leaf is healthy or contains powdery mildew.

### **Page 2: Cherry Tree Leaf Visualiser**

- To Answer Business Requirement 1: The client is interested in conducting a study to visually differentiate a healthy cherry leaf from one with powdery mildew.

    - Checkbox 1 - Display the difference between average and variability as an image.
    - Checkbox 2 - Display the differences between healthy cherry tree leaves and those containing powdery mildew.
    - Image Montage - Display a montage of images from the dataset. Filterable by Healthy/Infected.

### **Page 3: Powdery Mildew Detector**

- To Answer Business Requirement 2: The client is interested in predicting if a cherry leaf is healthy or contains powdery mildew.

    - Link to download a set of images to use for live prediction.
        - [kaggle - cherry_leaves](https://www.kaggle.com/datasets/codeinstitute/cherry-leaves)
    
    - File Uploader Widget
        - Allows the user to upload one or multiple images. 
        - For each image the widget will display the image and a prediction statement, indicating if a cherry leaf is healthy or contains powdery mildew. The probabbiliy associated with this statement will be displayed.
        - A table containing the image name and prediction results.
        - A button allowing user to download the results table.

### **Page 4: Project Hypothesis and Validation**

- Project Hypothesis

    - Describe concolusion
    - Describe validation

### **Page 5: ML Performance Metrics**

- Label Frequencies for Train, Validation and Test sets
    -
- Model History - Accuracy and Losses
    -
- Model Evaluation Result
    -
## Unfixed Bugs

- You will need to mention unfixed bugs and why they were unfixed. This section should include shortcomings of the frameworks or technologies used. Although time can be a significant variable for consideration, paucity of time and difficulty understanding implementation is not a valid reason to leave bugs unfixed.

## Deployment

### Heroku

- The App live link is: `https://YOUR_APP_NAME.herokuapp.com/`
- Set the runtime.txt Python version to a [Heroku-20](https://devcenter.heroku.com/articles/python-support#supported-runtimes) stack currently supported version.
- The project was deployed to Heroku using the following steps.

1. Log in to Heroku and create an App
2. At the Deploy tab, select GitHub as the deployment method.
3. Select your repository name and click Search. Once it is found, click Connect.
4. Select the branch you want to deploy, then click Deploy Branch.
5. The deployment process should happen smoothly if all deployment files are fully functional. Click the button Open App on the top of the page to access your App.
6. If the slug size is too large, then add large files not required for the app to the .slugignore file.

## Main Data Analysis and Machine Learning Libraries

- Here, you should list the libraries used in the project and provide an example(s) of how you used these libraries.

## Credits

### Content

- Code for functions *remove_non_image_file* and *split_train_validation_test_images* taken from Code Institute Course content [Walkthrough 1: Malaria Detector](https://learn.codeinstitute.net/courses/course-v1:code_institute+CI_DA_ML+2021_Q4/courseware/07a3964f7a72407ea3e073542a2955bd/29ae4b4c67ed45a8a97bb9f4dcfa714b/)

### Media

- Additional information on powdery mildew from:
    - [Royal Horticultural Society](https://www.rhs.org.uk/disease/powdery-mildews)
    - [Britanica](https://www.britannica.com/science/powdery-mildew)

## Acknowledgements (optional)

- Thank the people who provided support throughout this project.
