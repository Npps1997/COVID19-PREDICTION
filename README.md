# COVID-19 Test Prediction Project
**Author:** Neeraj Kumar Paikra

**Link:** https://covid19-prediction-4kpqw2ce2ftnppwshzkhsm.streamlit.app/

Welcome to the COVID-19 Test Predictor, an innovative project leveraging machine learning to predict COVID-19 test results. This tool is designed to assist in early identification and intervention, contributing to public health efforts.



![pic11](https://github.com/Npps1997/COVID19-PREDICTION/assets/96871890/20f47772-27a5-4778-a8c9-9ada79023b8e)


## Table of Contents
- [Overview](#overview)
- [Quick Start](#quick-start)
- [Website Screenshots](#website-screenshots)
- [Project Structure](#project-structure)
- [Features](#features)
- [Data Preprocessing](#data-preprocessing)
- [Model Performance](#model-performance)
- [Evaluation And Results](#evaluation-and-results)
- [ROC Curve Analysis](#roc-curve-analysis)


## Overview
This project aims to predict COVID-19 test results using machine learning techniques. By analyzing various features and data related to individuals, the goal is to build a predictive model that can assist in identifying potential COVID-19 cases.


## Quick Start
Get up and running quickly:

```bash
git clone https://github.com/Npps1997/COVID19-PREDICTION.git
cd COVID19-Test-Predictor
pip install -r requirements.txt
streamlit run app.py
```


## Website Screenshots
Explore the user-friendly interface of our COVID-19 Test Predictor website:

### Dashboard
![Homepage](https://github.com/Npps1997/COVID19-PREDICTION/assets/96871890/4aeecd6d-abd8-4be0-b7e8-8603e7bb9c8f)

### Prediction Results
![Prediction Page and Results](https://github.com/Npps1997/COVID19-PREDICTION/assets/96871890/fcdae84f-f863-4401-9872-18ee7d25a885)


## Project Structure

Project is organized into the following directories:

### 1. `artifacts`

   - Contains artifacts and hyperparameter information.
   - Example content:
     ```
     artifacts/
     ├── hyperparameter
     └── model
         └── done
     ```

### 2. `catboost_info`

   - Additional information related to the CatBoost model.
   - Example content:
     ```
     catboost_info/
     └── hyperparameter
     ```

### 3. `notebook/EDA`

   - Exploratory Data Analysis notebooks.
   - Example content:
     ```
     notebook/
     └── EDA
         └── your_notebook.ipynb
     ```

### 4. `src/Prediction pipeline`

   - Source code for the prediction pipeline.
   - Example content:
     ```
     src/
     └── Prediction pipeline
         └── predict.py
     ```

### 5. `templates`

   - Deployment configuration templates.
   - Example content:
     ```
     templates/
     └── Deployment Config
         └── your_config_file.yaml
     ```

### 6. `.gitignore`

   - Gitignore file to exclude certain files or directories from version control.
   - Example content:
     ```
     .gitignore
     ```

### 7. `app.py`

   - Main application file.
   - Example content:
     ```
     app.py
     ```

### 8. `pic11.jpg`

   - Image file used in the README.
   - Example content:
     ```
     pic11.jpg
     ```

### 9. `requirements.txt`

   - List of project dependencies.
   - Example content:
     ```
     requirements.txt
     ```

### 10. `setup.py`

   - Setup script for packaging your project.
   - Example content:
     ```
     setup.py
     ```

### 11. `README.md`

   - Documentation file providing an overview of the project.
   - Example content:
     ```
     README.md
     ```


## Features

Our prediction model considers various features to make accurate predictions. Here are the key features included:

### A. Basic Information:

| Feature                        | Description                                       |
| ------------------------------ | ------------------------------------------------- |
| ID (Individual ID)             | An identifier for each individual.                |
| Sex (male/female)              | Gender information.                               |
| Age ≥60 above years (true/false) | A boolean value indicating whether the individual is 60 years or older. |
| Test Date (date when tested for COVID) | The date when the individual was tested for COVID-19. |

### B. Symptoms:

| Feature                        | Description                                       |
| ------------------------------ | ------------------------------------------------- |
| Cough (true/false)             | Indicates whether the individual has a cough.     |
| Fever (true/false)             | Indicates whether the individual has a fever.     |
| Sore Throat (true/false)       | Indicates whether the individual has a sore throat. |
| Shortness of Breath (true/false)| Indicates whether the individual experiences shortness of breath. |
| Headache (true/false)          | Indicates whether the individual has a headache.  |

### C. Other Information:

| Feature                        | Description                                       |
| ------------------------------ | ------------------------------------------------- |
| Known Contact with an Individual Confirmed to have COVID-19 (true/false) | Specifies whether the individual had contact with someone confirmed to have COVID-19. |

### D. COVID Report:

| Feature                        | Description                                       |
| ------------------------------ | ------------------------------------------------- |
| Corona Positive or Negative    | The predicted result of the COVID-19 test, indicating whether the individual is likely positive or negative. |

These features contribute to the model's ability to make accurate predictions about COVID-19 test results. The combination of basic information, symptoms, and additional details enhances the precision of our predictions.


## Data Preprocessing

In this section, we detail the steps taken to preprocess the dataset, focusing on addressing data inconsistencies and handling missing values.

### Replaced 'None' with Null Values

We identified instances where 'None' values were present in the dataset and replaced them with null values for consistency and ease of handling.

![image](https://github.com/Npps1997/COVID19-PREDICTION/assets/96871890/f4f71b2e-e178-40cb-b3df-33990e96f1b6)


### Handled Missing Data in 'Age_60_above' and 'Sex'

We implemented strategies to address missing data in the 'Age_60_above' and 'Sex' columns to ensure completeness in our dataset.

![image](https://github.com/Npps1997/COVID19-PREDICTION/assets/96871890/0222943a-1faa-4ef4-9058-2c880712451c)


### Handling Inconsistencies in the Target Column ('Corona') and Other Columns

To enhance the quality of our dataset, we addressed inconsistencies in the target column ('Corona') and other relevant columns.

![image](https://github.com/Npps1997/COVID19-PREDICTION/assets/96871890/2b5b56a8-30f5-4919-a941-b1674cac4b4b)


### Label Encoding for Binary Variables

Binary variables were subjected to label encoding to convert them into a format suitable for machine learning algorithms.

![image](https://github.com/Npps1997/COVID19-PREDICTION/assets/96871890/7a65e30f-b16d-464f-9bb8-d6f1d1256848)


### One-Hot Encoding for Nominal Variables

Nominal variables underwent one-hot encoding to transform them into a numerical format while preserving their categorical nature.

![image](https://github.com/Npps1997/COVID19-PREDICTION/assets/96871890/2e67740f-66c4-4961-8a2a-eb05a716c487)


## Model Performance

| Model                     | Training Accuracy | Training Precision | Training Recall | Training F1 Score | Test Accuracy | Test Precision | Test Recall | Test F1 Score |
|---------------------------|-------------------|---------------------|------------------|-------------------|---------------|-----------------|-------------|--------------|
| Logistic Regression        | 0.9664            | 0.9641              | 0.9664           | 0.9650            | 0.9666        | 0.9645          | 0.9666      | 0.9653       |
| Random Forest Classifier   | 0.9677            | 0.9646              | 0.9677           | 0.9648            | 0.9682        | 0.9652          | 0.9682      | 0.9655       |
| K-Neighbors Classifier    | 0.9649            | 0.9616              | 0.9649           | 0.9625            | 0.9653        | 0.9621          | 0.9653      | 0.9630       |
| Decision Tree Classifier   | 0.9677            | 0.9646              | 0.9677           | 0.9648            | 0.9682        | 0.9652          | 0.9682      | 0.9655       |
| Support Vector Classifier  | 0.9677            | 0.9646              | 0.9677           | 0.9648            | 0.9682        | 0.9652          | 0.9682      | 0.9655       |
| Gaussian Naive Bayes       | 0.9665            | 0.9648              | 0.9665           | 0.9655            | 0.9666        | 0.9651          | 0.9666      | 0.9657       |
| AdaBoost                   | 0.9591            | 0.9531              | 0.9591           | 0.9537            | 0.9592        | 0.9533          | 0.9592      | 0.9542       |
| XGBoost                   | 0.9677            | 0.9646              | 0.9677           | 0.9648            | 0.9682        | 0.9652          | 0.9682      | 0.9655       |


## Evaluation And Results

Here is a comparison of the number of False Negatives for each model:

| Model Name                  | False Negatives |
|-----------------------------|-----------------|
| Gaussian Naive Bayes        | 1072            |
| Logistic Regression         | 1142            |
| K-Neighbors Classifier      | 1309            |
| Random Forest Classifier    | 1317            |
| Decision Tree Classifier    | 1317            |
| Support Vector Classifier   | 1317            |
| XGBoost                     | 1317            |
| AdaBoost                    | 1723            |

The table above indicates the number of instances where the model predicted a negative outcome (e.g., no COVID-19) when the actual outcome was positive. Notably, Gaussian Naive Bayes outperforms other models with fewer** False Negatives (1072)**, showcasing its effectiveness in minimizing the instances of false negatives compared to alternative models.


## ROC Curve Analysis

![ROC Curve](https://github.com/Npps1997/COVID19-PREDICTION/assets/96871890/4204af14-f25d-4311-af86-947ce4de26b8)

An **AUC value of 0.88 (or 88%)** suggests that the model is effective at distinguishing between the two classes, with a high probability of ranking a randomly chosen positive instance higher than a randomly chosen negative instance.


**Thanks for watching!!**
