# Titanic Survival Prediction: End-to-End ML Pipeline

## 📌 Overview
This repository contains a comprehensive, incremental machine learning project focused on predicting passenger survival on the Titanic. Using **Logistic Regression**, the project is structured as an end-to-end pipeline. It meticulously details every stage of the machine learning lifecycle: data ingestion, data cleaning, feature encoding, model training, and finally, model preservation and loading for future use.

## 📊 The Dataset
The project utilizes the `MarvellousTitanicDataset.csv`, a variation of the classic Titanic dataset. It includes passenger information such as:
* **Age:** Age of the passenger
* **Sex:** Gender of the passenger
* **Pclass:** Ticket class (1st, 2nd, 3rd)
* **SibSp & Parch:** Number of siblings/spouses or parents/children aboard
* **Survived:** Survival status (0 = No, 1 = Yes) - *Target Variable*

## 🚀 Incremental Approach & Project Structure
The code is divided into six distinct scripts, showcasing a modular, software-engineering approach to machine learning:

* **`TitanicLoadData_1.py` (Data Ingestion):** The first step in the pipeline. It loads the CSV, displays basic dataset shapes, and identifies missing values.
* **`TitanicPreprocessing_2.py` (Data Cleaning):** Introduces the `CleanTitanicData` function. This script handles missing data (filling null Ages with the median) and drops irrelevant columns (like `Passengerid`, `zero`, and `Fare`) to streamline the dataset.
* **`TitanicEncoding_3.py` (Feature Engineering):** Expands the cleaning function to handle categorical variables. It uses `pd.get_dummies` to apply one-hot encoding to the `Sex` and `Pclass` columns, ensuring the data is strictly numerical for the algorithm.
* **`TitanicTrainModel_4.py` (Model Building):** The core machine learning script. It splits the preprocessed data into training and testing sets and trains a `LogisticRegression` classifier.
* **`TitanicPreserveModel_5.py` (Serialization):** Demonstrates production-readiness. It introduces the `PreserveModel` function, utilizing the `joblib` library to save the trained Logistic Regression model to the local disk.
* **`TitanicTestModel_6.py` (Deserialization):** Completes the pipeline by introducing the `LoadPreserveModel` function, proving that the saved model can be successfully loaded back into memory for future inferences without retraining.

## 🛠️ Technologies Used
* **Language:** Python 3
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (`LogisticRegression`, `train_test_split`)
* **Serialization:** Joblib

## ⚙️ How to Run
1. Clone the repository to your local machine.
2. Ensure you have the required libraries installed: `pip install pandas numpy scikit-learn joblib`.
3. Make sure the `MarvellousTitanicDataset.csv` dataset is in the same directory as the scripts.
4. Run the scripts sequentially to observe the pipeline evolution, or run the later scripts to see the full process in action:
   ```bash
   python TitanicTrainModel_4.py
