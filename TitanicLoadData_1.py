import pandas as pd
import numpy as npm
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

#----------------------------------------------------------------------------------------------------
#   Function Name   : DisplayInfo
#   Description     : It displays the formatted title
#   Paramenters     : title(str)
#   Return          : None
#   Date            : 14/03/2026
#   Author          : Atharva Samadhan Khairnar
#----------------------------------------------------------------------------------------------------

def DisplayInfo(title):
    print("\n"+"="*110)
    print(title)
    print("\n"+"="*110)

#----------------------------------------------------------------------------------------------------
#   Function Name   : ShowData
#   Description     : It shows basic information about dataset
#   Paramenters     : df
#                     df ->     Pandas data frame object
#                     message
#                     message-> Heading text to display
#   Return          : None
#   Date            : 14/03/2026
#   Author          : Atharva Samadhan Khairnar
#----------------------------------------------------------------------------------------------------

def ShowData(df, message):
    DisplayInfo(message)

    print("First 5 rows of dataset :")
    print(df.head())

    print("\nShape of Dataset ")
    print(df.shape)

    print("\nColumn names : ")
    print(df.columns.tolist())

    print("Missing Values in each column")
    print(df.isnull().sum())

#----------------------------------------------------------------------------------------------------
#   Function Name   : ASKTitanicLogistic
#   Description     : This is main Pipeline controller
#                     It Loads Dataset, shows raw data
#                     It preprocesses the dataset & train the model
#   Paramenters     : Data Path of dataset file
#   Return          : None
#   Date            : 14/03/2026
#   Author          : Atharva Samadhan Khairnar
#----------------------------------------------------------------------------------------------------

def ASKTitanicLogistic(DataPath):
    DisplayInfo("Step 1 : Loading the Dataset")
    df = pd.read_csv(DataPath)

    ShowData(df,"Initial dataset")

#----------------------------------------------------------------------------------------------------
#   Function Name   : main
#   Description     : Starting Point of the application
#   Paramenters     : None
#   Return          : None
#   Date            : 14/03/2026
#   Author          : Atharva Samadhan Khairnar
#----------------------------------------------------------------------------------------------------

def main():
    ASKTitanicLogistic("MarvellousTitanicDataset.csv")

if __name__ == "__main__":
    main()