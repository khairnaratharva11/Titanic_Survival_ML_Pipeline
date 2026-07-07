import pandas as pd
import numpy as npm
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

#----------------------------------------------------------------------------------------------------
#   Function Name   : LoadPreserveModel
#   Description     : It is used to preserve model on secondary
#   Paramenters     : filename
#   Return          : None
#   Date            : 14/03/2026
#   Author          : Atharva Samadhan Khairnar
#----------------------------------------------------------------------------------------------------

def LoadPreserveModel(filename):
    
    loaded_model = joblib.load(filename)

    print("Model Successfully Loaded")

    return loaded_model
    
#----------------------------------------------------------------------------------------------------
#   Function Name   : PreserveModel
#   Description     : It is used to preserve model on secondary
#   Paramenters     : model , filename
#   Return          : None
#   Date            : 14/03/2026
#   Author          : Atharva Samadhan Khairnar
#----------------------------------------------------------------------------------------------------

def PreserveModel(model , filename):
    joblib.dump(model, filename)

    print("Model Preserved successfully with name : ", filename)

#----------------------------------------------------------------------------------------------------
#   Function Name   : TrainTitanicModel
#   Description     : It does split X,Y, training data, testing data
#   Paramenters     : df
#   Return          : None
#   Date            : 14/03/2026
#   Author          : Atharva Samadhan Khairnar
#----------------------------------------------------------------------------------------------------

def TrainTitanicModel(df):
    # Split features and labels
    X = df.drop("Survived", axis = 1)
    Y = df["Survived"]

    print("\nFeatures : ")
    print(X.head())
    print("\nLabels : ")
    print(Y.head())

    print("Shape of X : ", X.shape)
    print("Shape of Y : ", Y.shape)

    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2,random_state=42)
    print("X_train Shape : ",X_train.shape)
    print("X_test Shape : ",X_test.shape)
    print("Y_train Shape : ",Y_train.shape)
    print("Y_test Shape : ",Y_test.shape)

    model = LogisticRegression(max_iter=1000)

    model.fit(X_train,Y_train)

    print("Model trained Successfully")

    print("\nIntercept of model : ")
    print(model.intercept_)
    
    print("\nCoefficient of model : ")
    print(model.coef_)
    for feature,coefficient in zip(X.columns, model.coef_[0]):
        print(feature, " : ", coefficient)

    PreserveModel(model,"TitanicModel.pkl")

    loaded_model = LoadPreserveModel("TitanicModel.pkl")

    Y_pred = loaded_model.predict(X_test)

    accuracy = accuracy_score(Y_pred,Y_test)

    print("Accuracy is : ", accuracy)

    cm = confusion_matrix(Y_pred, Y_test)

    print("Confusion matrix is : ")
    print(cm)

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
#   Function Name   : CleanTitanicData
#   Description     : It does preprocessing
#                     It removes unnecessary columns
#                     It handles missing values
#                     It converts text data to numeric format
#                     It does encoding of categorical columns
#   Paramenters     : df -> Pandas dataframe
#   Return          : df -> Clean Pandas dataframe
#   Date            : 14/03/2026
#   Author          : Atharva Samadhan Khairnar
#----------------------------------------------------------------------------------------------------

def CleanTitanicData(df):
    DisplayInfo("Step 2 : Original Data")
    print(df.head())

    #   Remove unnecessary columns
    drop_columns = ["Passengerid","zero","Name","Cabin"]
    existing_columns = [col for col in drop_columns if col in df.columns]

    print("\n Columns to be dropped : ")
    print(existing_columns)

    # Drop the unwanted columns
    df = df.drop(columns = existing_columns)
    DisplayInfo("Step 2 : Data after Column Removal")
    print(df.head())

    # Handle age column
    if "Age" in df.columns:
        print("\nAge Column before filling missing values")
        print(df["Age"].head(10))

        # Invalid Value gets converted as NaN
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce")

        age_median = df["Age"].median()

        print("Median of Age : ",age_median)

        # Replace missing values with median
        df["Age"] = df["Age"].fillna(age_median)

        print("\nAge column after preprocessing")
        print(df["Age"].head(10))

    # Handle Fare Column
    if "Fare" in df.columns:
        print("\nFare Column before filling preprocessing")
        print(df["Fare"].head(10))

        df["Fare"] = pd.to_numeric(df["Fare"], errors="coerce")

        fare_median = df["Fare"].median()

        print("Median of Fare : ",fare_median)

        # Replace missing values with median
        df["Fare"] = df["Fare"].fillna(fare_median)

        print("\nFare column after preprocessing")
        print(df["Fare"].head(10))

    # Handle Embarked Column
    if "Embarked" in df.columns:
        print("\nEmbarked Column before preprocessing")
        print(df["Embarked"].head(10))

        # Convert the data into string
        df["Embarked"] = df["Embarked"].astype(str).str.strip()

        # Replace missing values with median
        df["Embarked"] = df["Embarked"].replace(['nan', 'None', ' '], npm.nan)

        # Get most frequent value
        embarked_mode = df["Embarked"].mode()[0]

        print("Mode of Embarked : ",embarked_mode)

        df["Embarked"] = df["Embarked"].fillna(embarked_mode)

        print("\nEmbarked column after preprocessing")
        print(df["Embarked"].head(10))

    # Handle Sex Column
    if "Sex" in df.columns:
        print("\nSex Column before filling preprocessing")
        print(df["Sex"].head(10))

        df["Sex"] = pd.to_numeric(df["Sex"], errors="coerce")

        print("\nSex column after preprocessing")
        print(df["Sex"].head(10))

    DisplayInfo("Data After preprocessing : ")
    print(df.head())

    print("Missing values after preprocessing")
    print(df.isnull().sum())

    # Encode Embarked Column
    df = pd.get_dummies(df,columns=["Embarked"], drop_first= True)
    print("\nData before encoding")

    print(df.head())

    print("Shape of Dataset : ",df.shape)

    # convert boolean columns into integer
    for col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)

    print("\nData after encoding")

    print(df.head())
    return df

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

    df = CleanTitanicData(df)

    TrainTitanicModel(df)

    print("="*110)

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