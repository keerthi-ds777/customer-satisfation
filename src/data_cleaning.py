import mlflow
import pandas as pd 
import numpy as np
from logging import info,error
import logging
from zenml import step
from pydantic import BaseModel
from typing import Union, Tuple,Annotated

class IngestData:
    
    """
    Ingesting data from a CSV file.
    """
    
    def __init__(self,data_path:str):
        """initialize the class with the path to the data"""
        
        self.path=data_path
        info("IngestData class initialized")
     
     
    
    
    def get_data(self)-> pd.DataFrame:
       """making the initialized data to a pandas dataframe"""

       try: #Data Ingesting
        import pandas as pd
        df = pd.read_csv(self.path)
        cols_dont_needed =['order_id','order_delivered_carrier_date','order_delivered_customer_date','order_estimated_delivery_date','customer_id','order_status','order_purchase_timestamp','customer_unique_id','order_approved_at','product_id','seller_id','customer_zip_code_prefix','shipping_limit_date','review_comment_message']
        df.drop(columns=cols_dont_needed,inplace=True)
        info("Data Ingested successfully")
        return df
       except Exception as e:
            logging.error(f'Only csv file is valid {e}')
            raise e
    


from logging import error, info

class DataCleaning:

    def __init__(self,data:pd.DataFrame):

        """initialize the class with the path to the data"""
        self.data=data
        self.columns=data.columns
        
        info("DataCleaning class initialized")
        
        # Use isinstance to check if data is a pandas DataFrame
        if not isinstance(data, pd.DataFrame):  
            error("Data is not a pandas dataframe")
            raise TypeError("Data is not a pandas dataframe")
        elif data.empty:
            error("Data is empty")
            raise ValueError("Data is empty")
        
    def preprocess(self)-> Tuple[np.ndarray, np.ndarray]:
        """clean the data by removing the null values and duplicates"""
        
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.preprocessing import LabelEncoder


        
        numeric_columns = self.data.select_dtypes(include=['int64', 'float64']).columns
        categorical_columns = self.data.select_dtypes(exclude=['int64', 'float64']).columns

        le = LabelEncoder()

        self.data[categorical_columns] = self.data[categorical_columns].apply(le.fit_transform)
        info("Label encoding applied to categorical columns")
        
        # splitting the data into train and test sets
        x = self.data.drop(columns=['review_score'])
        y = self.data['review_score'].values
       
        # scaling
        std_scaler = StandardScaler()
        x=std_scaler.fit_transform(x)
        info("Standard scaling applied to feature columns")
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
            error("preprocess() must return Tuple[np.ndarray, np.ndarray]")
            raise TypeError("❌ preprocess() must return Tuple[np.ndarray, np.ndarray]")

        return x,y
        


class ModelTrainig:
    def __init__(self,x_train, x_test, y_train, y_test):
        """initialize the class with the path to the data"""
        self.x_train=x_train
        self.y_train=y_train
        self.x_test=x_test
        self.y_test=y_test
        self.y_pred=None
        info("ModelTraining class initialized")
    
    def fit_model(self)->None:
        """train the model by only using pre-trained models in pkl files"""
        
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error,accuracy_score
        import joblib

        model = joblib.load('saved_model/model2.pkl')        
        model.fit(self.x_train, self.y_train)
        info("Model trained successfully")
        
        # predicting the test set
        self.y_pred = model.predict(self.x_test)

        return model
    
    
    def evaluate_model(self,y_test,y_pred)->None:
        # evaluating the model
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error,accuracy_score

        accuracy = accuracy_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        info(f"Model evaluation metrics:ACCURACY:{accuracy},MSE:{mse}, RMSE: {rmse}, R2: {r2}, MAE: {mae}")

if __name__ == "__main__":
       
       df = pd.read_csv('data/olist_customers_dataset.csv',nrows=100)
       x,y = DataCleaning(df).preprocess()

       print(type(x))
       print(type(y))
