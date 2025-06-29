from zenml import step
import pandas as pd
import numpy as np
from typing import Tuple,Annotated
import logging
from src.data_cleaning import DataCleaning
from typing import Tuple
from steps.ingest_data import ingesting
from steps.training_data import train_model
from sklearn.model_selection import train_test_split
import mlflow
from zenml.client import Client
experiment_tracker = Client().active_stack.experiment_tracker


#@step(experiment_tracker=experiment_tracker.name)
@step(enable_cache=False, experiment_tracker=experiment_tracker.name)
def cleaning(DataFrame: pd.DataFrame)-> Tuple[Annotated[np.ndarray,'x_train'],
                                              Annotated[np.ndarray,'x_test'],
                                              Annotated[np.ndarray,'y_train'], 
                                              Annotated[np.ndarray,'y_test']]: 
            """Clean the data and return train-test splits."""
            data = DataCleaning(DataFrame)
            x,y = data.preprocess()
            
            """Split the data into training and testing sets."""
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
            logging.info("Data split into train and test sets")
            return x_train, x_test, y_train, y_test


#if __name__ == "__main__":
 #   # Example usage
  #  data=ingesting(r"C:\Users\loges\Desktop\python\sample projects\MLOps\customer-segmentaion\data\olist_customers_dataset.csv")
   # k,g,f,i = cleaning(data)
   