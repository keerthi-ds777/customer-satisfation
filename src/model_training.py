from logging import info
from typing import Tuple,Annotated
import numpy as np
from sklearn.base import ClassifierMixin
import mlflow
from zenml.client import Client
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error,accuracy_score
import joblib


class ModelTrainig:
    def __init__(self,x_train, x_test, y_train, y_test):
        """initialize the class with the path to the data"""
        self.x_train=x_train
        self.y_train=y_train
        self.x_test=x_test
        self.y_test=y_test
        self.y_pred=None
        info("ModelTraining class initialized")
    
    def fit_model(self,registerd_model=False,model_name:str="rf_model",
                 model_version:str="1",log_model:bool=False)->Tuple[
    Annotated[object, 'MlModel'], 
    Annotated[np.ndarray, 'predictions']
]:
        """train the model by only using pre-trained models in pkl files"""
        if registerd_model==False:
            model = joblib.load('models/model2.pkl')        
            model.fit(self.x_train, self.y_train)
            info("Model trained successfully")
            
            # predicting the test set
            self.y_pred = model.predict(self.x_test)
            mlflow.sklearn.log_model(model, name="model", registered_model_name=model_name)
            return model,self.y_pred
        
        else:
            experiment_tracker = Client().active_stack.experiment_tracker
            tracker=experiment_tracker.get_tracking_uri()
            mlflow.set_tracking_uri(f'{tracker}')
            model = mlflow.pyfunc.load_model(
                                             model_uri=f"models:/{model_name}/{model_version}"
                                            ).get_raw_model()
            model.fit(self.x_train, self.y_train)
            info("Model trained successfully")
            
            # predicting the test set
            self.y_pred = model.predict(self.x_test)
            if log_model==True:
                mlflow.sklearn.log_model(model, name="model", registered_model_name=model_name)
            return model,self.y_pred
    
    def evaluate_model(self,y_test,y_pred)->None:
        # evaluating the model
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
