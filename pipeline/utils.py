

import logging
from src.data_cleaning import DataCleaning
import pandas as pd
from steps.preprocessing import cleaning
from steps.evaluate import evaluate_model
from steps.ingest_data import ingesting
from steps.training_data import train_model


def get_data_for_test():
    try:
        df = pd.read_csv("./data/olist_customers_dataset.csv")
        df = df[[
        "payment_sequential",
        "payment_installments",
        "payment_value",
        "price",
        "freight_value",
        "product_name_lenght",
        "product_description_lenght",
        "product_photos_qty",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
    ]]
        df = df.sample(n=100)
        x,y = DataCleaning(df).preprocess()
        result = df.to_json(orient="split")
        
        return result
    except Exception as e:
        logging.error(e)