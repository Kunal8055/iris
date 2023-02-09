import pandas as pd
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle



class iris():
    def __init__(self,array):
        # self.sl=sl
        # self.sw=sw
        # self.pl=pl
        # self.pw=pw
        self.array=array
        
       
    

    def prediction_Classification(self):
        with open ("Model_data_classification.pkl","rb") as f:
            Model=pickle.load(f)
        result=Model.predict(self.array)[0]
        return result
    
    def prediction_Petallength(self):
        with open ("Model_data_PetalLengthCm.pkl","rb") as f:
            Model=pickle.load(f)
        result=Model.predict(self.array)[0]
        return result

    def prediction_Petalwidth(self):
        with open ("Model_data_PetalWidthCm.pkl","rb") as f:
            Model=pickle.load(f)
        result=Model.predict(self.array)[0]
        return result

    def prediction_SepalLength(self):
        with open ("Model_data_SepalLengthCm.pkl","rb") as f:
            Model=pickle.load(f)
        result=Model.predict(self.array)[0]
        return result

    def prediction_SepalWidth(self):
        with open ("Model_data_SepalWidthCm.pkl","rb") as f:
            Model=pickle.load(f)
        result=Model.predict(self.array)[0]
        return result
