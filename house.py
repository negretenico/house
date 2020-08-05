import pandas as pd
import numpy as np
import sklearn
import pickle
from sklearn import linear_model
import math
from sklearn.preprocessing  import StandardScaler
from sklearn.utils import shuffle
class Model:
    def __init__(self):
        self.db = pd.read_csv('houses.csv')
    def extracts_features(self):
        listOfNames = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition','sqft_above','sqft_basement']
        std = StandardScaler()
        features = np.array(self.db.drop(columns = [c for c in self.db.columns if c not in listOfNames]))
        std.fit(features)
        labels = np.array(self.db['price'].values)
        return features,labels

    def do_ml(self):
        best, features_test, lables_test = self.find_best()
        pickle_in = open("housePrices.pickle", "rb")
        linear = pickle.load(pickle_in)
        print(best)
        predictions = linear.predict(features_test)
        for x in range(len(predictions)):
            print("We predicted ",predictions[x]," Actually was " ,lables_test[x])

    def find_best(self):
        best = 0
        features, lables = self.extracts_features()
        for i in range(0,100):
            features_train, features_test, lables_train, lables_test = sklearn.model_selection.train_test_split(features, lables, test_size=0.1)
            linear = linear_model.LinearRegression()
            linear.fit(features_train,lables_train)
            acc = linear.score(features_test, lables_test)
            print("Current Accuracy ",acc)
            if best < acc:
                best = acc
                with open("housePrices.pickle", "wb") as f:
                    pickle.dump(linear, f)
        return best, features_test, lables_test
