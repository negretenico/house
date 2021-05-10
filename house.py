import pandas as pd
import numpy as np
import sklearn
import pickle
from sklearn import linear_model
import math
import xgboost as xgb
from sklearn.preprocessing  import StandardScaler, PolynomialFeatures
from sklearn.utils import shuffle
import itertools as tools
class Model:
    def __init__(self):
        self.db = pd.read_csv('data.csv')
        self. labels = np.array(self.db['price'].values) 
    def data(self):
        print(self.db.isnull().sum())
    def extracts_features(self):
        std = StandardScaler()
        features = np.array(self.db.drop(columns = [c for c in self.db.columns if c not in listOfNames]))
        std.fit(features)
        labels = np.array(self.db['price'].values)
        return features,labels


    def get_best_features(self):
        feat_com = []
        scores  = {}
        listOfNames = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition','sqft_above','sqft_basement']
        df = self.db.drop(columns = [c for c in self.db.columns if c not in listOfNames])
        std = StandardScaler()
        for i in range(1,len(df.columns)):
            feat_com.append(list(tools.combinations(df.columns,i)))
        for i,combination in enumerate(feat_com):
            print(i)
            for entry in combination:
                feature= np.array(df[list(entry)])
                std.fit(feature)
                best,test_feature,test_label = self.train_linear(feature)
                scores[entry] = best
        max_pair = max(scores,key = scores.get)
        print(max_pair)
        with open('bestcombination.txt', 'w') as f:
            f.write(str(max_pair))
        return max_pair
    def do_linear(self):
        best =0
        with open('bestcombination.txt', 'r',encoding='utf8') as f:
            columns = f.read()
            columns = columns.replace("(","")
            columns = columns.replace(")","")
            columns = columns.replace("'","")
            columns = columns.replace(" ","")
            columns = columns.replace(")","")
            columns = columns.replace("'","")
            columns = columns.split(",")


            df = self.db.drop(columns = [c for c in self.db.columns if c not in columns])
            for i in range(0,1000):
                # print(df.shape)
                # print(self.labels.shape)
                features_train, features_test, lables_train, lables_test = sklearn.model_selection.train_test_split(df, self.labels, test_size=0.1)
                linear = linear_model.LinearRegression()
                linear.fit(features_train,lables_train)
                acc = linear.score(features_test,lables_test)
                if best < acc:
                    best = acc
                    with open("skPickle.pickle", "wb") as f:
                        pickle.dump(linear, f)
                    print(best)
    def train_poly(self):
        features, labels = self.extracts_features()
        degree = 2
        best =0
        for i in range(0,100):
            features_train, features_test, lables_train, lables_test = sklearn.model_selection.train_test_split(features,
                                                                                                                labels,
                                                                                                                test_size=0.1)
            poly_features = PolynomialFeatures(degree)
            poly_features_train = poly_features.fit_transform(features_train)
            model = linear_model.LinearRegression()
            model.fit(poly_features_train,lables_train)
            poly_features_test = poly_features.fit_transform(features_test)
            prediction = model.predict(poly_features_test)
            acc = model.score(poly_features_test,lables_test)
            if best< acc:
                best  = acc
                
        return prediction, best, lables_test

    def do_poly(self,showPred):
        pred, acc, lables_test = self.train_poly()
        print("Polynomial Regression")
        print("Accuracy ", acc)
        if(showPred == "Y"):
            for x in range(len(pred)):
                print("We predicted ", pred[x], " Actually was ", lables_test[x])

    def train_linear(self,feature):
        best = 0
        for i in range(0,100):
            features_train, features_test, lables_train, lables_test = sklearn.model_selection.train_test_split(feature, self.labels, test_size=0.1)
            linear = linear_model.LinearRegression()
            linear.fit(features_train,lables_train)
            acc = linear.score(features_test, lables_test)
            if best < acc:
                best = acc
                with open("housePrices.pickle", "wb") as f:
                    pickle.dump(linear, f)
        return best, features_test, lables_test
    def train_xg(self):
        features, labels = self.extracts_features()
        features_train, features_test, lables_train, lables_test = sklearn.model_selection.train_test_split(features, labels,
                                                                                                    test_size=0.1)
        xg_reg = xgb.XGBRegressor(objective='reg:linear', colsample_bytree=0.3, learning_rate=0.1,
                                  max_depth=5, alpha=10, n_estimators=10)
        xg_reg.fit(features_test,lables_test)
        acc = xg_reg.score(features_test,lables_test)
        with open("xgHousePrices.pickle", "wb") as f:
            pickle.dump(xg_reg, f)
        return acc, features_test, lables_test
    def do_xg(self,showPred):
        acc, features_test, lables_test = self.train_xg()
        pickle_in = open("xgHousePrices.pickle", "rb")
        xg_reg = pickle.load(pickle_in)
        predictions = xg_reg.predict(features_test)
        print("XGRegressor")
        print("Accuracy ", acc)
        if (showPred == "Y"):
            for x in range(len(predictions)):
                print("We predicted ",predictions[x]," Actually was " ,lables_test[x])