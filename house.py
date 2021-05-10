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
import warnings

warnings.filterwarnings('ignore')
class Model:
    def __init__(self):
        self.db = pd.read_csv('data.csv')
        self. labels = np.array(self.db['price'].values)
        self.degree = 2 
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
  
    def train_linear(self):
        best =0
        print("Linear Regression Scores")
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
            for i in range(0,2000):
                # print(df.shape)
                # print(self.labels.shape)
                features_train, features_test, lables_train, lables_test = sklearn.model_selection.train_test_split(df, self.labels, test_size=0.1)
                linear = linear_model.LinearRegression()
                linear.fit(features_train,lables_train)
                acc = linear.score(features_test,lables_test)
                if best < acc:
                    best = acc
                    with open("linear.pickle", "wb") as f:
                        pickle.dump(linear, f)
                    print(best)

    def train_poly(self):
        best =0
        print("Polynomial Model")
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
            for i in range(0,2000):
                features_train, features_test, lables_train, lables_test = sklearn.model_selection.train_test_split(df,
                                                                                                                    self.labels,
                                                                                                                    test_size=0.1)
                poly_features = PolynomialFeatures(self.degree)
                poly_features_train = poly_features.fit_transform(features_train)
                model = linear_model.LinearRegression()
                model.fit(poly_features_train,lables_train)
                poly_features_test = poly_features.fit_transform(features_test)
                acc = model.score(poly_features_test,lables_test)
                if best< acc:
                    with open("polyModel.pickle", "wb") as f:
                        pickle.dump(model, f)
                    best  = acc
                    print(best)    


    def predict(self,new_house):
        linear_in = open("linear.pickle","rb")
        linear_model = pickle.load(linear_in)
        
        poly_in =  open("polyModel.pickle","rb")
        poly_model = pickle.load(poly_in)
        
        xg_in = open("XGBoostModel.pickle","rb")
        xg_model = pickle.load(xg_in)

        poly_features = PolynomialFeatures(self.degree)
        
        poly_pred = poly_model.predict(poly_features.fit_transform(new_house))
        linear_pred = linear_model.predict(new_house)
        xg_pred = xg_model.predict(np.asarray(new_house))
        return max(poly_pred,linear_pred,xg_pred)



    def train_xg(self):
        best =0
        print("XGBoost Model")
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
            model = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.8, learning_rate=0.1,
                                max_depth=7, n_estimators=500,subsample = .7)
            features_train, features_test, lables_train, lables_test = sklearn.model_selection.train_test_split(df,
                                                                                                                self.labels,
                                                                                                                test_size=0.1)
            model.fit(features_test,lables_test)
            acc = model.score(features_test,lables_test)

            with open("XGBoostModel.pickle", "wb") as f:
                pickle.dump(model, f)
      