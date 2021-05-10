from house import Model

model = Model()

#model.train_xg()
# model.train_linear()
# model.train_poly()
#('bedrooms', 'bathrooms', 'sqft_lot', 'floors', 'waterfront', 'sqft_above', 'sqft_basement')
new_house = [3,2.5,1500,2,0,0,100]
best_estimate = model.predict([new_house])