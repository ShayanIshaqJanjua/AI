import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
#reads the csv file
df = pd.read_csv('First Week\FuelConsumptionCo2.csv')

#print(dataset.head())

#print(dataset.describe())
#creating constricted dataframe
cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
#print(cdf.head())

viz =- cdf[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]

#Vizualising constricted dataframe
viz.hist()
plt.show()
plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS,  color='teal')
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("CO2EMISSIONS")
plt.show()

plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='teal')
plt.xlabel("ENGINESIZE")
plt.ylabel("CO2EMISSIONS")
plt.show()

plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS,  color='teal')
plt.xlabel("CYLINDERS")
plt.ylabel("CO2EMISSIONS")
plt.show()

#Splitting the data into training and testing data
msk = np.random.rand(len(cdf)) < 0.8
train = cdf[msk]
test = cdf[~msk]

#training the model
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color = 'cyan')
plt.xlabel("Engine Size")
plt.ylabel("Emission")
plt.title("Training Data")
plt.show()
#Instantiate the model
regr = linear_model.LinearRegression()
#convert training data to array and rovide to model
train_x = np.asanyarray(train[["ENGINESIZE"]])
train_y = np.asanyarray(train[["CO2EMISSIONS"]])
regr.fit(train_x, train_y)
#returm coefficient and intercept of regression line
print("coefficient:" , regr.coef_)
print("Intercepts: ", regr.intercept_)
#Draw regression line on graph
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color = 'turquoise')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], "-r")
plt.xlabel("Engine Size")
plt.ylabel("Emmision")
plt.title("Regression Line")
plt.show()

#testing model
from sklearn.metrics import r2_score
test_x = np.asanyarray(test[["ENGINESIZE"]])
test_y = np.asanyarray(test[["CO2EMISSIONS"]])
test_y_ = regr.predict(test_x)

print("Mean Absolute Error:  %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Redisual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y)**2))
print("r2 score: %.2f" % r2_score(test_y, test_y_))


def predict_col(engine_size):
    engine_size_array = np.array([[engine_size]])
    predicted_co2 = regr.predict(engine_size_array)
    return predicted_co2

input_engine_size = float(input("Enter the engine size: "))
print(predict_col(input_engine_size))





