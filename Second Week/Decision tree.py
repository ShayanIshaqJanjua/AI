import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# Loaded the data from the url
url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/drug200.csv'
myData = pd.read_csv(url)

# Get the values from the csv and encode them to the values we need
x = myData[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
le_s = preprocessing.LabelEncoder()
le_s.fit(['F', 'M'])
x[:, 1] = le_s.transform(x[:, 1])

le_b = preprocessing.LabelEncoder()
le_b.fit(['LOW', 'NORMAL', 'HIGH'])
x[:, 2] = le_b.transform(x[:, 2])

le_c = preprocessing.LabelEncoder()
le_c.fit(['NORMAL', 'HIGH'])
x[:, 3] = le_c.transform(x[:, 3])

# Convert values to Drug values
y = myData['Drug'].values

# Split the datasat into training data and testing data for the model
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=3)

# Training the model using the training data
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth=4)
drugTree.fit(x_train, y_train)

# Test a Prediction and compare it to the actual value
y_pred = drugTree.predict(x_test)
print("Predictions:", y_pred[:5])
print("Actual:", y_test[:5])

# Plot the Decision Tree using MatPlotLib to visualise the resulting data tree
plt.figure(figsize=[12, 8])
tree.plot_tree(
    drugTree,
    feature_names=['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K'],  # Include all features
    class_names=drugTree.classes_,  # Dynamically use unique class labels
    filled=True,
    rounded=True
)
plt.show()
