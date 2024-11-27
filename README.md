# AI
## What is this repository about?
This repository was created for the purpose of storing prjects created during the 4 week optional Ai module at the City Computer Science Society. This module goes specifically into Machine Learning and Deep Learning, creating projects that will be based on topics such as K random means and SVMs.
## How this repository is organised.
This repository is organised into weeks, where the projects I have completed for each week are kept in folders aptly named "Week 1", "Week 2", "Week 3" and "Week 4". Additionally, there is a folder called "Extra" which will contain projects I have decided to do on my own outside of the specific module projects which expand on the topics being covered in the lectures.
## Languages and Technologies used
Every project is coded solely in Python with these technologies: Numpy, Matplotlib, SciKit-learn, TensorFlow, Pandas
## Week 1
In the first week, we were isntroduced into AI and Machine Learning. We were given an initial rundown on what everything is before beginning programming. We used numpy, pandas, scikit-learn and matplotlib, to train and test a regression line model. This began with importing our data form an already existing csv file on the web containing the data we trained the model on. We then loaded that data and begam constricting our dataframe so that we had only the data we needed to train the model with. We then plotted this data using matplotlib to see what our data looks like visually. 

Using a random variable mask, we took a random sample of 80% of our data which we will use as training data. We then saved this data along with the other 20%, which were used as text data.

Using numpy, the data was transformed into an array that we then fed into out linear regression model we imporeted using Scikit-learn. After training, we then plot this new regression line visually to see the accuracy of it.

We then tested the model using our test data to check the accuracy of our model, returning 3 metrics, Mean Absolute Error, Residual sun of squares (MSE) and our r^2 score.

We then coded a function that would allow the user to input an x value and return the predicted y value based on our calculated regression line.
