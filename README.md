# US-Election-Prediction
Using Machine Learning to predict which party wins each district in the US

Explanation

Decision trees carry the advantages of being able to represent non-linear relationships and use categorical data. Because the relationship between the features given and the target variable is likely non-linear and some of the features contained categorical data, I deduced that a decision tree would be useful.

The neural network is a multilayer perceptron. This algorithm is capable of approxiamating any continuous function and so it is ideal for situations in which the ideal function mapping the features to the labels is a complex one. There are many features that have complex relationships when it comes to which party wins an election and so an algorithm that is capable of representing any relationship works well for this problem.





Decision Tree

I tested the accuracy of decision trees with different maximum leaf node limits and maximum depth limits. Each model was trained with the training set and its accuracy tested on the validation set.

Initially, I used a for loop to test the accuracy of decision trees with different depths, during this process, I found that I was given a different value for the best depth each time, and so I opted to use nested for loops to test each depth several times and take the average of all of the best depths.

Neural Network

For the neural network I used for loops to test a wide range of values for learning rate, alpha, maximum number of iterations, activation function and hidden layer size. Interestingly changes to these hyperparameters had very little effect on the prediction accuracy.

Changing the number of hidden layers and size of each layer also had very little effect on predictive accuracy.

