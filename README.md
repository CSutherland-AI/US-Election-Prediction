# US-Election-Prediction
Using Machine Learning to predict which party wins each district in the US

Features

FIPS - Federal Information Processing Series, are numeric codes assigned by the National Institute of Standards and Technology (NIST). US counties are identified by a 3-digit number.
County - the name of the county, given by county_name, state_initials.

DEM - number of democratic votes recieved.

GOP - number of republican votes recieved.

MedianIncome - median household income.

MigraRate - the migration rate as the difference between the number of persons entering and leaving a country during the year per 1,000 persons (based on midyear population).

BirthRate - frequency of live births in a given population as live births per 1,000 inhabitants.

DeathRate - the ratio of deaths to the population of a particular area or during a particular period of time, calculated as the number of deaths per one thousand people per year.

BachelorRate - percent of inhabitants with a bachelors degree (who are above some age threshold).

UnemploymentRate - percent of the labor force that is unemployed (who are above some age threshold).


Source: 
The node features are constructed from USDA demographic data:

https://www.ers.usda.gov/data-products/county-level-data-sets/

as well as county-level U.S. presidential election statistics formatted by Tony McGovern:

https://github.com/tonmcg/US_County_Level_Election_Results_08-16






Rationale For Algorithms and Design Decisions 

Decision trees carry the advantages of being able to represent non-linear relationships and use categorical data. Because the relationship between the features given and the target variable is likely non-linear and some of the features contained categorical data, I deduced that a decision tree would be useful.

The neural network is a multilayer perceptron. This algorithm is capable of approxiamating any continuous function and so it is ideal for situations in which the ideal function mapping the features to the labels is a complex one. There are many features that have complex relationships when it comes to which party wins an election and so an algorithm that is capable of representing any relationship works well for this problem.




Decision Tree

Tested the accuracy of decision trees with different maximum leaf node limits and maximum depth limits. Each model was trained with the training set and its accuracy tested on the validation set.

Initially, I used a for loop to test the accuracy of decision trees with different depths, during this process, it was found that a different value was given for the best depth each time, and so I opted to use nested for loops to test each depth several times and take the average of all of the best depths.

Neural Network

For the neural network I used for loops to test a wide range of values for learning rate, alpha, maximum number of iterations, activation function and hidden layer size. Interestingly changes to these hyperparameters had very little effect on the prediction accuracy.Changing the number of hidden layers and size of each layer also had very little effect on predictive accuracy.


Random Forest

Random forests are more robust that single decision trees. Since a random forest uses many decision trees it limits overfitting, thus reducing generalization error.

This time around, for model selection I used a random search grid with kfold cross-validation. Instead of testing one parameter at a time using for loops as was done in the basic solution, different parameter combintions were simultaneously tested. This allows one to test more possible models, thus increasing the chance of finding a model that generalizes well.

In deciding what ranges of parameter values to test, special attention was paid to ensuring that n_estimaors (number of trees in the forest) was sizable and that max_depth was not too large in order to increase generalizability.

A random search grid was chosen over a grid search (which tests all possible parameter combinations) in order to give the algorithm a reasonable runtime. Gridsearch may have yielded more accurate parameters but the runtime would have been too long.



