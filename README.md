# FootSeer: Real-Time Soccer Match Prediction Using Machine Learning Algorithms

<h2> Goal </h2>
The goal of our project was to create models that can predict the outcome of a soccer match. The three possible outcomes are home team wins, away team wins, or draw. Using our models, we created a web application that would allow a user to input real time parameters and the relevant time, and receive a predicted outcome.

<h2> Relevance </h2>
It is estimated that there are about 3.5 billion soccer fans worldwide. A good majority of soccer fans are fiercely loyal, whether it is to a club team, or their national team. In addition, betting on soccer matches is a common form of leisure. Thus, the ability to predict match outcomes is an extremely useful tool for those engaged in betting. For fans that do not indulge in such pleasures, they will likely still find great joy in being able to predict the outcome of a soccer match. Either way, our tool is a fun and useful interactive tool for soccer fans.

<h2> Development </h2>

1. Scrapped data from 5 countries (England, France, Germany, Italy, Spain), across 8 years (2008-2015), resulting in 9,123 individual matches. See data_scrape.py

2. Data put in MySQL, and match parameters were aggregated for every 5 minute interval up till 90 minutes.

3. 3 Performance metrics were created and assigned to each team based on their performance in home and away games in the recent past

4. 3 classification models were built and tested in R, and were interfaced with python using rpy2 library

5. Web application was developed using Python and Flask

<h2> Model Building </h2>
<h4> Multinomial Logistic Regression </h4>
The following steps were done at each 5 minute interval up till 85 minutes. (i.e. there is a separate model at every 5 minutes)

1. Data was split into training data and testing data.

2. 10-folds LASSO Cross Validation was done to determine the best lambda that reduced residual error

3. Using the best lambda, the model was re-trained on training data

4. The model was used to predict outcomes on the test data, and the accuracy was obtained by determining the proportion of correct predictions
<h4> K-Nearest Neighbors </h4>
The data was split into a training and test dataset (70:30 ratio). The two major parameters to be decided for this method were:

1. The parameters which would help in predicting the final outcome – We noticed that the Goal difference, number of red and yellow cards, the number of substitutions and the difference in the performance score of the two teams were the most accurate in predicting the outcome of the game

2. The number of nearest neighbors – This was taken to 75 as this was the square root of the number of records in the training data set
<h4> Decision Trees </h4>
A decision tree was built for every 5-minute interval until the 85th minute. The data was split in training (70%) and test (30%). The depth for every tree was chosen to get the best accuracy on the training data that wouldn’t over-fit it.

<h2> Model Results </h2>
<h4> Multinomial Logistic Regression </h4>
The accuracy of each model ranged from about 50% in the 5 minute model, to about 88% in the 85 minute model. We note that this is expected, since the closer we get to the end of the match, the more closely the variables are related to the final outcome. The value in a prediction increases the further away from the end of the game it is made, thus, we find that the Multinomial Logistic Regression model does not perform so well. At halftime (45 minutes), the accuracy on the test data is only about 62%.
<h4> K-Nearest Neighbors </h4>
The K-Nearest Neighbors mainly gives the outcome and not a p-value by classifying the incoming data point to the majority of the nearest neighbors. The accuracy of the model was around 50% at the start of them game and increase to about 89% by the 85th minute. The probability that is shown in the application is the percentage of the nearest neighbors falling into the predicted outcome.
<h4> Decision Trees </h4>
The decision trees obtained ranged from 1 to 3 levels of depth, where the Goal Difference (home goals minus away goals) was the only variable selected almost all the time. The accuracy of the decision trees started at 49% at the 5th minute and ended with 89% at the 85th minute. The results show that the accuracy improves faster as we approach to the end of the games, that is the improvement of accuracy is bigger going from the 75th minute to the 80th, than from the 15th to the 20th minute.

<h2> Improving Model Performance </h2>
We note that the accuracies of our models are not ideal, which might suggest that more parameters should be added to the models. Such parameters could be metrics for team performance, metrics for player performance at a given position (striker, midfielder, defender), weather conditions etc. Other classification techniques such as Naïve Bayes or Random Forest can also be explored.


<h2> Project Team </h2>
Keshava Dilwali, Yingqiu Lee, Srijan Kumar, Claudio Flores (Columbia University)
