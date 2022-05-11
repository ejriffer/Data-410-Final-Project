# Modeling the NHL: Using KPIs to Predict the Outcome of Games

With the recent legalization of sports betting across many states in the US many are looking for statistical ways to beat the odds makers. While more popular betting sports (like football, boxing, and racing) have extremely sophisticated models to try and predict outcomes some smaller sports, such as hockey, do not. So, for my final project I wanted to see if I could create a model to predict the outcome of NHL games based on readily avaliable statistical information. 

## Introduction 

For this project I would like to train different classification models on statistical output from specific NHL games to see if a model can be created to predict the outcome of that specific game. Then, that model can be used predict the outcome of future games based on the recent data for the teams involved in that game.  

## Description of Data

The data that I used for this project comes from the Python libraries of sportsipy and sportsreference. Below are the specific data sets from each library that I imported. 

```
from sportsreference.nhl.teams import Teams
from sportsipy.nhl.roster import Roster
from sportsipy.nhl.roster import Player
from sportsipy.nhl.schedule import Schedule
from sportsipy.nhl.boxscore import Boxscore
```

These libraries contain information about all the the teams, players, and games from the NHL over the last few decades, as they started keeping track of these statistics. I chose to use the seasons 2014-15, 2015-16, and 2016-17 to train my model and test it on the 2017-18 and 2018-19 seasons. I chose these seasons because they are the last five full seasons the NHL played before the COVID-19 Pandemic. 

The first step in the project was subsetting the data based on year and team. I decided to create one dataframe for each NHL season with the rows grouped by team. I started by creating a list of each team's abbreviation to be able to iterate through for the rest of the project. I chose to exclude the Vegas Golden Knights  because their first season of existence was the 2017-18 season, and thus would have no games in the seasons that trained the model. I also had to exclude the Seattle Kraken because their first season was this past year and would not have any data across any of the chosen seasons but would still show up in the data if I did not exclude them .

Below is the code to create the team abbreviation list. 

```
teams = Teams()
team_abb = []
for team in teams:
    team_abb.append(team.abbreviation)
team_abb.remove('VEG')
team_abb.remove('SEA')
```

Below is the code that I used to create each data set. For each season I changed the input year from 2015 through 2019.

```
def create_boxscores_table():
    boxscore_index_list = []
    boxscores_table = pd.DataFrame() 
    for year in range(2016,2017,1):
    
        for team in team_abb:

            for boxscore_index in Schedule(str(team), year = year).dataframe['boxscore_index']:
                if team != 'VEG' or 'SEA':
                    boxscore = Boxscore(str(boxscore_index)).dataframe # get boxscore info for specific game in specific season
                    boxscore_index_list.append(boxscore_index) # append boxscore_id to list to add to df

                    boxscores_table = boxscores_table.append(boxscore, ignore_index = True)

    return boxscores_table
```

I then saved each output table (which I called 'boxscores' + the specific year) to be able to call them in the future. Each boxscore table had 2,460 columns, or 82 regular season games for the 30 teams. The rows were grouped by team and within each team it was grouped chronologically through the season.

Each boxscore table had 42 columns for each game of which I used 16. One of the columns was *winning_abbr* which showed which team won that particular game, which will help me create my output or classification variable. Another column was *boxscore_index* which helped me determine if the game was a home or away game for a specific team. The other 14 columns of importance held statistical information from the game, which will be my input variables. They were:

**assists**: the number of assists the team registered

**even_strength_assists**: the number of assists the team registered while at even strength

**even_strength_goals**: the number of goals the team scored at even strength

**goals**: the number of goals the team scored

**penalties_in_minutes**: the length of time the team spent in the penalty box

**points**:the number of points the team registerd

**power_play_assists**: the number of assists the team registered while on the power play

**power_play_goals**: the number of goals the team scored while on the power play

**save_percentage**: the percentage of shots the team saved (ranges from 0-1)

**saves**: the number of saves the team made

**shooting_percentage**: the percentage of the teams shots that scored (ranges from 0-100)

**short_handed_assists**: the number of assists the team registered while short handed

**short_handed_goals**: the number of goals the teams scored while short handed

**shots_on_goal**: the number of shots on goal the team registered


## Methods

### Pre-Processing Methods

Now that I had my five data frames I created my *X* data frame and corresponding *y* variable to use in my classification models. As stated before I wanted to use the 2014-15, 2015-16, and 2016-17 seasons to train my model and  the 2017-18 and 2018-19 seasons to test it. 

For the *Xtrain* data I used took the dataframes in units of 82 rows each (a full season for one team) and used the *boxscore_index* to determine whether or not the game was a home or away game for a team. Then I either took the *home* or *away* statistics from the game that corresponded to the team I was looking at at that specific moment. The below code was used to create the *Xtrain* data.

```
Xtrain = pd.DataFrame()
for boxscore in [boxscores15, boxscores16, boxscores17]:
    for i, team in zip(range(0,2461,82),team_abb):
        temp = boxscore[i:i+82].reset_index()
        for index, row in temp.iterrows():
            if temp['boxscore_index'].iloc[index][-3:] == team:
                tempX = temp[['home_assists','home_even_strength_assists','home_even_strength_goals','home_goals',
                          'home_penalties_in_minutes','home_points','home_power_play_assists','home_power_play_goals',
                          'home_save_percentage', 'home_saves', 'home_shooting_percentage','home_short_handed_assists',
                          'home_short_handed_goals','home_shots_on_goal']]
            else:
                tempX = temp[['away_assists','away_even_strength_assists','away_even_strength_goals','away_goals',
                          'away_penalties_in_minutes','away_points','away_power_play_assists','away_power_play_goals',
                          'away_save_percentage', 'away_saves', 'away_shooting_percentage','away_short_handed_assists',
                          'away_short_handed_goals','away_shots_on_goal']]
        tempX.columns = ['assists','even_strength_assists','even_strength_goals','goals',
                          'penalties_in_minutes','points','power_play_assists','power_play_goals',
                          'save_percentage', 'saves', 'shooting_percentage','short_handed_assists',
                          'short_handed_goals','shots_on_goal']
        Xtrain = Xtrain.append(tempX)
Xtrain = Xtrain.reset_index(drop = True)
```

For the *ytrain* data I used the *winning_abbr* column from the data to create an array where 0 = loss and 1 = win. The below code was used to create the *ytain* data.

```
ytrain = pd.DataFrame()
for boxscore in [boxscores15, boxscores16, boxscores17]:
    for i, team in zip(range(0,2461,82),team_abb):
        temp = boxscore[i:i+82].reset_index()
        temp['win'] = 'tbd'
        for index, row in temp.iterrows():
           if temp['winning_abbr'].iloc[index] == team:
               temp['win'].iloc[index] = 1
           else:
              temp['win'].iloc[index] = 0
        ytrain = ytrain.append(temp)
ytrain = ytrain.reset_index(drop = True)
ytrain = ytrain['win']
```

While the train data was straightforward (simply the statistical output from each team in each game) the test data was a bit more complicated to create. The train data is statistical information only avaliable after a game is complete, but in order to predict future games we need to use informaiton that is avaliable before the game starts. So, I chose to average the statistical output from a team over their last five games as the input varaibles. 

The first step in creating the *Xtest* data was to create a dataframe for each team with their last 5 games from the 2016-17 season and all of their games from the 2017-18 and 2018-19 seasons. The last five games from the 2016-17 season are needed for the first few games of the 2017-18 season. The below code shows the creation of these initial dataframes. 

```
d = {}
for i, team in zip(range(0,2461,82),team_abb):
    d[team] = pd.DataFrame()
    temp = boxscores17[i:i+82].reset_index()
    team_last_5 = temp[-5:]
    d[team] = team_last_5
for i, team in zip(range(0,2461,82),team_abb):
  temp = boxscores18[i:i+82].reset_index()
  d[team] = d[team].append(temp)
for i, team in zip(range(0,2461,82),team_abb):
  temp = boxscores19[i:i+82].reset_index()
  d[team] = d[team].append(temp)
  d[team] = d[team].reset_index(drop = True)
```

The next step is to use these dataframes to create the *Xtest* dataframe. I used a shifting subset of the data to grab five rows shifting down one row in each iteration. The rest of the code is the same as for *Xtrain* above. The below code was used to create *Xtrain*.

```
Xlist = []
for team in team_abb:
    for i,j in zip(range(0,169),range(5,169)):
        temp = d[team].iloc[i:j].reset_index(drop = True)
        X = pd.DataFrame()
        for index, row in temp.iterrows():
            if temp['boxscore_index'].iloc[index][-3:] == team:
                tempX = temp[['home_assists','home_even_strength_assists','home_even_strength_goals','home_goals',
                          'home_penalties_in_minutes','home_points','home_power_play_assists','home_power_play_goals',
                          'home_save_percentage', 'home_saves', 'home_shooting_percentage','home_short_handed_assists',
                          'home_short_handed_goals','home_shots_on_goal']]
            else:
                tempX = temp[['away_assists','away_even_strength_assists','away_even_strength_goals','away_goals',
                          'away_penalties_in_minutes','away_points','away_power_play_assists','away_power_play_goals',
                          'away_save_percentage', 'away_saves', 'away_shooting_percentage','away_short_handed_assists',
                          'away_short_handed_goals','away_shots_on_goal']]
            tempX.columns = ['assists','even_strength_assists','even_strength_goals','goals',
                          'penalties_in_minutes','points','power_play_assists','power_play_goals',
                          'save_percentage', 'saves', 'shooting_percentage','short_handed_assists',
                          'short_handed_goals','shots_on_goal']
        Xlisttemp = []
        for i in tempX.columns:
            Xlisttemp.append(tempX[i].mean())
        Xlist.append(Xlisttemp)

Xtest = pd.DataFrame(Xlist, columns = ['assists','even_strength_assists','even_strength_goals','goals',
                          'penalties_in_minutes','points','power_play_assists','power_play_goals',
                          'save_percentage', 'saves', 'shooting_percentage','short_handed_assists',
                          'short_handed_goals','shots_on_goal'])
```

The *ytest* data was created in the same way as the *ytrain*. The below code was used to creat *ytest*.

```
ytest = pd.DataFrame()
for boxscore in [boxscores18, boxscores19]:
    for i, team in zip(range(0,2461,82),team_abb):
        temp = boxscore[i:i+82].reset_index()
        temp['win'] = 'tbd'
        for index, row in temp.iterrows():
           if temp['winning_abbr'].iloc[index] == team:
               temp['win'].iloc[index] = 1
           else:
              temp['win'].iloc[index] = 0
        ytest = ytest.append(temp)
ytest = ytest.reset_index(drop = True)
ytest = ytest['win']
```

### Classification Methods 

For this project I chose to use four classification models to see if one performed better than the others. The models I chose were Logistic Regression, Decision Tree, Random Forest, and K-Nearest Neighbors. 

**Logistic Regression**

Logistic regression is a classification algorithm that determines the probability of one event out of two options. It is a linear regression model but the final output of the model is transformed by a logistic/sigmoid function which maps the output variable between 0 and 1. 

These values are the probability of each input variable belonging to one or the other group. For this project I used hard classifications, or assigning the variable to the group (either 0/loss or 1/win) that it sayd it is more likely to be in. 

**Decision Tree Classifier**

Decision tree classifiers use decision trees to split the data into different nodes. The goal of this split is to group all of the data with others of it's same group. This splitting criteria is based on gini impurity which is defined as:

![CodeCogsEqn-14](https://user-images.githubusercontent.com/74326062/167920906-48dbf239-61fc-4016-98d0-a1c3d95f6451.svg) 

The goal of the decision tree is to minimize the gini impurity. 

**Random Forest Classifier**

Random forest classifier is an ensemble model, made up of many decision trees. The prediction from a random forest is found by taking the average from every tree in the forest. To make sure each tree is not exactly identical new data will be boostraped, or randomly generated in a way that is still very similar to the original set. 

**K-Nearest Neighbors Classifier**

K-nearest neighbors classifiers assigns the input to whatever group the majority of it's *k* nearest neighbors are in. The variable *k* defaults at 5 but can any number the user chooses. 

### Application and Validation

Before running my data through the classifiers I standardized the *X* data so all of the values were on the same scale and no one variable dominated the model. I also turned my *y* data into a numpy array of integers so the models could classify the games as wins and losses properly. The below code shows this for the train and test data.

```
scaler = StandardScaler()
Xtrainscaled = scaler.fit_transform(Xtrain)
ytrain = np.array(ytrain)
ytrain = ytrain.astype('int')

scaler = StandardScaler()
Xtestscaled = scaler.fit_transform(Xtest)
ytest = np.array(ytest)
ytest = ytest.astype('int')
```

The validation techniques that I chose to use were accuracy scores and a confusion matrix. The accuracy score shows the what percentage of the data was accurately classified as a win or a loss. The confusion matrix shows the number of true positives or correctly identified 'win' data, true negatives or correctly identified 'loss' data, false positives or incorrectly identified 'win' data, and false negatives or incorrectly identified 'win' data. 

The first step in creating my model was running the train data through all four classification models I had chosen.  

**Train Data**

The first classifier model I ran was a logistic regression. The below code shows this implementation. 

```
log_reg = LR()
log_reg.fit(Xtrainscaled, ytrain)
predicted_log = log_reg.predict(Xtrainscaled)
cm_df_log_reg = pd.DataFrame(data = cm(ytrain, predicted_log), columns = ['lose', 'win'], index = ['lose','win'])
print(accuracy_score(ytrain, predicted_log))
cm_df_log_reg
```

The accuracy for the above model was 0.5897, and the confusion matrix is shown below.

<img width="138" alt="Screen Shot 2022-05-11 at 2 43 15 PM" src="https://user-images.githubusercontent.com/74326062/167922942-36d08ffe-e99e-4820-a62a-18e0df245cc7.png">

The next classifier model I ran was a decision tree. The below code shows this implementation. 

```
dtc = DTC(random_state = 1693)
dtc.fit(Xtrainscaled, ytrain)
predicted_dtr = dtc.predict(Xtrainscaled)
cm_df_dtr = pd.DataFrame(data = cm(ytrain, predicted_dtr), columns = ['lose', 'win'], index = ['lose','win'])
print(accuracy_score(ytrain, predicted_dtr))
cm_df_dtr
```

The accuracy for the above model was 0.8027, and the confusion matrix is shown below.

<img width="138" alt="Screen Shot 2022-05-11 at 2 44 05 PM" src="https://user-images.githubusercontent.com/74326062/167923255-fd902770-1967-46ad-854d-f79005fbb003.png">

The next classifier model I ran was a random forest. The below code shows this implementation. 

```
rfc = RFC(random_state = 1693)
rfc.fit(Xtrainscaled, ytrain)
predicted_rfc = rfc.predict(Xtrainscaled)
print(sum(ytrain == predicted_rfc)/ytrain.shape[0])
cm_df_rfc = pd.DataFrame(data = cm(ytrain, predicted_rfc), columns = ['lose', 'win'], index = ['lose','win'])
cm_df_rfc
```

The accuracy for the above model was 0.8027, and the confusion matrix is shown below.

<img width="138" alt="Screen Shot 2022-05-11 at 2 45 26 PM" src="https://user-images.githubusercontent.com/74326062/167923338-c9c0d5f5-09f4-4e85-88d2-2c69a8d5edd9.png">


The next classifier model I ran was a k-nearest neighbors. The below code shows this implementation. 

```
knn = KNeighborsClassifier()
knn.fit(Xtrainscaled, ytrain)
predicted_knn = knn.predict(Xtrainscaled)
cm_df_knn = pd.DataFrame(data = cm(ytrain, predicted_knn), columns = ['lose', 'win'], index = ['lose','win'])
print(accuracy_score(ytrain, predicted_knn))
cm_df_knn
```

The accuracy for the above model was 0.6658, and the confusion matrix is shown below.

<img width="138" alt="Screen Shot 2022-05-11 at 2 45 52 PM" src="https://user-images.githubusercontent.com/74326062/167923420-05767820-7e70-4a5c-ae0f-80211065e020.png">

**Test Data**

The next step was to use the models from above and run the test data through them. 

The first classifier model I ran was a logistic regression. The below code shows this implementation. 

```
predicted_log = log_reg.predict(Xtestscaled)
cm_df_log_reg = pd.DataFrame(data = cm(ytest, predicted_log), columns = ['lose', 'win'], index = ['lose','win'])
print(accuracy_score(ytest, predicted_log))
cm_df_log_reg
```

The accuracy for the above model was 0.5900, and the confusion matrix is shown below.

<img width="138" alt="Screen Shot 2022-05-11 at 2 48 33 PM" src="https://user-images.githubusercontent.com/74326062/167923933-f33d465f-68c7-433d-b23f-affb91d93a78.png">

The next classifier model I ran was a decision tree. The below code shows this implementation. 

```
predicted_dtr = dtc.predict(Xtestscaled)
cm_df_dtr = pd.DataFrame(data = cm(ytest, predicted_dtr), columns = ['lose', 'win'], index = ['lose','win'])
print(accuracy_score(ytest, predicted_dtr))
cm_df_dtr
```

The accuracy for the above model was 0.5406, and the confusion matrix is shown below.

<img width="138" alt="Screen Shot 2022-05-11 at 2 49 27 PM" src="https://user-images.githubusercontent.com/74326062/167924085-55e25fe3-2e55-44d6-8167-2d62d7f84b5e.png">

The next classifier model I ran was a random forest. The below code shows this implementation. 

```
predicted_rfc = rfc.predict(Xtestscaled)
cm_df_rfc = pd.DataFrame(data = cm(ytest, predicted_rfc), columns = ['lose', 'win'], index = ['lose','win'])
print(accuracy_score(ytest, predicted_rfc))
cm_df_rfc
```

The accuracy for the above model was 0.5434, and the confusion matrix is shown below.

<img width="138" alt="Screen Shot 2022-05-11 at 2 49 59 PM" src="https://user-images.githubusercontent.com/74326062/167924188-296f4b1b-3f2a-413c-be02-cd9100369920.png">

The next classifier model I ran was a k-nearest neighbors. The below code shows this implementation. 

```
predicted_knn = knn.predict(Xtestscaled)
cm_df_knn = pd.DataFrame(data = cm(ytest, predicted_knn), columns = ['lose', 'win'], index = ['lose','win'])
print(accuracy_score(ytest, predicted_knn))
cm_df_knn
```

The accuracy for the above model was 0.5262, and the confusion matrix is shown below.

<img width="138" alt="Screen Shot 2022-05-11 at 2 51 14 PM" src="https://user-images.githubusercontent.com/74326062/167924440-eb579e27-0c2a-432c-baea-aa9ca3675c37.png">

## Discussion and Inferences

Of the four models logistic regression had the highest accuracy on the test data at 0.5900. However, looking at the confusion matrix we see this is because it predicted all losses. This might be because the averaged values over 5 games are lower than a lot of the data the model saw in the train set, and now does not see any values that are close to what the model would predict as a win. I think there are ways to improve upon this model, including trying different penalties, or adding a constant\intercept to the model. 

The model with the next highest accuracy on the test data was random forest classifier at 0.5434. While this is a decrease from the train data (accuracy 0.8027) it is above 0.5000 which shows that the model is utilizing the input variables to predict at a higher rate than randomly guessing. In order to improve this model you could look at hyperparameters such as max_depth or min_samples_split. 

The decision tree classifier was just behind random forest at 0.5406, and is  above the 0.5000 threshold. The decision tree model has similar hyperparameters to random forest, and max_depth and min_samples_split could also be used to try and improve this model. 

K-nearest neighbors operated at a 0.5262, the lowest of the models but also above the 0.5000 threshold. This model could be improved by running the model with different *k* values to try and optimize the model. Investigating the different weights with the different *k* values could also be used to improve the model.

Other ways to improve upon these models would be to add more *X* variables. The statisticals values used in these models are very basic but there are a number of additional variables that go into a team's win. Categories such as hits, blocks, face-off percentage, zone entries, takeaways, giveaways, etc. could improve a model's understanding of a team, and therefore their odds to win any given game. 

In the context of the goals of this project, having a model above 0.5000 in accuracy mean that betting on every game would return more money than you lose. Using this model to bet would also require knowledge of how sports betting worked, because some of the predicted winners the model picks are probably also favorites in various sports betting companies. So, implimenting this model it would probably be most beneficial, from a financial standpoint, to wait until the model predicts a winner that is not favored in Vegas.

If someone wanted to research further into NHL statistics they could try and create some sort of wins above replacement (WAR)/value over replacement player (VORP) variable. These variables are more common in other sports such as baseball or basketball and measure how much more value a player adds to their team at their position than a slighly below avergae (or 'replacement') player would in their place. While this statistic has been attempted for NHL players there is still not one gererally accepted formula or variable. If one were to create this variable it could be added to this model to help predict team wins, and could also be used for other individual player bets or fantasy leagues.

## References

Bento, C. (2021, July 18). Decision tree classifier explained in real-life: Picking a vacation destination. Medium. Retrieved May 11, 2022, from https://towardsdatascience.com/decision-tree-classifier-explained-in-real-life-picking-a-vacation-destination-6226b2b60575 

Clark, R. (2018). NHL package¶. NHL Package - sportsipy 0.1.0 documentation. Retrieved May 11, 2022, from https://sportsreference.readthedocs.io/en/stable/nhl.html 

Thanda, A., Anamika Thanda Anamika ThandaOriginally from India, Thanda, A. T. A., Thanda, A., &amp; India, O. from. (2021, December 16). What is logistic regression? A beginner's guide [2022]. CareerFoundry. Retrieved May 11, 2022, from https://careerfoundry.com/en/blog/data-analytics/what-is-logistic-regression/ 

Uberoi, A. (2022, March 10). K-nearest neighbours. GeeksforGeeks. Retrieved May 11, 2022, from https://www.geeksforgeeks.org/k-nearest-neighbours/ 

Vadapalli, P. (2022, April 18). Random forest classifier: Overview, how does it work, pros &amp; cons. upGrad blog. Retrieved May 11, 2022, from https://www.upgrad.com/blog/random-forest-classifier/ 

