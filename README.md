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

Each boxscore table had 42 columns for each game of which I used 15. One of the columns was 'winning_abbr' which showed which team wone that particular game, which will help me create my output or classification variable. The other 14 columns of importance held statistical information from the game, which will be my input variables. They were:

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

The below code was used to create the *Xtrain* data.

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

EXPLAIN CODE

The below code was used to create the *ytain* data.

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

### Classification Methods 

Consider expanding this section with a lot of details from concepts and theories.

Feel free to add a lot of details.

### The actual application of the methods and the validation procedure

Here you can include flow chart diagrams and describe your coding approach.

Suggestion: make sure you have anough computing power.  (Consider Colab Pro)

## Discussion and inferences

Include your perspective and critical thinking. Comment what seems to be working and identify possible future research.

## References
