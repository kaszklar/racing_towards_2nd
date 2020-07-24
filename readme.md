# Racing Towards Second Place

### Why does a F1 driver arrive in second place between 1950 and 2010?

After upsampling to account for a very small minority class, I used  the logit implementation from StatsModels and eventually arrived at a model that has an **r<sup>2</sup> of .241** - meaning that the independent variables used explain approximately 24% of the variation that we see in the dependent variable (whether or not the result was a second place). 
The variables chosen were done so based on theory and availability. For example, it did not make theoretical sense to use the driver's number in the model, or the number of laps since all drivers in a race would typically have the same number of laps. The number of points awarded to each result does not make sense as points are awarded based on result position (the most go to first place, the second most to second, etc). Additional variables based on the pit stops and laps were not available prior to 2011 and 2004 respectively, so they could not be used for a half-century's worth of data. In retrospect, I believe I was a bit too conservative in making these choices, as I would have still liked to compare the effects amongst different covariates.

In the end I settled on using the driver's age at the time of the race (perhaps with age comes more experience and a greater chance at doing well?), the constructor id (which is a categorical variable that represents the combination of the engine and the chassis used), and the grid position (where the driver is located in the starting grid). The results of the analysis can be seen in Table 1 below, and the text-based MLFlow run assets can be found in `resources/reports/q1_runs.csv`

I determined which of these variables was the most important by using a forward looking methodology, or adding in variables one by one. As can be seen in Table 1, the greatest jump in r<sup>2</sup> came with the addition of the grid variable. 

#### Table 1

|                            | model 1              | model 2                | model 3               |
| -------------------------- | -------------------- | ---------------------- | --------------------- |
| constructorId <sup>†</sup> | -.0046 ***           | -.0043 ***             | -.0045 ***            |
| grid                       |                      | - 1.4696 ***           | - 1.4649 ***          |
| driverAge                  |                      |                        | .1215 ***             |
|                            | r<sup>2 </sup> .0137 | r<sup>2</sup>  of .239 | r<sup>2</sup> of .241 |

<sup>†</sup> non-standardized 

In this case, looking at the significance levels is not enough to tell us what is "important." Rather, it is the large jump in r<sup>2</sup> that we get with the addition of grid, as well as the size of grid's standardized coefficient.

The story behind grid's importance is simple and disappointing (at least in this case). The grid is the physical location at the starting line that the car is located in, prior to the fire of the gun. As there are are often quite a few cars, they must be staggered in rows - so grid position 1 has the most favorable spot; front row, inside the track (which is a shorter distance and a valuable place within the ring of the track). It stands to reason that there is not much movement of the cars during the race, so that the grid position you start in is very often your finishing position, or quite nearly so. Thus, I do not view this as an "explanation" for a second place finish - but rather an association that we see.

### Building upon theory from question 1, predict whether a driver between 2011 and 2017 will come in second place.

In order to predict a second place finish between 2011 and 2017, I continued with my forward-looking methodology, continuously adding additional variables engineered in `src/features/features.py`. These additional variables were avgLapPosition, avgPitMs, fastestLap, fastestLapSpeed,fastestLapTime, and lapPositionVar. For an explanation of each of these, please refer to the [dictionary](references/dictionary.md). These features were also selected based on data availability, as well as some theories, such as - "Perhaps drivers who finish second tend to vary in the lap by lap rankings in a race." Quite interestingly, variables having to do with raw measures of time (average milliseconds spent in the pit, fastest lap time) were the least predictive and even insignificantly so. For the variable fastestLapTime, this could be due to different lengths of the various circuits over the years. I switched to the Scikit-learn framework, the gold standard for predictive analytics in python, and made my model selection based on the f<sub>1</sub> metric.

The final model used the liblinear solver the LASSO (l1) penalty, all coefficients, and had an f<sub>1</sub> score of .908. The text-based MLFlow assets can be found in `resources/reports/q2_runs.csv`.

#### Table 2: Coefficients from final predictive model

|                           | coef    |
| ------------------------- | ------- |
| avgLapPosition            | -6.1686 |
| avgPitMs                  | 0.0068  |
| constructorId<sup>†</sup> | 0.0037  |
| driverAge                 | 0.2291  |
| fastestLap                | 0.2088  |
| fastestLapSpeed           | 0.0321  |
| fastestLapTime            | -0.0756 |
| grid                      | 0.4361  |
| lapPositionVar            | 0.9974  |

<sup>†</sup> non-standardized 

#### Metrics

|                | value |
| -------------- | ----- |
| r<sup>2 </sup> | 0.611 |
| f<sub>1 </sub> | 0.908 |

**Classification Matrix**

|				 | actually positive | actually negative |
| -------------- | ----------------- | ----------------- |
| pred positive  | 1082				 | 180				 |
| pred negative  | 38				 | 939				 |

The most important variable from question 1, grid, is still present. However, it's importance has diminished greatly, as can be seen by comparing it's standardized coefficient in Table 1: model 3  (- 1.4696) to the one in Table 2 (0.4361). Not only has the magnitude changed quite a bit, but the sign has as well meaning it now predicts in the opposite direction in this binary classification.

Initially I was pleased to achieve a 0.611 r<sup>2 </sup> and a 0.908 f<sub>1 </sub>. However, I realized that most of this "great predictive power" is all tied up in the average lap position. I have fallen into a trap again in using lap positioning - it turns out that they do not in fact move much in the line up as the race progresses. Just as in grid position in question 1, If they begin in a desirable position they tend to end in a desirable position.