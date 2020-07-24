# Data Dictionary

This dataset is located in the s3 bucket kea2143 under `final_project/final/matrix.csv`.

| Field Name      | Data Type | Variable Type | Description                                                  |
| --------------- | --------- | :------------ | ------------------------------------------------------------ |
| driverId        | integer   | N/A           | Driver's unique ID                                           |
| raceId          | integer   | N/A           | Race's unique ID                                             |
| constructorId   | integer   | Discrete      | Unique constructor id. The constructor is the combination of the engine and the chassis. |
| grid            | integer   | Ordinal       | The driver's location in the starting grid for that race     |
| driverAge       | Float64   | Continuous    | The age of the driver at the time of the race.               |
| lapPositionVar  | Float64   | Continuous    | The variance in the driver's position per lap in the race.   |
| avgLapPosition  | Float64   | Continuous    | The average of the driver's position per lap of the race.    |
| fastestLap      | Integer   | Ordinal       | Which lap of the race was the fastest per driver per race.   |
| fastestLapTime  | Float64   | Continuous    | The time of the driver's fastest lap per race in milliseconds. |
| fastestLapSpeed | Float64   | Continuous    | The speed of the driver's fastest lap per race in km/h.      |
| date            | Datetime  | N/A           | Date of the race in format YYYY-MM-DD.                       |
| avgPitMs        | Float64   | Continuous    | The number of milliseconds the driver spent in the pit per race, averaged across all laps. |
| resultId        | integer   | N/A           | Index of the data.                                           |
| target          | integer   | Discrete      | A driver's place in a given race.                            |



#### Notes 

Fields are marked N/A for variable type when they will not be used for modeling.