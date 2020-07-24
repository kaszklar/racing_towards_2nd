# Databricks notebook source
import boto3
import pandas as pd
import numpy as np
import seaborn as sns
from io import StringIO, BytesIO
import datetime
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# COMMAND ----------

# Databricks notebook source
ACCESS_KEY = ""
# Encode the Secret Key as that can contain "/"
SECRET_KEY = "".replace("/", "%2F")
AWS_BUCKET_NAME = ""
MOUNT_NAME = ""

s3 = boto3.client(
    's3',
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY
)

# COMMAND ----------

# create results dataframe
results = "raw/results.csv"
obj_results = s3.get_object(Bucket = AWS_BUCKET_NAME, Key = results) 
results_df = pd.read_csv(obj_results['Body'], parse_dates=True)

#create drivers dataframe
drivers = "raw/drivers.csv"
obj_drivers = s3.get_object(Bucket = AWS_BUCKET_NAME, Key = drivers) 
drivers_df = pd.read_csv(obj_drivers['Body'], parse_dates=True)

#create pit stops dataframe
pitstops = "raw/pit_stops.csv"
obj_pitstops = s3.get_object(Bucket = AWS_BUCKET_NAME, Key = pitstops) 
pitstops_df = pd.read_csv(obj_pitstops['Body'], parse_dates=True)

#create races dataframe
races = "raw/races.csv"
obj_races = s3.get_object(Bucket = AWS_BUCKET_NAME, Key = races) 
races_df = pd.read_csv(obj_races['Body'], parse_dates=True)

#create lap times dataframe
lap_times = "raw/lap_times.csv"
obj_laptimes = s3.get_object(Bucket = AWS_BUCKET_NAME, Key = lap_times) 
laptimes_df = pd.read_csv(obj_laptimes['Body'], parse_dates=True)

# COMMAND ----------

def write_to_katie(df, location):
  session = boto3.Session(
    aws_access_key_id='',
    aws_secret_access_key='',
  )
  AWS_BUCKET_NAME = ""
  MOUNT_NAME = ""  
  
  s3 = session.resource('s3')
  csv_buffer = StringIO()
  df.to_csv(csv_buffer)
  s3 = session.resource('s3')
  s3.Object(AWS_BUCKET_NAME, location).put(Body=csv_buffer.getvalue())

# COMMAND ----------

# convert dates to datetime
races_df.date = pd.to_datetime(races_df['date'], format='%Y-%m-%d')
drivers_df.dob = pd.to_datetime(drivers_df['dob'], format='%Y-%m-%d')

# COMMAND ----------

# MAGIC %md ## pitstops data

# COMMAND ----------

pitstops_df.describe()

# COMMAND ----------

pitstops_df.dtypes

# COMMAND ----------

pitstops_df.isna().sum()

# COMMAND ----------

pitstops_df = pitstops_df.merge(races_df[['raceId','date']])

# COMMAND ----------

pitstops_df.date.min()

# COMMAND ----------

# pitstop information can't be used for q1

# COMMAND ----------



# COMMAND ----------

# MAGIC %md ## laptimes data

# COMMAND ----------

laptimes_df.describe()

# COMMAND ----------

laptimes_df.dtypes

# COMMAND ----------

laptimes_df.isna().sum()

# COMMAND ----------

laptimes_df.position.unique()

# COMMAND ----------

laptimes_df = laptimes_df.merge(races_df[['raceId','date']])

# COMMAND ----------

laptimes_df.date.min()

# COMMAND ----------

#laptimes information not suitable for dates prior to 1996

# COMMAND ----------

# MAGIC %md ## driver data

# COMMAND ----------

drivers_df.describe()

# COMMAND ----------

drivers_df.dtypes

# COMMAND ----------



# COMMAND ----------

# MAGIC %md ### races data

# COMMAND ----------

races_df.describe()

# COMMAND ----------

races_df.isna().sum()

# COMMAND ----------

races_df.dtypes

# COMMAND ----------

# MAGIC %md ## results data

# COMMAND ----------

results_df.describe()

# COMMAND ----------

results_df.dtypes

# COMMAND ----------

# replace \N values with NaNs
results_df.replace({'\\N':np.nan}, inplace=True)

# COMMAND ----------

results_df.isna().sum()

# COMMAND ----------

results_df = results_df.merge(races_df[['raceId','date']], left_on='raceId', right_on='raceId')

# COMMAND ----------

results_df = results_df.sort_values(by=['date'], ascending=False)

# COMMAND ----------

results_df[results_df.fastestLapSpeed.isna()].date.max()

# COMMAND ----------

results_df[results_df.fastestLapSpeed.isna()].date.min()

# COMMAND ----------

# fastestLapSpeed probablyy won't be helpful

# COMMAND ----------

results_df[results_df.fastestLap.isna()].date.max()

# COMMAND ----------

results_df[results_df.fastestLap.isna()].date.min()

# COMMAND ----------

# fastest lap similarly; they probably have a similar distribution

# COMMAND ----------

results_df.position.isna().sum()

# COMMAND ----------

results_df.positionOrder.unique()

# COMMAND ----------

results_df.positionOrder.isna().sum()
# assumption being made: positionOrder can stand in for position

# COMMAND ----------

results_df['rank'].isna().sum()
#can't using ranking as feature

# COMMAND ----------

len(results_df.constructorId.unique())

# COMMAND ----------

heatmap = sns.heatmap(results_df.isnull(), cbar=False)
heatmap.set_title("Heatmap of Missing Values in Results Data, Date Descending")
display(heatmap)

# COMMAND ----------

# convert the fastest lap time from stop watch format to milliseconds

def convert_to_ms(s):
  ''' Convert time in stopwatch format (minute : seconds.milliseconds) 
      to milliseconds. If input is null return null.
  '''
  try:
    stopwatch = [float(x) for x in s.split(":")]
    stopwatch[0] = stopwatch[0]*60
    final_stopwatch = sum(stopwatch) * 1000
  except:
    final_stopwatch=s
  return final_stopwatch

results_df.fastestLapTime = results_df.fastestLapTime.apply(convert_to_ms)

# COMMAND ----------

#drop variables that I'm not interested in 
results_df = results_df.drop(columns=['number', 'position', 'positionText', 'points', 'laps', 'time', 'milliseconds', 'rank', 'statusId'])
results_df.head()

# COMMAND ----------

write_to_katie(results_df.set_index('resultId'), 'final_project/processed/results_df.csv')
write_to_katie(pitstops_df, 'final_project/processed/pitstops_df.csv')
write_to_katie(laptimes_df, 'final_project/processed/laptimes_df.csv')
write_to_katie(drivers_df.set_index('driverId'), 'final_project/processed/drivers_df.csv')

# COMMAND ----------


