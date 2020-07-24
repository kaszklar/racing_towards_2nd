# Databricks notebook source
import boto3
import pandas as pd
import numpy as np
from io import StringIO, BytesIO
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# COMMAND ----------

# Databricks notebook source
ACCESS_KEY = ""
# Encode the Secret Key as that can contain "/"
SECRET_KEY = "".replace("/", "%2F")
AWS_BUCKET_NAME = "kea2143"
MOUNT_NAME = "kea2143"

s3 = boto3.client(
    's3',
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY
)

# COMMAND ----------

def write_to_katie(df, location):
  session = boto3.Session(
    aws_access_key_id='',
    aws_secret_access_key='',
  )
  AWS_BUCKET_NAME = "kea2143"
  MOUNT_NAME = "kea2143"  
  
  s3 = session.resource('s3')
  csv_buffer = StringIO()
  df.to_csv(csv_buffer)
  s3 = session.resource('s3')
  s3.Object(AWS_BUCKET_NAME, location).put(Body=csv_buffer.getvalue())

# COMMAND ----------

# create results dataframe
results = "final_project/processed/results_df.csv"
obj_results = s3.get_object(Bucket = AWS_BUCKET_NAME, Key = results) 
results_df = pd.read_csv(obj_results['Body'])

#create lap times dataframe
lap_times = "final_project/processed/laptimes_df.csv"
obj_laptimes = s3.get_object(Bucket = AWS_BUCKET_NAME, Key = lap_times) 
laptimes_df = pd.read_csv(obj_laptimes['Body'])

#create pit stops dataframe
pitstops = "final_project/processed/pitstops_df.csv"
obj_pitstops = s3.get_object(Bucket = AWS_BUCKET_NAME, Key = pitstops) 
pitstops_df = pd.read_csv(obj_pitstops['Body'])

#create drivers dataframe
drivers = "final_project/processed/drivers_df.csv"
obj_drivers = s3.get_object(Bucket = AWS_BUCKET_NAME, Key = drivers) 
drivers_df = pd.read_csv(obj_drivers['Body'])

# COMMAND ----------

# convert dates to datetime
results_df.date = pd.to_datetime(results_df['date'], format='%Y-%m-%d')
drivers_df.dob = pd.to_datetime(drivers_df['dob'], format='%Y-%m-%d')

# COMMAND ----------

results_df.head()

# COMMAND ----------

# add driver dob to results df
results_df = results_df.set_index('driverId').join(drivers_df[['driverId','dob']].set_index('driverId'), on='driverId')
results_df.reset_index(inplace=True)

# calculate drivers age at time of race
age = pd.Series([int((results_df.date[x] - results_df.dob[x]).days /365.25) for x in results_df.index])
results_df['driverAge'] = age

# COMMAND ----------

results_df.head()

# COMMAND ----------

### create features for lap position mean and variance
lap_position_info = laptimes_df.groupby(['raceId','driverId'])['position'].agg(['min','max','mean','var'])
lap_position_info = lap_position_info.rename(columns={'min':'min_lap_pos', 'max':'max_lap_pos', 'mean':'avgLapPosition', 'var':'lapPositionVar'})

# COMMAND ----------

results_df.head()

# COMMAND ----------

# create dataframe that will eventually hold target and features
mat = results_df.set_index(['raceId','driverId']).join(lap_position_info[['avgLapPosition','lapPositionVar']])

# COMMAND ----------

# create avg milliseconds each driver spent at the pitstop for each race & join to mat
avg_ms_pitstops = pd.DataFrame(pitstops_df.groupby(['driverId','raceId'])['milliseconds'].agg('mean')).rename(columns={'milliseconds':'avgPitMs'})
mat = mat.reset_index().merge(avg_ms_pitstops.reset_index(), on = ['driverId', 'raceId'], how='left')

# COMMAND ----------

mat.head()

# COMMAND ----------

# recode target variable: positionOrder
mat['target'] = np.where(mat.positionOrder == 2, 1, 0)

# COMMAND ----------

# drop unnecessary columns & rename the index col
mat.drop(columns=['dob', 'positionOrder'], inplace=True)

# COMMAND ----------

mat = mat.sort_values(by=['date'], ascending=False)

# COMMAND ----------

less2011_df = mat[mat.date < datetime.datetime(year=2011, month=1, day=1)]
greater2011_df = mat[mat.date >= datetime.datetime(year=2011, month=1, day=1)]

# COMMAND ----------

mat.head()

# COMMAND ----------

mat.set_index('resultId', inplace=True)

# COMMAND ----------

write_to_katie(mat, 'final_project/final/matrix.csv')

# COMMAND ----------



