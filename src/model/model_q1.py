# Databricks notebook source
dbutils.library.installPyPI("mlflow", "1.0.0")

# COMMAND ----------

import boto3
import pandas as pd
import numpy as np
import datetime

from sklearn.utils import resample
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import mlflow.sklearn

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

# pull in experiment id from databricks experiment
q1_experiment_id = 608390620757909

# COMMAND ----------

# bring in final data
matrix = "final_project/final/matrix.csv"
obj_matrix = s3.get_object(Bucket = AWS_BUCKET_NAME, Key = matrix) 
mat = pd.read_csv(obj_matrix['Body'], parse_dates=['date'], index_col='resultId')

# COMMAND ----------

# drop data post 2011
pre2011 = mat[mat.date < datetime.datetime(year=2011, month=1, day=1)]

# COMMAND ----------

# check class balance
pre2011.target.value_counts()

# COMMAND ----------

# use sklearn utils resample to balance out the classes
majority_df = pre2011[pre2011.target==0]
minority_df = pre2011[pre2011.target==1]

upsampled_minority = resample(minority_df, 
                                 replace=True,     # sample with replacement
                                 n_samples=19931,    # to match majority class
                                 random_state=15) # reproducible results

pre2011 = pd.concat([majority_df, upsampled_minority])

#confirm class balance
pre2011.target.value_counts()

# COMMAND ----------

# drop unwanted columns
pre2011 = pre2011[['driverAge','constructorId','grid','target']]
target = pre2011[['target']]

# COMMAND ----------

# confirm no missing values
pre2011.isna().sum()

# COMMAND ----------

# scale continuous features so that we can compare the coefficients
scaler = StandardScaler()
pre2011_features = pd.DataFrame(scaler.fit_transform(pre2011[['driverAge', 'grid']]),columns = ['driverAge','grid'], index=pre2011.index)

# add in the unscaled, categorical feture back to the feature matrix
pre2011_features['constructorId'] = pre2011['constructorId']

# COMMAND ----------

# MAGIC %md ## experiment

# COMMAND ----------

def log_logit(experimentID, run_name, endog, exog):
  with mlflow.start_run(experiment_id=experimentID, run_name=run_name) as run:
    # run model
    logit = sm.Logit(endog, exog)
    logit_res = logit.fit()
    
    # log model
    mlflow.sklearn.log_model(logit, "logistic-regression-model")
    print(logit_res.summary())
    
    # run metric
    logit_r2 = logit_res.prsquared
    
    # run marg effects
    logit_coeff = logit_res.params.to_dict()
    for s in logit_coeff.keys():
      new = s+"_coeff"
      logit_coeff[new] = logit_coeff[s]
      del logit_coeff[s]
    
    logit_pvals = logit_res.pvalues.to_dict()
    for s in logit_pvals.keys():
      new = s+"_pval"
      logit_pvals[new] = logit_pvals[s]
      del logit_pvals[s]
    
    # log metrics
    mlflow.log_metric("r2", logit_r2)
    
    # log marg effects
    [mlflow.log_param(param, value) for param, value in logit_coeff.items()]
    [mlflow.log_param(param, value) for param, value in logit_pvals.items()]
    
    return run.info.run_uuid

# COMMAND ----------

log_logit(q1_experiment_id, "+ constructorId", endog = target, exog = pre2011_features['constructorId'])

# COMMAND ----------

log_logit(q1_experiment_id, "+ grid", endog = target, exog = pre2011_features[['constructorId','grid']])

# COMMAND ----------

log_logit(q1_experiment_id, "+ driverAge", endog = target, exog = pre2011_features[['constructorId','grid','driverAge']])

# COMMAND ----------


