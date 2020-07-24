# Databricks notebook source
import boto3
import pandas as pd
import datetime
import tempfile

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import r2_score

from sklearn.metrics import confusion_matrix
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
#dbutils.library.installPyPI("mlflow", "1.0.0")
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

# bring in final data
matrix = "final_project/final/matrix.csv"
obj_matrix = s3.get_object(Bucket = AWS_BUCKET_NAME, Key = matrix) 
mat = pd.read_csv(obj_matrix['Body'], parse_dates=['date'], index_col='resultId')

# COMMAND ----------

# define columns to be standardized & dropped
std_cols = ['grid','fastestLap','fastestLapTime','fastestLapSpeed','driverAge','avgLapPosition','lapPositionVar','avgPitMs']
remove_cols = ['driverId','raceId','date','target']

# COMMAND ----------

# drop data prior to 2011
post2011 = mat[mat.date >= datetime.datetime(year=2011, month=1, day=1)]

# COMMAND ----------

post2011.shape

# COMMAND ----------

# check for missing values
post2011.isna().sum()

# COMMAND ----------

# drop missing instead of imputation 
post2011 = post2011.dropna()

# COMMAND ----------

# scale continuous features so that we can compare the coefficients
scaler = StandardScaler()
post2011[std_cols] = scaler.fit_transform(post2011[std_cols])

# COMMAND ----------

post2011.target.value_counts()

# COMMAND ----------

# use sklearn utils resample to balance out the classes
majority_df = post2011[post2011.target==0]
minority_df = post2011[post2011.target==1]

upsampled_minority = resample(minority_df, 
                                 replace=True,     # sample with replacement
                                 n_samples=3391,    # to match majority class
                                 random_state=15) # reproducible results

post2011 = pd.concat([majority_df, upsampled_minority])

#confirm class balance
post2011.target.value_counts()

# COMMAND ----------

# define features, target, training & test
feature_cols = post2011.columns.difference(remove_cols)
X_train, X_test, y_train, y_test = train_test_split(post2011[feature_cols], post2011['target'], test_size=0.33, random_state=15)

# COMMAND ----------

# MAGIC %md ### experiments

# COMMAND ----------

# pull in experiment id from databricks experiment
q2_experiment_id = 967899901878080

# COMMAND ----------

# define an experiment run
def log_logit(experimentID, run_name, params, X_train, X_test, y_train, y_test):
  with mlflow.start_run(experiment_id=experimentID, run_name=run_name) as run:
    # run model
    log = LogisticRegression(**params)
    log.fit(X_train, y_train)
    preds = log.predict(X_test)
    
    # log model
    mlflow.sklearn.log_model(log, "logistic-regression-model")
    
    # log model parameters & coefs
    [mlflow.log_param(param, value) for param, value in params.items()]
    
    coeffs = {}
    for i, col in enumerate(X_train.columns):
      coeffs[col] = log.coef_[0][i]
    [mlflow.log_param(param,value) for param,value in coeffs.items()]
    
    # run metrics
    logit_f1 = f1_score(y_test, preds)
    confusion = confusion_matrix(y_test, preds)
    logit_r2 = r2_score(y_test, preds)
    
    # log metrics
    mlflow.log_metric("f1", logit_f1)
    mlflow.log_metric("r2", logit_r2)
    t_n, f_p, f_n, t_p = confusion.ravel()
    mlflow.log_metric("tn", t_n)
    mlflow.log_metric("fp", f_p)
    mlflow.log_metric("fn", f_n)
    mlflow.log_metric("tp", t_p)

    
    # plot confusion matrix
    fig, ax = plt.subplots()
    sns.heatmap(confusion, cmap='Blues',annot=True)
    plt.ylabel("True labels")
    plt.xlabel("Predicted labels")
    title = run_name + " Confusion Matrix"
    plt.title(title)
    
    # log confusion matrix to temp file
    temp = tempfile.NamedTemporaryFile(prefix="confusionmatrix-", suffix=".png")
    temp_name = temp.name
    try:
      fig.savefig(temp_name)
      mlflow.log_artifact(temp_name, "confusionmatrix.png")
    finally:
      temp.close() # Delete the temp file
    
    return run.info.run_uuid

# COMMAND ----------

feature_cols

# COMMAND ----------

# define starting cols from q1
q1_features = ['constructorId','driverAge','grid']

# define LogisticRegression parameters to try
params = {
  'params1' : {'penalty' : 'l2', 'solver' : 'lbfgs'}, #ridge penalty using lbfgs solver
  'params2' : {'solver' : 'saga', 'penalty' : 'l1'}, #lasso penalty using saga solver
  'params3' : {'solver' : 'sag', 'penalty' : 'l2'}, # ridge using sag solver
  'params4' : {'penalty' : 'l1', 'solver' : 'liblinear'} #lasso using liblinear solver
}
 

# COMMAND ----------

# iterate through and progressively add additional features. for each addition, run previously defined parameter options
log_logit(q2_experiment_id, "q1", param_set, X_train[q1_features], X_test[q1_features], y_train, y_test)
for col in feature_cols.difference(q1_features):
  q1_features.append(col)
  for _, param_set in params.items():
    log_logit(q2_experiment_id, "+"+col, param_set, X_train[q1_features], X_test[q1_features], y_train, y_test)
