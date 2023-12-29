# learn FastAPI and create a simple API
# learn templating using FastAPI
# use Bootstrap, HTML and CSS for the frontend

import pickle
import sklearn
import imblearn
import numpy as np
import pandas as pd

# load the models
with open('models.pkl', 'rb') as f:
  best_log_reg, best_rf, best_knn, best_xgb = pickle.load(f)

print(best_log_reg)