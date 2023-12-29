import pickle
import sklearn
import imblearn
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request, Form, Depends
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from features import Features

# to start up the app: uvicorn main:app --reload --port=5000
# TODOS:
# see if theres a better solution to shorten the forms input (like create a class or some shit)
# render different templates when predicting using different models for html

# load the models with pickle pip install -U Jinja2
with open('models.pkl', 'rb') as f:
  best_log_reg, best_rf, best_knn, best_xgb = pickle.load(f)

# create app object
app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

# set up templating
templates = Jinja2Templates(directory='templates')

@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse('index.html', {'request': request})

@app.get('/{name}')
async def welcomel(name: str):
   return {'Welcome to my model': f'{name}'}

# endpoint for using logistic regression to predict
@app.post('/log_reg')
async def testing(
   request: Request,
   credit_policy: int = Form(...),
   purpose: str = Form(...),
   int_rate: float = Form(...),
   installment: float = Form(...),
   log_annual_inc: float = Form(...),
   dti: float = Form(...),
   fico: int = Form(...),
   days_with_cr_line: int = Form(),
   revol_bal: int = Form(...),
   revol_util: float = Form(...),
   inq_last_6mths: int = Form(...),
   delinq_2yrs: int = Form(...),
   pub_rec: int = Form(...)
):
   
   data = {
      "credit_policy": credit_policy,
      "purpose": purpose,
      "int_rate": int_rate,
      "installment": installment,
      "log_annual_inc": log_annual_inc,
      "dti": dti,
      "fico": fico,
      "days_with_cr_line": days_with_cr_line,
      "revol_bal": revol_bal,
      "revol_util": revol_util,
      "inq_last_6mths": inq_last_6mths,
      "delinq_2yrs": delinq_2yrs,
      "pub_rec": pub_rec
   }

   data = pd.DataFrame(data, index=[0])

   threshold = 0.5
   probability = best_log_reg.predict_proba(data).tolist()[0][1]
   default = (probability > threshold)
   model = 'Logistic Regression'

   if default:
      pred = 'Borrower will default'
   else:
      pred = 'Borrower will not default'

   return templates.TemplateResponse(
      name='model.html',
      request=request,
      context={'pred': pred, 'probability': probability, 'threshold': threshold, 'model': model}
   )

# endpoint for using random forest to predict
@app.post('/rf')
async def testing(
   request: Request,
   credit_policy: int = Form(...),
   purpose: str = Form(...),
   int_rate: float = Form(...),
   installment: float = Form(...),
   log_annual_inc: float = Form(...),
   dti: float = Form(...),
   fico: int = Form(...),
   days_with_cr_line: int = Form(),
   revol_bal: int = Form(...),
   revol_util: float = Form(...),
   inq_last_6mths: int = Form(...),
   delinq_2yrs: int = Form(...),
   pub_rec: int = Form(...)
):
   
   data = {
      "credit_policy": credit_policy,
      "purpose": purpose,
      "int_rate": int_rate,
      "installment": installment,
      "log_annual_inc": log_annual_inc,
      "dti": dti,
      "fico": fico,
      "days_with_cr_line": days_with_cr_line,
      "revol_bal": revol_bal,
      "revol_util": revol_util,
      "inq_last_6mths": inq_last_6mths,
      "delinq_2yrs": delinq_2yrs,
      "pub_rec": pub_rec
   }

   data = pd.DataFrame(data, index=[0])

   threshold = 0.3
   probability = best_rf.predict_proba(data).tolist()[0][1]
   default = (probability > threshold)

   model = 'Random Forest Classifier'

   if default:
      pred = 'Borrower will default'
   else:
      pred = 'Borrower will not default'

   return templates.TemplateResponse(
      name='model.html',
      request=request,
      context={'pred': pred, 'probability': probability, 'threshold': threshold, 'model': model}
   )


# endpoint for using knn to predict
@app.post('/knn')
async def testing(
   request: Request,
   credit_policy: int = Form(...),
   purpose: str = Form(...),
   int_rate: float = Form(...),
   installment: float = Form(...),
   log_annual_inc: float = Form(...),
   dti: float = Form(...),
   fico: int = Form(...),
   days_with_cr_line: int = Form(),
   revol_bal: int = Form(...),
   revol_util: float = Form(...),
   inq_last_6mths: int = Form(...),
   delinq_2yrs: int = Form(...),
   pub_rec: int = Form(...)
):
   
   data = {
      "credit_policy": credit_policy,
      "purpose": purpose,
      "int_rate": int_rate,
      "installment": installment,
      "log_annual_inc": log_annual_inc,
      "dti": dti,
      "fico": fico,
      "days_with_cr_line": days_with_cr_line,
      "revol_bal": revol_bal,
      "revol_util": revol_util,
      "inq_last_6mths": inq_last_6mths,
      "delinq_2yrs": delinq_2yrs,
      "pub_rec": pub_rec
   }

   data = pd.DataFrame(data, index=[0])

   threshold = 0.4
   probability = best_knn.predict_proba(data).tolist()[0][1]
   default = (probability > threshold)

   model = 'K Nearest Neighbours Classifier'

   if default:
      pred = 'Borrower will default'
   else:
      pred = 'Borrower will not default'

   return templates.TemplateResponse(
      name='model.html',
      request=request,
      context={'pred': pred, 'probability': probability, 'threshold': threshold, 'model': model}
   )

# endpoint for using xgb classifier to predict
@app.post('/xgb')
async def testing(
   request: Request,
   credit_policy: int = Form(...),
   purpose: str = Form(...),
   int_rate: float = Form(...),
   installment: float = Form(...),
   log_annual_inc: float = Form(...),
   dti: float = Form(...),
   fico: int = Form(...),
   days_with_cr_line: int = Form(),
   revol_bal: int = Form(...),
   revol_util: float = Form(...),
   inq_last_6mths: int = Form(...),
   delinq_2yrs: int = Form(...),
   pub_rec: int = Form(...)
):
   
   data = {
      "credit_policy": credit_policy,
      "purpose": purpose,
      "int_rate": int_rate,
      "installment": installment,
      "log_annual_inc": log_annual_inc,
      "dti": dti,
      "fico": fico,
      "days_with_cr_line": days_with_cr_line,
      "revol_bal": revol_bal,
      "revol_util": revol_util,
      "inq_last_6mths": inq_last_6mths,
      "delinq_2yrs": delinq_2yrs,
      "pub_rec": pub_rec
   }

   data = pd.DataFrame(data, index=[0])

   threshold = 0.3
   probability = best_xgb.predict_proba(data).tolist()[0][1]
   default = (probability > threshold)

   model = 'XGB Classifier'

   if default:
      pred = 'Borrower will default'
   else:
      pred = 'Borrower will not default'

   return templates.TemplateResponse(
      name='model.html',
      request=request,
      context={'pred': pred, 'probability': probability, 'threshold': threshold, 'model': model}
   )