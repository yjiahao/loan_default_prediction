from pydantic import BaseModel
from dataclasses import dataclass

@dataclass
class Features(BaseModel):
    credit_policy: int
    purpose: str
    int_rate: float
    installment: float
    log_annual_inc: float
    dti: float
    fico: int
    days_with_cr_line: float
    revol_bal: int
    revol_util: float
    inq_last_6mths: int
    delinq_2yrs: int
    pub_rec: int