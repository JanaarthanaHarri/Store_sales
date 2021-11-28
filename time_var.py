import datetime
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

# time variables
# data time range to train on the full training set
full_train_start_day = datetime.datetime(2015, 6, 16)
full_train_end_day = datetime.datetime(2017, 8, 15)

# data time range for train/validation split 
train_start_day = full_train_start_day
train_end_day = datetime.datetime(2017, 7, 30)
val_start_day = datetime.datetime(2017, 7, 31)
val_end_day = datetime.datetime(2017, 8, 15)
# can be smart to set val_end_day to (2017, 7, 31) or (2017, 8, 1) when testing a change or debugging

# data time range of test set
test_start_day = datetime.datetime(2017, 8, 16)
test_end_day = datetime.datetime(2017, 8, 31)

if full_train_start_day > full_train_end_day:
    raise ValueError("full_train_start_day must be less than full_train_end_day")

# other key variables
max_lag = 7

mod_1 = LinearRegression()
mod_2 = XGBRegressor()

hybrid_forecasting_type = "day_by_day_refit_all_days"