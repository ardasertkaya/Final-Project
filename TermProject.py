import pandas as pd
import numpy as np
from statsmodels.tsa.api import SimpleExpSmoothing, Holt   
import matplotlib.pyplot as plt 
import warnings
warnings.filterwarnings("ignore")

def estimate_naive(df, seriesname):
     numbers = np.asarray ( df[seriesname] )
     return float( numbers [-1])

def estimate_moving_average(df,seriesname,windowsize):
    avg = df[seriesname].rolling(windowsize).mean().iloc[-1]
    return avg

def estimate_ses(df, seriesname, alpha=0.9):
    numbers = np.asarray(df[seriesname])
    estimate = SimpleExpSmoothing(numbers).fit(smoothing_level=alpha,optimized=False).forecast(1)
    return estimate

def estimate_holt(df, seriesname, alpha=0.9, slope=0.1):
    numbers = np.asarray(df[seriesname])
    model = Holt(numbers)
    fit = model.fit(alpha,slope)
    estimate = fit.forecast(1)[-1]
    return estimate

dataframe= pd.read_csv("TermProject.csv", sep=";")
turk="Turkey"
greek="Greece"
testing_df= dataframe.head(len(dataframe)-1)

# Naive estimation

naive_turk= round(estimate_naive(testing_df, turk),0)
print("2019,naive estimation for Turkey:", naive_turk)

naive_greek= round(estimate_naive(testing_df, greek),0)
print("2019,naive estimation for Greece:",naive_greek)

# Moving average

movingaverage_turk = round(estimate_moving_average(testing_df,turk,3),0)
print("2019's moving average estimation for ", turk, ":", movingaverage_turk)

movingaverage_greek = round(estimate_moving_average(testing_df,greek,3),0)
print("2019's moving average estimation for ", greek, ":",movingaverage_greek)

# Simple Exponential Smoothing

alpha = 0.9
ses_turk = round ( estimate_ses(testing_df, turk, alpha)[0], 0)
print("2019's exponential smoothing estimation with alpha =", alpha,"for",turk,": ", ses_turk)

ses_greek = round ( estimate_ses(testing_df, greek, alpha)[0], 0)
print("2019's exponential smoothing estimation with alpha =", alpha,"for",greek, ": ", ses_greek)

# Holt Estimation

alpha = 0.9
slope = 0.1
holt_turk = round(estimate_holt(testing_df,turk,alpha, slope),0)
print("2019's holt trend estimation with alpha =", alpha, ", and slope =", slope, "for",turk,": ", holt_turk)

holt_greek = round(estimate_holt(testing_df,greek,alpha, slope),0)
print("2019's holt trend estimation with alpha =", alpha, ", and slope =", slope,"for",greek, ": ", holt_greek)

# Since holt estimation gives the closest value, we will use it as the prediction method.
plt.plot(dataframe.YEARS,dataframe.Turkey, linestyle="-")
plt.title("Numbers of Arrivals,Turkey")
plt.xlabel("YEARS")
plt.ylabel("NUMBERS")
plt.show()

plt.plot(dataframe.YEARS, dataframe.Greece, linestyle="-")
plt.title("Numbers of Arrivals,Greece")
plt.xlabel("YEARS")
plt.ylabel("NUMBERS")
plt.show()
# Real numbers of arrivals for 2019

turk_2019= dataframe.ix[12,"Turkey"]
greek_2019= dataframe.ix[12,"Greece"]
print("In 2019, numbers of arrivals in Turkey : ", turk_2019, "people")
print("In 2019, numbers of arrivals in Greece : ", greek_2019, "people")

# Forecasting number of arrivals in 2020 with Holt estimation
newdataframe = dataframe.head(len(dataframe))
turk="Turkey"
greek="Greece"
alpha = 0.9
slope = 0.1
holt_turk = round(estimate_holt(newdataframe,turk,alpha, slope),0)
print("2020's holt trend estimation with alpha =", alpha, ", and slope =", slope, "for",turk,": ", holt_turk)

holt_greek = round(estimate_holt(newdataframe,greek,alpha, slope),0)
print("2020's holt trend estimation with alpha =", alpha, ", and slope =", slope,"for",greek, ": ", holt_greek)

#Calculating growth rate

grow_turk= (holt_turk - turk_2019)/turk_2019
print("Growth rate of numbers of arrivals for Turkey : ", "%",round(grow_turk*100,2))

grow_greek= (holt_greek - greek_2019)/greek_2019
print("Growth rate of numbers of arrivals for Greece : ","%",round(grow_greek*100,2))