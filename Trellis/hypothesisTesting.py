import pandas as pd 
from datetime import datetime, timedelta
import matplotlib.pyplot as mp
import matplotlib as mpl
import numpy as np
import statsmodels.api as sm
from scipy.stats import linregress
from scipy import stats
import math 

#30
def sumOfDeviation(x):
    avg = np.mean(x)
    sum = 0
    for i in x:
        sum += (i-avg)**2
    return sum

def sumOfError (Regression,x,y):
    sum = 0
    for i in     range(len(y)):
        sum+=(y[i]-(Regression.slope*x[i]+Regression.intercept))**2
    return sum
        
def SeCoefficient (Regression,x,y):
    return math.sqrt(sumOfError(Regression,x,y)/((len(x)-2)*sumOfDeviation(x)))


def constantP(beforeDealProfit,afterDealProfit):
    ybeforeDealProfit = beforeDealProfit.to_numpy()
    accumulator = []
    for i in range(beforeDealProfit.count()):
        accumulator.append(i+1)
    xbeforeDealProfit = np.array(accumulator)
    RegressionA = linregress(xbeforeDealProfit,ybeforeDealProfit)

    yafterDealProfit = afterDealProfit.to_numpy()
    accumulator = []
    for i in range(afterDealProfit.count()):
        accumulator.append(i+1)
    xafterDealProfit = np.array(accumulator)
    RegressionB = linregress(xafterDealProfit,yafterDealProfit)

    constantGuy = RegressionB.intercept - RegressionA.intercept
    pooledError = (sumOfError(RegressionA,xbeforeDealProfit,ybeforeDealProfit)+sumOfError(RegressionB,xafterDealProfit,yafterDealProfit))/((beforeDealProfit.count()+afterDealProfit.count())-4)
    SEValue = math.sqrt(pooledError * ((1/afterDealProfit.count()) + (1/beforeDealProfit.count()) + ((np.mean(xbeforeDealProfit)**2)/sumOfDeviation(xbeforeDealProfit))+((np.mean(xafterDealProfit)**2)/sumOfDeviation(xafterDealProfit))))
    return stats.t.sf(abs(constantGuy/SEValue), beforeDealProfit.count() + afterDealProfit.count()-2) * 2

def slopeP(beforeDealProfit,afterDealProfit):
    ybeforeDealProfit = beforeDealProfit.to_numpy()
    accumulator = []
    for i in range(beforeDealProfit.count()):
        accumulator.append(i+1)
    xbeforeDealProfit = np.array(accumulator)
    RegressionA = linregress(xbeforeDealProfit,ybeforeDealProfit)

    yafterDealProfit = afterDealProfit.to_numpy()
    accumulator = []
    for i in range(afterDealProfit.count()):
        accumulator.append(i+1)
    xafterDealProfit = np.array(accumulator)
    RegressionB = linregress(xafterDealProfit,yafterDealProfit)
    se_coefficient_a = SeCoefficient(RegressionA,xbeforeDealProfit,ybeforeDealProfit)
    se_coefficient_b = SeCoefficient(RegressionB,xafterDealProfit,yafterDealProfit)
    coefficientSlope = RegressionB.slope - RegressionA.slope
    TValue = coefficientSlope/math.sqrt(se_coefficient_a**2+se_coefficient_b**2)
    return stats.t.sf(abs(TValue), beforeDealProfit.count() + afterDealProfit.count()-2) * 2
    
    

x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39])
yA = np.array([13.47264244224,7.83596710395,16.16566148297,11.69379417238,6.47808934715,13.54142842089,11.72734269439,15.64885615487,15.32619483197,17.86064771923,19.06303410948,18.18973625630,22.57924965642,17.85486017633,25.43002741595,23.52799146391,25.77953083330,26.01250099523,27.88022480992,24.31963386662,32.06605139282,27.91002926925,25.55236360121,30.14232786745,27.93618878140,28.39962661359,33.08663291608,27.08191078536,33.02675670369,33.56041606880,31.73005540945,35.57886509898,33.22213848242,37.80297003040,36.65451428660,37.52814845927,34.00787591858,38.91543553712,39.22747405465])  # Dependent variable
yB = np.array([7.90927961501,10.61532724544,10.42163177190,15.11624927817,10.65361635146,11.03364750692,19.23162150723,11.90895557903,10.09663078460,19.51099663957,20.12830189949,20.70804427779,21.07535698440,18.11969178464,22.41840182292,22.38900516669,30.55638418276,30.29607551933,23.92053102559,23.06118816478,33.65013442932,30.80713882025,32.24124736643,31.14005796778,33.99441378864,31.50638000837,37.05197399437,41.58040891169,36.15836862404,35.00466471152,41.68644412569,43.95584832551,38.09102131411,39.71880540217,38.95617699841,43.43712124162,42.98664448776,46.01421802177,46.75994328170])
dfA = pd.DataFrame({'X-Values': x, 'Y-Values': yA}, columns=['X-Values', 'Y-Values'])
dfB = pd.DataFrame({'X-Values': x, 'Y-Values': yB}, columns=['X-Values', 'Y-Values'])
# print ("Constant P-Value:",constantP(dfA['Y-Values'],dfB['Y-Values']))
# print ("Constant P-Value:",slopeP(dfA['Y-Values'],dfB['Y-Values']))

# RegressionA = linregress(x,yA)
# se_coefficient_a = SeCoefficient(RegressionA,x,yA)

# RegressionB = linregress(x,yB)
# se_coefficient_b = SeCoefficient(RegressionB,x,yB)
# coefficientSlope = RegressionB.slope - RegressionA.slope

# constantGuy = RegressionB.intercept - RegressionA.intercept
# pooledError = (sumOfError(RegressionA,x,yA)+sumOfError(RegressionB,x,yB))/((2*len(x))-4)
# xMeanOverError = (np.mean(x)**2)/sumOfDeviation(x)
# constantError = math.sqrt(pooledError*((2/len(x))+2*(xMeanOverError)))\

# print("Coefficient: ", coefficientSlope)
# print("SE Coefficient:", math.sqrt(se_coefficient_a**2+se_coefficient_b**2))
# print("T-Value:", coefficientSlope/math.sqrt(se_coefficient_a**2+se_coefficient_b**2))
# print ("P-value: ", stats.t.sf(abs(coefficientSlope/math.sqrt(se_coefficient_a**2+se_coefficient_b**2)), 78) * 2)
# print('Constant T:',constantGuy/constantError)




# coeff = -2.36
# SEcoeff = 1.39
# tvalue = coeff/se_coefficient
# p_value = stats.t.sf(abs(4.03), 78) * 2
# print(p_value)

