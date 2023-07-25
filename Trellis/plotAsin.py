import pandas as pd 
from datetime import datetime, timedelta
import matplotlib.pyplot as mp
import matplotlib as mpl
import numpy as np
from scipy.stats import linregress
from matplotlib.widgets import Button
import termtables as tt
from scipy.stats import linregress
from scipy import stats
import math
import hypothesisTesting
 
TIME_LIMIT = 120
NULL_CONSTANT = -1

def filterByAsin(asinValue):
    df = pd.read_csv("LUXE_SALES.csv") 
    start_date = "2022-07-01"
    end_date = "2023-07-14"
    df = df[df['asin'] == asinValue].dropna()
    df['date_trunc'] = df['date_trunc'].str[:10]
    df['sales'] = df['sales'].astype(float).astype(int)
    df['units'] = df['units'].astype(int)
    #Same data frame^
    #Ordered Sales DataFrame
    sales_df = df.drop(['units'],axis=1).groupby('date_trunc')['sales'].sum()
    sales_df.index = pd.to_datetime(sales_df.index)
    idx = pd.date_range(start_date,end_date)
    sales_df  = sales_df.reindex(idx,fill_value=0)

    #Units Ordered 
    units_df = df.drop(['sales'],axis=1).groupby('date_trunc')['units'].sum()
    units_df.index = pd.to_datetime(units_df.index)
    idx = pd.date_range(start_date,end_date)
    units_df  = units_df.reindex(idx,fill_value=0)  
    return [units_df, sales_df]


def meanAverage(df):
    return df.sum()/(pd.to_datetime(df.index.max()) - pd.to_datetime(df.index.min())).days


def averageDates(asin,start_date,end_date):
    df = filterByAsin(asin)[1]
    beforeDeal = df.truncate(after=start_date)
    beforeDeal.drop(beforeDeal.tail(1).index,inplace=True) 
    duringDeal = df.truncate(before=start_date,after = end_date)
    afterDeal = df.truncate(before = end_date)
    afterDeal.drop(afterDeal.head(1).index,inplace=True) 
    return [meanAverage(beforeDeal), meanAverage(duringDeal), meanAverage(afterDeal)]


def linAverage(df):
    y = df.to_numpy()
    accumulator = []
    for i in range(df.count()):
        accumulator.append(i)
    x = np.array(accumulator)
    return linregress(x, y)


def line_function(df):
    funcVals = linAverage(df)
    accumulator = []
    for i in range(df.count()):
        accumulator.append(funcVals.slope * i + funcVals.intercept)
    return accumulator


def profit_dataframe(revenue,cost,cost_per_unit):
    data = {'profit':[]} 
    df = pd.DataFrame(data)

    for i in range(revenue.count()):
        df.loc[i] = [revenue._get_value(i, 'sales') - (cost._get_value(i, 'units') * cost_per_unit)]
    date_range = pd.date_range(start=revenue.index.min(), end=revenue.index.max())
    df.index = date_range[:len(df)]
    return df


def normalizeAfterDeal (beforeProfit, afterProfit):
    if (slopeP(beforeProfit['profit'], afterProfit['profit']) > 0.05 and constantP(beforeProfit['profit'], afterProfit['profit']) > 0.05):
        return pd.DataFrame({'profit' : []})
    count = 0
    regressionBefore = linAverage(beforeProfit['profit'])
    normalizedPostDeal = afterProfit.copy()
    for i in afterProfit.index:
        normalizedPostDeal.at[i,'profit'] = normalizedPostDeal.at[i,'profit'] - (regressionBefore.slope*count+regressionBefore.intercept)
        count+=1
    return normalizedPostDeal
        

def calcProfit(revenue, costs, costPerUnit):
    return (revenue.sum() - (costs.sum()*costPerUnit))/revenue.count()


# def netLoss (beforeDeal, duringDeal, unitBeforeDeal, unitDuringDeal, cost_per_unit, discount):
#     dailyLoss = calcProfit(beforeDeal,unitBeforeDeal,cost_per_unit) - calcProfit(duringDeal,duringDeal,cost_per_unit)
#     return duringDeal.sum() * (1-discount)
#This calculation is Fs wrong 
def netLoss(unitsBeforeDeal,salesBeforeDeal,salesDuringDeal,unitsDuringDeal):
    sales_per_unit_before = salesBeforeDeal.sum()/unitsBeforeDeal.sum()
    # print(sales_per_unit_before)
    # print(sales_per_unit_before*unitsDuringDeal.sum() - salesDuringDeal.sum())
    return sales_per_unit_before*unitsDuringDeal.sum() - salesDuringDeal.sum()

# def netLoss(duringDeal,discount):
#     return duringDeal.sum() * (discount/100)

def timeUntilProfit(normalizedProfit, netLoss):
    if normalizedProfit.empty:
        return NULL_CONSTANT
    normalizedPostDeal = linAverage(normalizedProfit)
    numberOfDays = 0
    sumOfProfit = 0
    while sumOfProfit<netLoss:
        sumOfProfit+=normalizedPostDeal.slope*numberOfDays + normalizedPostDeal.intercept
        numberOfDays+=1  
        if (numberOfDays == TIME_LIMIT):
            return NULL_CONSTANT
    return numberOfDays


def profitableTime(afterDealProfit, beforeDealProfit):
    normalizedDeal = normalizeAfterDeal(beforeDealProfit,afterDealProfit)
    if normalizedDeal.empty:
        return NULL_CONSTANT
    for i in normalizedDeal.index:
        sum=0
        for j in normalizedDeal.truncate(before=i, after=i+timedelta(days=14)).index:
            sum += normalizedDeal.at[j,'profit']
        if sum/14 < 0:
            return i+timedelta(days=14)
    return NULL_CONSTANT

# def profitableTime(afterDealProfit,beforeDea,unitBeforeDeal,cost_per_unit):
#     beforeDealDailyProfit = calcProfit(beforeDeal,unitBeforeDeal,cost_per_unit)
#     for i in afterDealProfit.index:
#         sum=0
#         for j in afterDealProfit.truncate(before=i, after=i+timedelta(days=7)).index:
#             sum += afterDealProfit.at[j,'profit']
#         if sum/7 < beforeDealDailyProfit:
#             return i+timedelta(days=7)


def changeInGrowthRate(beforeDealProfit, afterDealProfit):
    normalizedDeal = normalizeAfterDeal(beforeDealProfit,afterDealProfit)
    if normalizedDeal.empty:
        return NULL_CONSTANT
    afterDealProfit.truncate(after=(afterDealProfit.index.min()+timedelta(days=14)))
    if linAverage(beforeDealProfit['profit']).slope < 0:
        return -1*((linAverage(afterDealProfit['profit']).slope - linAverage(beforeDealProfit['profit']).slope) / linAverage(beforeDealProfit['profit']).slope)
    return (linAverage(afterDealProfit['profit']).slope - linAverage(beforeDealProfit['profit']).slope) / linAverage(beforeDealProfit['profit']).slope


# def growthTime(beforeDealProfit, afterDealProfit):
#     normalizedDeal = normalizeAfterDeal(beforeDealProfit,afterDealProfit)
#     if normalizedDeal.empty:
#         return NULL_CONSTANT
#     beforeRegression = linAverage(beforeDealProfit['profit'])
#     if beforeRegression.slope < 0:
#         growthRateOriginal = -1*((linAverage(afterDealProfit.truncate(after=(afterDealProfit.index.min()+timedelta(days=14)))['profit']).slope-beforeRegression.slope)/beforeRegression.slope)
#         for i in range (afterDealProfit['profit'].count()):
#             result = (linAverage(afterDealProfit.truncate(after = afterDealProfit.index.min()+timedelta(days=i)+timedelta(days=15))['profit']))
#             growthRateInstant = -1*((result.slope-beforeRegression.slope)/beforeRegression.slope)
#             if growthRateInstant < 0:
#                 return i + 15
#             if i+15 ==  120 or i+15 == afterDealProfit['profit'].count()-1:
#                 return NULL_CONSTANT 
#     else:
#         growthRateOriginal = linAverage(afterDealProfit.truncate(after=(afterDealProfit.index.min()+timedelta(days=14)))['profit']).slope-beforeRegression.slope/beforeRegression.slope
#         for i in range (afterDealProfit['profit'].count()):
#             result = linAverage(afterDealProfit.truncate(after = afterDealProfit.index.min()+timedelta(days=i)+timedelta(days=15))['profit'])
#             growthRateInstant = (result.slope-beforeRegression.slope)/beforeRegression.slope
#             if growthRateInstant < 0:
#                 return i+15
#             if i+15 == 120 or i+15 == afterDealProfit['profit'].count()-1:
#                 return NULL_CONSTANT 
def growthTime(beforeDealProfit, afterDealProfit):
    normalizedDeal = normalizeAfterDeal(beforeDealProfit,afterDealProfit)
    if normalizedDeal.empty:
        return NULL_CONSTANT
    beforeRegression = linAverage(beforeDealProfit['profit'])
    for i in range (afterDealProfit['profit'].count()):
        result = (linAverage(afterDealProfit.truncate(after = afterDealProfit.index.min()+timedelta(days=i)+timedelta(days=15))['profit']))
        if result.slope < beforeRegression.slope:
            return i + 15
        if i+15 ==  120 or i+15 == afterDealProfit['profit'].count()-1:
            return NULL_CONSTANT
    
def graphLabels(recovery_date,afterDeal,profit_date):
    #Net Loss Recovered
    if recovery_date != NULL_CONSTANT:
        mp.axvline(afterDeal.index.min()+timedelta(days=recovery_date),color='purple', linestyle='--')
    #Profitable Time
    mp.axvline(profit_date, color='aqua', linestyle='--')
    #Y-axis,x-axis labels
    mp.xlabel('Date')
    mp.ylabel('Sales')
    #Legend
    legend_handles = [
    mp.Rectangle((0, 0), 1, 1, color='aqua', label='Deal Induced Effects'),
    mp.Rectangle((0, 0), 1, 1, color='purple', label='Net Losses Recovered')
    ]
    mp.legend(handles=legend_handles, loc='upper left')



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
    pooledError = (hypothesisTesting.sumOfError(RegressionA,xbeforeDealProfit,ybeforeDealProfit)+hypothesisTesting.sumOfError(RegressionB,xafterDealProfit,yafterDealProfit))/((beforeDealProfit.count()+afterDealProfit.count())-4)
    SEValue = math.sqrt(pooledError * ((1/afterDealProfit.count()) + (1/beforeDealProfit.count()) + ((np.mean(xbeforeDealProfit)**2)/hypothesisTesting.sumOfDeviation(xbeforeDealProfit))+((np.mean(xafterDealProfit)**2)/hypothesisTesting.sumOfDeviation(xafterDealProfit))))
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
    se_coefficient_a = hypothesisTesting.SeCoefficient(RegressionA,xbeforeDealProfit,ybeforeDealProfit)
    se_coefficient_b = hypothesisTesting.SeCoefficient(RegressionB,xafterDealProfit,yafterDealProfit)
    coefficientSlope = RegressionB.slope - RegressionA.slope
    TValue = coefficientSlope/math.sqrt(se_coefficient_a**2+se_coefficient_b**2)
    return stats.t.sf(abs(TValue), beforeDealProfit.count() + afterDealProfit.count()-2) * 2

def main():
    #assigning variables
    asinValue = input()#B00LCJAW06
    start_date = input() #2023-03-10
    end_date = input() #2023-05-01
    cost_per_unit = int(input()) #25   

    grouped_df = filterByAsin(asinValue)[1]
    unit_df = filterByAsin(asinValue)[0]

    #Unit Dataframes Seperation
    unitBeforeDeal = unit_df.truncate(after=start_date,before=(datetime.strptime(start_date, '%Y-%m-%d')-timedelta(days=30)))
    unitBeforeDeal.drop(unitBeforeDeal.tail(0).index,inplace=True) 
    unitDuringDeal = unit_df.truncate(before=start_date,after = end_date)
    unitAfterDeal = unit_df.truncate(before = end_date)
    unitAfterDeal.drop(unitAfterDeal.head(0).index,inplace=True) 


    #sales Dataframes Seperation
    beforeDeal = grouped_df.truncate(after=start_date,before=(datetime.strptime(start_date, '%Y-%m-%d')-timedelta(days=30)))
    beforeDeal.drop(beforeDeal.tail(0).index,inplace=True) 
    duringDeal = grouped_df.truncate(before=start_date,after = end_date)
    afterDeal = grouped_df.truncate(before = end_date)
    afterDeal.drop(afterDeal.head(0).index,inplace=True) 
    print(beforeDeal.truncate(before =(datetime.strptime(start_date, '%Y-%m-%d')-timedelta(days=7))))
    print(afterDeal.truncate(after =(datetime.strptime(end_date, '%Y-%m-%d')+timedelta(days=7))))

    #getting mean averages
    list_of_averages = averageDates(asinValue,start_date,end_date)
    
    #Calculate profits
    profitBefore= profit_dataframe(beforeDeal,unitBeforeDeal, cost_per_unit)
    profitDuring= profit_dataframe(duringDeal,unitDuringDeal, cost_per_unit)
    profitAfter= profit_dataframe(afterDeal,unitAfterDeal, cost_per_unit)
    #recovery_date = timeUntilProfit(normalizeAfterDeal(profitBefore,profitAfter)['profit'],netLoss(beforeDeal,duringDeal,unitBeforeDeal,unitDuringDeal,cost_per_unit))
    recovery_date = timeUntilProfit(normalizeAfterDeal(profitBefore,profitAfter)['profit'],netLoss(unitBeforeDeal,beforeDeal,duringDeal,unitDuringDeal))
    profit_date = profitableTime(profitAfter,profitBefore) #profitableTime(profitAfter,beforeDeal,unitBeforeDeal,cost_per_unit)
    string = tt.to_string(
    [["Last Day Of Deal Induced Effects", "Never" if profit_date == NULL_CONSTANT else str(profit_date - profitAfter.index.min())[:str(profit_date - profitAfter.index.min()).find('s')+1]], 
     ["Time Untill Net Losses Are Recovered", "Never" if recovery_date == NULL_CONSTANT else str(recovery_date)+" days"], 
     ["Change in Growth Rate", "Never" if changeInGrowthRate(profitBefore, profitAfter) == NULL_CONSTANT else str(round(changeInGrowthRate(profitBefore, profitAfter)*100, 2))+'%'], 
     ["Last Day of Growth From Deal", "Never" if growthTime(profitBefore, profitAfter) == NULL_CONSTANT else str(growthTime(profitBefore, profitAfter)) + " days"]],
    header=["Statistic", "Product"],  
    style=tt.styles.ascii_thin_double,
    )
    print (string)

    #Graphing
    fig1 = mp.figure(figsize=(12, 8))
    fig2 = mp.figure(figsize=(12, 8))
    fig3 = mp.figure(figsize=(12, 8))
    #Plot graph with linear regression of revenue
    mp.figure(fig1.number) #mp.subplot(3,1,1)
    grouped_df.truncate(before=beforeDeal.index.min()).plot(x="date_trunch", y=["sales"],grid = True, kind="line")
    beforeLine = line_function(beforeDeal)
    duringLine = line_function(duringDeal)
    afterLine = line_function(afterDeal)
    mp.plot(afterDeal.index,afterLine, color='orange')
    mp.plot(beforeDeal.index,beforeLine,color='orange')
    mp.plot(duringDeal.index,duringLine, color='orange')
    mp.axvspan(duringDeal.index.min(),duringDeal.index.max(), facecolor='0.2', alpha=0.5)
    graphLabels(recovery_date,afterDeal,profit_date)
    mp.title("Sales per Day")

    #Plot graph with averages 
    mp.figure(fig2.number)
    grouped_df.truncate(before=beforeDeal.index.min()).plot(x="date_trunch", y=["sales"],grid = True, kind="line")
    mp.hlines(y=list_of_averages[0], xmin=beforeDeal.index.min(), xmax=beforeDeal.index.max(), colors='#301934', linestyles='-', lw=2, label='Average 1')
    mp.hlines(y=list_of_averages[1], xmin=duringDeal.index.min(), xmax=duringDeal.index.max(), colors='#301934', linestyles='-', lw=2, label='Average 2')
    mp.hlines(y=list_of_averages[2], xmin=afterDeal.index.min(), xmax=afterDeal.index.max(), colors='#301934', linestyles='-', lw=2, label='Average 3')    
    mp.axvspan(duringDeal.index.min(),duringDeal.index.max(), facecolor='0.2', alpha=0.5)
    graphLabels(recovery_date,afterDeal,profit_date)
    mp.title("Average Sales per Day")

    #Plot trend lines for profit
    mp.figure(fig3.number) 
    grouped_df.truncate(before=beforeDeal.index.min()).plot(x="date_trunc", y=["sales"],grid = True, kind="line")
    profitBeforeLine = line_function(profitBefore['profit'])
    profitDuringLine = line_function(profitDuring['profit'])
    profitAfterLine = line_function(profitAfter['profit'])
    mp.plot(beforeDeal.index,profitBeforeLine, color='green')
    mp.plot(duringDeal.index,profitDuringLine, color='green')
    mp.plot(afterDeal.index,profitAfterLine, color='green')
    mp.axvspan(duringDeal.index.min(),duringDeal.index.max(), facecolor='0.2', alpha=0.5)
    graphLabels(recovery_date,afterDeal,profit_date)
    mp.title("Profit per Day")
    
    mp.show()

if __name__ == "__main__":
    main()

""" 
B0921FSF2F
2022-09-19
2022-09-26
17
0
"""

"""
B0921FSF2F

"""

""""
significant
B09VCVNXZN
2022-08-22
2023-08-29
25
15
"""


# Change Net Loss, find deals.

