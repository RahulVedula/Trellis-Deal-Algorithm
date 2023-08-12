import pandas as pd 
from datetime import datetime, timedelta
import matplotlib.pyplot as mp
import numpy as np
from scipy.stats import linregress
import termtables as tt
from scipy.stats import linregress
from scipy import stats
import math
import hypothesisTesting

""" 
This code will parse through all deals ran on child asins for the past year, and determine how many pass a metric (Uplift, profit per day higher than 50$...)

Input Format:
ASIN
2022-09-26 (Deal Date)
2022-10-03 (Deal End Date)
30 (Cost Per Unit)
"""

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
    afterDeal = df.truncate(before = end_date, after = datetime.strptime(end_date, '%Y-%m-%d')+timedelta(days=14))
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

def find_most_frequent_value(sales_df, units_df):
    # Create deep copies of the DataFrames
    sales_copy = sales_df.copy(deep=True)
    units_copy = units_df.copy(deep=True)
    sales_copy = sales_copy.reset_index()
    units_copy = units_copy.reset_index()
    if not sales_copy.empty and not units_copy.empty and (units_copy.iloc[:, 1] != 0).any():
        non_zero_units = units_copy.loc[units_copy.iloc[:, 1] != 0]
        merged_df = pd.merge(sales_copy, non_zero_units, on='index', suffixes=['_sales', '_units'])
        divided_values = merged_df.iloc[:, 1] / merged_df.iloc[:, 2]
        if not divided_values.empty:
            most_frequent_value = divided_values.mode().values[0]

            return most_frequent_value

    # Return None if there are no matching indices or valid divided values
    return None
# def netLoss (beforeDeal, duringDeal, unitBeforeDeal, unitDuringDeal, cost_per_unit, discount):
#     dailyLoss = calcProfit(beforeDeal,unitBeforeDeal,cost_per_unit) - calcProfit(duringDeal,duringDeal,cost_per_unit)
#     return duringDeal.sum() * (1-discount)
#This calculation is Fs wrong 
def discountLoss(unitsBeforeDeal,salesBeforeDeal,salesDuringDeal,unitsDuringDeal,afterDeal,unitAfterDeal):
    if find_most_frequent_value(salesBeforeDeal,unitsBeforeDeal) == None and find_most_frequent_value(afterDeal,unitAfterDeal) == None:
        sales_per_unit_before = 0
    if (find_most_frequent_value(salesBeforeDeal,unitsBeforeDeal) == None):
        sales_per_unit_before = (find_most_frequent_value(afterDeal,unitAfterDeal))
    else:
        sales_per_unit_before = (find_most_frequent_value(salesBeforeDeal,unitsBeforeDeal))
    # print(sales_per_unit_before)
    # print(sales_per_unit_before*unitsDuringDeal.sum() - salesDuringDeal.sum())
    return sales_per_unit_before*unitsDuringDeal.sum() - salesDuringDeal.sum()

def netLoss():
    return 1


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

def changeInGrowthRate(beforeDealProfit, afterDealProfit):
    normalizedDeal = normalizeAfterDeal(beforeDealProfit,afterDealProfit)
    if normalizedDeal.empty:
        return NULL_CONSTANT
    afterDealProfit= afterDealProfit.truncate(after=(afterDealProfit.index.min()+timedelta(days=14)))
    if (linAverage(beforeDealProfit['profit']).slope > 0 and linAverage(beforeDealProfit['profit']).slope < 1)  or (linAverage(beforeDealProfit['profit']).slope < 0 and linAverage(beforeDealProfit['profit']).slope > -1):
        return (linAverage(afterDealProfit['profit']).slope)
    elif(linAverage(beforeDealProfit['profit']).slope == 0):
        return linAverage(afterDealProfit['profit']).slope
    elif linAverage(beforeDealProfit['profit']).slope < 0:
        return -1*((linAverage(afterDealProfit['profit']).slope - linAverage(beforeDealProfit['profit']).slope) / linAverage(beforeDealProfit['profit']).slope)
    return (linAverage(afterDealProfit['profit']).slope - linAverage(beforeDealProfit['profit']).slope) / linAverage(beforeDealProfit['profit']).slope

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

def unitsAtInterval(unitBeforeDeal,start_date,end_date,unitAfterDeal):
    after30DaysUnits = unitAfterDeal.truncate(after =(datetime.strptime(end_date, '%Y-%m-%d')+timedelta(days=30)))
    before30DaysUnits = unitBeforeDeal.truncate(before =(datetime.strptime(start_date, '%Y-%m-%d')-timedelta(days=30)), after = (datetime.strptime(start_date, '%Y-%m-%d')-timedelta(days=1)))
    return [before30DaysUnits.sum(),after30DaysUnits.sum()]

def profitNumbers(profitAfter,end_date):
    profitAfter7Days = profitAfter.truncate(after =(datetime.strptime(end_date, '%Y-%m-%d')+timedelta(days=7)))
    profitAfter30Days = profitAfter.truncate(after =(datetime.strptime(end_date, '%Y-%m-%d')+timedelta(days=30)))
    return(profitAfter7Days.sum(),profitAfter30Days.sum())


def main():
    #assigning variables
    df = pd.read_csv("LUXE_DEALS.csv")
    list_top_slopes = []
    count=0
    for i in range(len(df)):
        asinValue = df.iloc[i, df.columns.get_loc('asin')]
        start_date = df.iloc[i, df.columns.get_loc('start_datetime')][:10]
        end_date = df.iloc[i, df.columns.get_loc('end_datetime')][:10]  
        discount = 0
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
        if find_most_frequent_value(beforeDeal,unitBeforeDeal) == None and find_most_frequent_value(afterDeal,unitAfterDeal) == None:
            cost_per_unit = 0
        elif (find_most_frequent_value(beforeDeal,unitBeforeDeal) == None):
            cost_per_unit = (find_most_frequent_value(afterDeal,unitAfterDeal))/3
        else:
            cost_per_unit = (find_most_frequent_value(beforeDeal,unitBeforeDeal))/3
        #getting mean averages
        list_of_averages = averageDates(asinValue,start_date,end_date)
        
        #Calculate profits
        profitBefore= profit_dataframe(beforeDeal,unitBeforeDeal, cost_per_unit)
        profitDuring= profit_dataframe(duringDeal,unitDuringDeal, cost_per_unit)
        profitAfter= profit_dataframe(afterDeal,unitAfterDeal, cost_per_unit) 
    
        profit30DaysAfter = profitAfter['profit'].truncate(after = profitAfter.index.min()+timedelta(days=30))
        if (profitBefore['profit'].sum()<profit30DaysAfter.sum()) and (unitBeforeDeal.sum() > 5):
            count+=1

    print('profit deals:', count)


if __name__ == "__main__":
    main()

