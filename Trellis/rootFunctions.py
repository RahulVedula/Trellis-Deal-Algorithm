import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as mp
import matplotlib as mpl
import numpy as np
from scipy.stats import linregress
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

def line_function_extend(df):
    funcVals = linAverage(df)
    accumulator = []
    for i in range(df.count()+16):
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
        
def normalizeAfterDealUnits (beforeProfit, afterProfit):
    count = 0
    regressionBefore = linAverage(beforeProfit)
    normalizedPostDeal = afterProfit.copy()
    for i in afterProfit.index:
        normalizedPostDeal.at[i] = normalizedPostDeal.at[i] - (regressionBefore.slope*count+regressionBefore.intercept)
        count+=1
    return normalizedPostDeal
        


def calcProfit(revenue, costs, costPerUnit):
    return (revenue.sum() - (costs.sum()*costPerUnit))/revenue.count()

def find_most_frequent_value(sales_df, units_df):
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
    return None


def discountLoss(unitsBeforeDeal,salesBeforeDeal,salesDuringDeal,unitsDuringDeal,salesAfterDeal,unitAfterDeal):
    if find_most_frequent_value(salesBeforeDeal,unitsBeforeDeal) == None and find_most_frequent_value(salesAfterDeal,unitAfterDeal) == None:
        sales_per_unit_before = 0
    elif (find_most_frequent_value(salesBeforeDeal,unitsBeforeDeal) == None):
        sales_per_unit_before = (find_most_frequent_value(salesAfterDeal,unitAfterDeal))
    else:
        sales_per_unit_before = (find_most_frequent_value(salesBeforeDeal,unitsBeforeDeal))
    return sales_per_unit_before*unitsDuringDeal.sum() - salesDuringDeal.sum()


def netLoss(unitsBeforeDeal,salesBeforeDeal,unitsDuringDeal,salesDuringDeal,cost_per_unit):
    profitBefore = profit_dataframe(salesBeforeDeal,unitsBeforeDeal,cost_per_unit).sum()/salesBeforeDeal.count()
    profitDuring = profit_dataframe(salesDuringDeal,unitsDuringDeal,cost_per_unit)['profit'].sum()
    return profitDuring-(profitBefore['profit']*salesDuringDeal.count())


def timeUntilDealRecovery(normalizedProfit, netLoss):
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
        counter = 0
        sum=0
        for j in normalizedDeal.truncate(before=i, after=i+timedelta(days=14)).index:
            sum += normalizedDeal.at[j,'profit']
        if sum/14 < 0:
            for k in j,(j+timedelta(days=30)):
                # if normalizedDeal.truncate(before=k, after=k+timedelta(days=14)).sum<0:
                #     counter+=1
                bounceSum = 0
                for l in  normalizedDeal.truncate(before=k, after=k+timedelta(days=30)).index:
                    bounceSum += normalizedDeal.at[l,'profit']
                if bounceSum/30 < 0:
                    counter += 1
                if counter>3:
                    return i+timedelta(days=14)
            counter=0
    return NULL_CONSTANT

def changeInGrowthRate(beforeDealProfit, afterDealProfit):
    normalizedDeal = normalizeAfterDeal(beforeDealProfit,afterDealProfit)
    if normalizedDeal.empty:
        return NULL_CONSTANT
    afterDealProfit = afterDealProfit.truncate(after=(afterDealProfit.index.min()+timedelta(days=14)))
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

def profitNumbers(profitBefore, start_date, profitAfter, end_date, profitDuring):
    profitAfter30Days = profitAfter.truncate(after =(datetime.strptime(end_date, '%Y-%m-%d')+timedelta(days=30)))
    profitBefore30Days = profitBefore.truncate(before =(datetime.strptime(start_date, '%Y-%m-%d')-timedelta(days=30)), after = (datetime.strptime(start_date, '%Y-%m-%d')-timedelta(days=1)))
    return(profitBefore30Days.sum(),profitDuring.sum(),profitAfter30Days.sum())

    
def profitNumbers14Days(profitBefore, start_date, profitAfter, end_date, profitDuring):
    profitAfter14Days = profitAfter.truncate(after =(datetime.strptime(end_date, '%Y-%m-%d')+timedelta(days=14)))
    profitBefore14Days = profitBefore.truncate(before =(datetime.strptime(start_date, '%Y-%m-%d')-timedelta(days=14)), after = (datetime.strptime(start_date, '%Y-%m-%d')-timedelta(days=1)))
    return(profitBefore14Days.sum(),profitDuring.sum(),profitAfter14Days.sum())

def unitsAt14Days(unitBeforeDeal,start_date,end_date,unitAfterDeal,unitDuringDeal):
    after14DaysUnits = unitAfterDeal.truncate(after =(datetime.strptime(end_date, '%Y-%m-%d')+timedelta(days=14)))
    before14DaysUnits = unitBeforeDeal.truncate(before =(datetime.strptime(start_date, '%Y-%m-%d')-timedelta(days=14)), after = (datetime.strptime(start_date, '%Y-%m-%d')-timedelta(days=1)))
    return [before14DaysUnits.sum(),unitDuringDeal.sum(), after14DaysUnits.sum()]

def getSimpleInfo2(salesBefore,unitsBefore,salesDuring,unitsDuring,cost_per_unit,start_date,end_date,salesAfter,unitsAfter):
    profitBefore = profit_dataframe(salesBefore, unitsBefore,cost_per_unit).truncate(before =(datetime.strptime(start_date, '%Y-%m-%d')-timedelta(days=30)), after = (datetime.strptime(start_date, '%Y-%m-%d')-timedelta(days=1)))
    profitDuring = profit_dataframe(salesDuring,unitsDuring,cost_per_unit)
    profitAfter = profit_dataframe(salesAfter,unitsAfter,cost_per_unit).truncate(after =(datetime.strptime(end_date, '%Y-%m-%d')+timedelta(days=30)))
    unitsAfterDeal = unitsAfter.truncate(after =(datetime.strptime(end_date, '%Y-%m-%d')+timedelta(days=30)))
    changeInProfitsDeal = ((profitDuring['profit'].sum()/profitDuring['profit'].count())-(profitBefore['profit'].sum()/30))/(profitBefore['profit'].sum()/30)*100
    changeInUnitsDeal = (unitsDuring.sum() - ((unitsBefore.sum()/30)*unitsDuring.count()))/((unitsBefore.sum()/30)*unitsDuring.count()) * 100
    changeInProfitAfter = (((profitAfter['profit'].sum()/30)-(profitBefore['profit'].sum()/30))/(profitBefore['profit'].sum()/30))*100
    changeInUnitsAfter = (((unitsAfterDeal.sum()/30)-(unitsBefore.sum()/30))/(unitsBefore.sum()/30)) * 100
    # changeInProfitAfter = changeInGrowthRate(profit_dataframe(salesBefore,unitsBefore, cost_per_unit), profit_dataframe(salesAfter,unitsAfter, cost_per_unit)) *100
    # changeInUnitsAfter = (linAverage(unitsAfter).slope - linAverage(unitsBefore).slope)/linAverage(unitsBefore).slope * 100
    
    data = [
        ["Profits", str(round(changeInProfitsDeal,2)) + "%", "Not Statistically Different" if changeInProfitAfter == -100 else (str(round(changeInProfitAfter,2))+"%")],
        ["Units", str(round(changeInUnitsDeal,2)) + "%", str(round(changeInUnitsAfter,2)) + "%"]
    ]
    df = pd.DataFrame(data, columns=[' ', 'During Deal Effects', 'Long Term Deal Effects(30 days)'])
    return df

def round_nested_list_to_two_decimal_places(nested_list):
    return [[round(element, 2) if isinstance(element, (float, int)) else element for element in row] for row in nested_list]

def get7DayProfitInfo(beforeDeal,unitBeforeDeal,duringDeal,unitDuringDeal,afterDeal,unitAfterDeal,cost_per_unit,start_date,end_date):
    before7DaysSales = beforeDeal.truncate(before =(datetime.strptime(start_date, '%Y-%m-%d')-timedelta(days=7)), after = (datetime.strptime(start_date, '%Y-%m-%d')-timedelta(days=1)))
    after7DaysSales = afterDeal.truncate(after =(datetime.strptime(end_date, '%Y-%m-%d')+timedelta(days=7)))
    before7DaysUnits = unitBeforeDeal.truncate(before =(datetime.strptime(start_date, '%Y-%m-%d')-timedelta(days=7)), after = (datetime.strptime(start_date, '%Y-%m-%d')-timedelta(days=1)))
    after7DaysUnits= unitAfterDeal.truncate(after =(datetime.strptime(end_date, '%Y-%m-%d')+timedelta(days=7)))
    data = [["Price",find_most_frequent_value(beforeDeal,unitBeforeDeal),find_most_frequent_value(duringDeal,unitDuringDeal),find_most_frequent_value(afterDeal,unitAfterDeal)], 
        ["Fixed Cost (33%)",cost_per_unit,cost_per_unit,cost_per_unit], 
        ["Units",before7DaysUnits.sum()/7,unitDuringDeal.sum()/unitDuringDeal.count(),after7DaysUnits.sum()/7],
        ["Revenue",before7DaysSales.sum()/7,duringDeal.sum()/duringDeal.count(),after7DaysSales.sum()/7],
        ["Cost",(cost_per_unit*before7DaysUnits.sum())/7,(cost_per_unit * unitDuringDeal.sum())/unitDuringDeal.count(),(cost_per_unit*after7DaysUnits.sum())/7],
        ["Profit",(before7DaysSales.sum()-cost_per_unit*before7DaysUnits.sum())/7,(duringDeal.sum() - cost_per_unit * unitDuringDeal.sum())/duringDeal.count(), (after7DaysSales.sum() - cost_per_unit*after7DaysUnits.sum())/7]]
    rounded_data = round_nested_list_to_two_decimal_places(data)
    sales7Days_df = pd.DataFrame(rounded_data, columns=['  ','7 Days Before', 'During Deal','7 Days After'])
    return sales7Days_df

def moreInformation (profitBefore,profitAfter,unitsBeforeDeal,salesBeforeDeal,salesDuringDeal,unitsDuringDeal,afterDeal,unitAfterDeal):
    recovery_date = timeUntilDealRecovery(normalizeAfterDeal(profitBefore,profitAfter)['profit'], discountLoss(unitsBeforeDeal,salesBeforeDeal,salesDuringDeal,unitsDuringDeal,afterDeal,unitAfterDeal))
    profit_date = profitableTime(profitAfter,profitBefore) #profitableTime(profitAfter,beforeDeal,unitBeforeDeal,cost_per_unit)
    data = [["Last Day Of Deal Induced Effects", "Never" if profit_date == NULL_CONSTANT else str(profit_date - profitAfter.index.min())[:str(profit_date - profitAfter.index.min()).find('s')+1]], 
     ["Time Untill Net Losses Are Recovered", "Never" if recovery_date == NULL_CONSTANT else str(recovery_date)+" days"], 
     ["Change in Growth Rate", "Never" if changeInGrowthRate(profitBefore, profitAfter) == NULL_CONSTANT else str(round(changeInGrowthRate(profitBefore, profitAfter)*100, 2))+'%'], 
     ["Last Day of Growth From Deal", "Never" if growthTime(profitBefore, profitAfter) == NULL_CONSTANT else str(growthTime(profitBefore, profitAfter)) + " days"]]
    moreInfoDf = pd.DataFrame(data, columns=['Statistics','Product'])
    return moreInfoDf
 

def case_1(beforeDeal,afterDeal,duringDeal,unitDuringDeal,unitBeforeDeal,unitAfterDeal,cost_per_unit,start_date,end_date,profit_date,profitAfter,recovery_date,profitBefore):
    before7DaysSales = beforeDeal.truncate(before =(datetime.strptime(start_date, '%Y-%m-%d')-timedelta(days=7)), after = (datetime.strptime(start_date, '%Y-%m-%d')-timedelta(days=1)))
    after7DaysSales = afterDeal.truncate(after =(datetime.strptime(end_date, '%Y-%m-%d')+timedelta(days=7)))
    before7DaysUnits = unitBeforeDeal.truncate(before =(datetime.strptime(start_date, '%Y-%m-%d')-timedelta(days=7)), after = (datetime.strptime(start_date, '%Y-%m-%d')-timedelta(days=1)))
    after7DaysUnits= unitAfterDeal.truncate(after =(datetime.strptime(end_date, '%Y-%m-%d')+timedelta(days=7)))
    print("Profit Information: ")
    data = [["Price",find_most_frequent_value(beforeDeal,unitBeforeDeal),find_most_frequent_value(duringDeal,unitDuringDeal),find_most_frequent_value(afterDeal,unitAfterDeal)], 
        ["Fixed Cost Per Unit(33%)",cost_per_unit,cost_per_unit,cost_per_unit], 
        ["Units",before7DaysUnits.sum(),unitDuringDeal.sum(),after7DaysUnits.sum()],
        ["Revenue",before7DaysSales.sum(),duringDeal.sum(),after7DaysSales.sum()],
        ["Cost",cost_per_unit*before7DaysUnits.sum(),cost_per_unit * unitDuringDeal.sum(),cost_per_unit*after7DaysUnits.sum()],
        ["Profit",before7DaysSales.sum()-cost_per_unit*before7DaysUnits.sum(),duringDeal.sum() - cost_per_unit * unitDuringDeal.sum(), after7DaysSales.sum() - cost_per_unit*after7DaysUnits.sum()]]
    sales7Days_df = pd.DataFrame(data, columns=['  ','7 Days Before', 'During Deal','7 Days After'])
    print(sales7Days_df.to_string(index=False))
    print("_______________________________________________________________")
    string = tt.to_string(
    [["Last Day Of Deal Induced Effects", "Never" if profit_date == NULL_CONSTANT else str(profit_date - profitAfter.index.min())[:str(profit_date - profitAfter.index.min()).find('s')+1]], 
    ["Time Untill Net Losses Are Recovered", "Never" if recovery_date == NULL_CONSTANT else str(recovery_date)+" days"]],
    header=["Statistic", "Product"],  
    style=tt.styles.ascii_thin_double,
    )
    print (string)
    
def case_2(beforeDeal,afterDeal,duringDeal,unitDuringDeal,unitBeforeDeal,unitAfterDeal,cost_per_unit,start_date,end_date,profit_date,profitAfter,recovery_date,profitBefore):
    string = tt.to_string(
    [["Change in Growth Rate", "Never" if changeInGrowthRate(profitBefore, profitAfter) == NULL_CONSTANT else str(round(changeInGrowthRate(profitBefore, profitAfter)*100, 2))+'%'], 
    ["Last Day of Growth From Deal", "Never" if growthTime(profitBefore, profitAfter) == NULL_CONSTANT else str(growthTime(profitBefore, profitAfter)) + " days"]],
    header=["Statistic", "Product"],  
    style=tt.styles.ascii_thin_double,
    )
    print (string)

def case_3():
    print("")
def switch_case(case_number,beforeDeal,afterDeal,duringDeal,unitDuringDeal,unitBeforeDeal,unitAfterDeal,cost_per_unit,start_date,end_date,profit_date,profitAfter,recovery_date,profitBefore):
    switcher = {
        1: case_1,
        2: case_2,
        3: case_3,
    }
        # Get the function based on the case_number, and execute it
    selected_case = switcher.get(case_number)
    if selected_case:
        selected_case(beforeDeal,afterDeal,duringDeal,unitDuringDeal,unitBeforeDeal,unitAfterDeal,cost_per_unit,start_date,end_date,profit_date,profitAfter,recovery_date,profitBefore)
    else:
        print("Invalid case number")
