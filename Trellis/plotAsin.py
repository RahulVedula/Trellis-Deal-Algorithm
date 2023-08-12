import pandas as pd 
from datetime import datetime, timedelta
import matplotlib.pyplot as mp
import numpy as np
from scipy.stats import linregress
from matplotlib.widgets import Button
import termtables as tt
from scipy.stats import linregress
from scipy import stats
import math
import hypothesisTesting
import rootFunctions
 
TIME_LIMIT = 120
NULL_CONSTANT = -1
TRELLIS_PINK_LIGHT='#FFB6C1'
TRELLIS_PINK='#D91266'
TRELLIS_PURPLE='#850066'
GREY='#D3D3D3' 
TRELLIS_PURPLE_DARK='#2E0054'
TRELLIS_CREAM = '#FFFBF8'
TRELLIS_LILAC= '#CCCCFF'
TRELLIS_YELLOW= '#FFDF5A'

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
    afterDealProfit= afterDealProfit.truncate(after=(afterDealProfit.index.min()+timedelta(days=14)))
    if (linAverage(beforeDealProfit['profit']).slope > 0 and linAverage(beforeDealProfit['profit']).slope < 1)  or (linAverage(beforeDealProfit['profit']).slope < 0 and linAverage(beforeDealProfit['profit']).slope > -1):
        return (linAverage(afterDealProfit['profit']).slope)
    elif(linAverage(beforeDealProfit['profit']).slope == 0):
        return linAverage(afterDealProfit['profit']).slope
    elif linAverage(beforeDealProfit['profit']).slope < 0:
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
    
def graphLabels(recovery_date,duringDeal,profit_date, asinValue):
    #Net Loss Recovered
    # if recovery_date != NULL_CONSTANT:
    #     mp.axvline(duringDeal.index.max()+timedelta(days=recovery_date),color='black', linestyle='--')
    mp.axvline(duringDeal.index.min(),color='black', linestyle='--')
    mp.axvline(duringDeal.index.max(),color='black', linestyle='--')
    #Profitable Time
    #Y-axis,x-axis labels
    mp.xlabel('Date')
    mp.ylabel('Sales')
    #Legend
    legend_handles = [
    # mp.Rectangle((0, 0), 1, 1, color='black', label='Deal Discount Recovered'),
    mp.Rectangle((0, 0), 1, 1, color=TRELLIS_PINK, alpha = 0.5, label= "Deal Period"),
    mp.Rectangle((0, 0), 1, 1, color=TRELLIS_PURPLE_DARK, alpha = 0.5, label= "Regression Lines for Profit"),

    ]
    legend = mp.legend(handles=legend_handles, loc='upper left')
    
    # Set the background color of the legend
    legend.get_frame().set_facecolor(TRELLIS_CREAM)  # You can change 'lightgray' to any color you prefer



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
    #getting mean averages
    list_of_averages = averageDates(asinValue,start_date,end_date)
    if rootFunctions.find_most_frequent_value(beforeDeal,unitBeforeDeal) == None and rootFunctions.find_most_frequent_value(afterDeal,unitAfterDeal) == None:
        cost_per_unit = 0
    elif (rootFunctions.find_most_frequent_value(beforeDeal,unitBeforeDeal) == None):
        cost_per_unit = (rootFunctions.find_most_frequent_value(afterDeal,unitAfterDeal))/3 
    else:
        cost_per_unit = (rootFunctions.find_most_frequent_value(beforeDeal,unitBeforeDeal))/3 
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
    # print (string)

    #Graphing
    fig1 = mp.figure(figsize=(12, 8))
    fig2 = mp.figure(figsize=(12, 8))
    fig3 = mp.figure(figsize=(12, 8))
    #Plot graph with linear regression of revenue
    mp.figure(fig1.number) #mp.subplot(3,1,1)
    mp.ylim(0, 2000)
    grouped_df.truncate(before=beforeDeal.index.min(), after=profitAfter.index.min()+timedelta(days=30)).plot(x="date_trunc", y=["sales"], kind="line", color=TRELLIS_PURPLE, alpha=0.5)
    mp.fill_between(grouped_df.truncate(before=beforeDeal.index.min(), after=duringDeal.index.min()).index.values, grouped_df.truncate(before=beforeDeal.index.min(), after=duringDeal.index.min()), color=TRELLIS_PURPLE, alpha=0.2)
    mp.fill_between(grouped_df.truncate(before=duringDeal.index.min(), after=duringDeal.index.max()).index.values, grouped_df.truncate(before=duringDeal.index.min(), after=duringDeal.index.max()), color=TRELLIS_PURPLE_DARK, alpha=0.3)
    mp.fill_between(grouped_df.truncate(before=duringDeal.index.max(), after=profitAfter.index.min()+timedelta(days=30)).index.values, grouped_df.truncate(before=duringDeal.index.max(), after=profitAfter.index.min()+timedelta(days=30)), color=TRELLIS_PURPLE, alpha=0.2)
    beforeLine = line_function(beforeDeal)
    duringLine = line_function(duringDeal)
    afterLine = line_function_extend(afterDeal.truncate(after=(afterDeal.index.min()+timedelta(days=14))))
    mp.plot(afterDeal.truncate(after=(afterDeal.index.min()+timedelta(days=14))).index,afterLine[:15], color=TRELLIS_PINK)
    mp.plot(afterDeal.truncate(before=(afterDeal.index.min()+timedelta(days=14)),after=(afterDeal.index.min()+timedelta(days=30))).index,afterLine[14:], color=TRELLIS_PINK,linestyle='--')
    mp.plot(beforeDeal.index,beforeLine,color=TRELLIS_PINK)
    mp.plot(duringDeal.index,duringLine, color=TRELLIS_PINK)
    graphLabels(recovery_date,duringDeal,profit_date, asinValue)
    mp.title("Sales per Day")

    #Plot graph with averages 
    # mp.figure(fig2.number)
    # grouped_df.truncate(before=beforeDeal.index.min(), after=profitAfter.index.min()+timedelta(days=30)).plot(x="date_trunch", y=["sales"],grid = True, kind="line")
    # mp.hlines(y=list_of_averages[0], xmin=beforeDeal.index.min(), xmax=beforeDeal.index.max(), colors='#301934', linestyles='-', lw=2, label='Average 1')
    # mp.hlines(y=list_of_averages[1], xmin=duringDeal.index.min(), xmax=duringDeal.index.max(), colors='#301934', linestyles='-', lw=2, label='Average 2')
    # mp.hlines(y=list_of_averages[2], xmin=afterDeal.index.min(), xmax=afterDeal.index.max(), colors='#301934', linestyles='-', lw=2, label='Average 3')    
    # mp.axvspan(duringDeal.index.min(),duringDeal.index.max(), facecolor='0.2', alpha=0.5)
    # graphLabels(recovery_date,afterDeal,profit_date, asinValue)
    # mp.title("Average Sales per Day")

    #Plot trend lines for profit

    mp.ylim(0, 2000)
    grouped_df.truncate(before=beforeDeal.index.min(), after=profitAfter.index.min()+timedelta(days=30)).plot(x="date_trunc", y=["Profit"], kind="line", color=TRELLIS_PURPLE, alpha = 0.5)
    profitBeforeLine = line_function(profitBefore['profit'])
    profitDuringLine = line_function(profitDuring['profit'])
    profitAfterLine = line_function_extend(profitAfter['profit'].truncate(after=(profitAfter.index.min()+timedelta(days=14))))
    mp.fill_between(grouped_df.truncate(before=beforeDeal.index.min(), after=duringDeal.index.min()).index.values, grouped_df.truncate(before=beforeDeal.index.min(), after=duringDeal.index.min()), color=TRELLIS_PURPLE, alpha=0.2)
    mp.fill_between(grouped_df.truncate(before=duringDeal.index.min(), after=duringDeal.index.max()).index.values, grouped_df.truncate(before=duringDeal.index.min(), after=duringDeal.index.max()), color=TRELLIS_PINK, alpha=0.4)
    mp.fill_between(grouped_df.truncate(before=duringDeal.index.max(), after=profitAfter.index.min()+timedelta(days=30)).index.values, grouped_df.truncate(before=duringDeal.index.max(), after=profitAfter.index.min()+timedelta(days=30)), color=TRELLIS_PURPLE, alpha=0.2)
    mp.plot(beforeDeal.index,profitBeforeLine, color=TRELLIS_PURPLE_DARK)
    mp.plot(duringDeal.index,profitDuringLine, color=TRELLIS_PURPLE_DARK)
    mp.plot(profitAfter.truncate(after=(profitAfter.index.min()+timedelta(days=14))).index,profitAfterLine[:15], color=TRELLIS_PURPLE_DARK)
    mp.plot(profitAfter.truncate(before=(profitAfter.index.min()+timedelta(days=14)), after = (profitAfter.index.min()+timedelta(days=30))).index,profitAfterLine[14:], color=TRELLIS_PURPLE_DARK,linestyle='--')
    graphLabels(recovery_date,duringDeal,profit_date, asinValue)    
    mp.gca().set_facecolor(TRELLIS_CREAM)
    # mp.axvspan(duringDeal.index.min(),duringDeal.index.max(), facecolor=TRELLIS_YELLOW, alpha=0.3)
    mp.title("Profit per Day")
    columns = ['Asin', 'Incremental Profit/Loss','Profit Before(30 Days)', 'Profit During', 'Profit After(30 Days)','Units Before(30 Days)', 'Units During', 'Units After(30 Days)']
    dealInfo = pd.DataFrame(columns=columns)
    if (rootFunctions.find_most_frequent_value(beforeDeal,unitBeforeDeal) >= rootFunctions.find_most_frequent_value(afterDeal,unitAfterDeal) and (rootFunctions.unitsAtInterval(unitBeforeDeal,start_date,end_date,unitAfterDeal)[0])/30 < (rootFunctions.unitsAtInterval(unitBeforeDeal,start_date,end_date,unitAfterDeal)[1])/30):
            Uplift = 'Y'
    else:
            Uplift = 'N'
    new_row = [asinValue, rootFunctions.netLoss(unitBeforeDeal,beforeDeal,unitDuringDeal,duringDeal,cost_per_unit),
                     rootFunctions.profitNumbers(profitBefore['profit'],start_date,profitAfter['profit'],end_date,profitDuring['profit'])[0]/30, 
                   rootFunctions.profitNumbers(profitBefore['profit'],start_date,profitAfter['profit'],end_date,profitDuring['profit'])[1]/profitDuring['profit'].count(),rootFunctions.profitNumbers(profitBefore['profit'],start_date,profitAfter['profit'],end_date,profitDuring['profit'])[2]/30,
                   (rootFunctions.unitsAtInterval(unitBeforeDeal,start_date,end_date,unitAfterDeal)[0])/30, unitDuringDeal.sum()/unitDuringDeal.count(), 
                   (rootFunctions.unitsAtInterval(unitBeforeDeal,start_date,end_date,unitAfterDeal)[1])/30]
        #rootFunctions.profitNumbers(profitAfter['profit'],end_date)[0],rootFunctions.profitNumbers(profitAfter['profit'],end_date)[1]
    new_row[-7:] = [round(element, 2) for element in new_row[-7:]]
    dealInfo.loc[0] = new_row
    print(dealInfo.to_string(index=False))
    df1 = rootFunctions.getSimpleInfo2(beforeDeal,unitBeforeDeal,duringDeal,unitDuringDeal,cost_per_unit,start_date,end_date,afterDeal,unitAfterDeal)
    df2 = rootFunctions.get7DayProfitInfo(beforeDeal,unitBeforeDeal,duringDeal,unitDuringDeal,afterDeal,unitAfterDeal,cost_per_unit,start_date,end_date)
    df3 = rootFunctions.moreInformation(profitBefore,profitAfter,unitBeforeDeal,beforeDeal,duringDeal,unitDuringDeal,afterDeal,unitAfterDeal)
    print('\n---------------------------------------------------\n',df1.to_string(index=False), '\n----------------------------------------------------\n',df2.to_string(index=False),'\n---------------------------------------------------\n', df3.to_string(index=False))
    mp.show()

if __name__ == "__main__":
    main()

""" 
B0921FSF2F
2022-09-19
2022-09-26
17
0
with higher intercept 

yes: [8.156722689075629, 'B0921TP88K', '2023-01-16', '2023-01-23']
maybe: [3.268986662454383, 'B087NWM539', '2022-08-08', '2022-08-15'], [3.076787127563983, 'B0B1SD12R6', '2023-02-06', '2023-02-13']


B0921TP88K
2023-01-16
2023-01-23


[[1.9523809523809519, 'B0B1S7NN9G', '2022-08-29', '2022-09-05'], [1.952380952380952, 'B09VYGC6RL', '2022-08-01', '2022-08-08'], [2.207875457875458, 'B09VCSSQY1', '2022-08-29', '2022-09-05'], [2.3785714285714286, 'B088T5LXP5', '2023-01-16', '2023-01-23'], [2.615324675324675, 'B08143GCL2', '2023-02-13', '2023-02-20'], [3.076787127563983, 'B0B1SD12R6', '2023-02-06', '2023-02-13'], [3.268986662454383, 'B087NWM539', '2022-08-08', '2022-08-15'], [8.156722689075629, 'B0921TP88K', '2023-01-16', '2023-01-23'], [10.103571428571426, 'B0B1SFDGY7', '2023-01-09', '2023-01-16'], [27.04060902373137, 'B01IRCAUXO', '2022-10-10', '2022-10-11']]
"""

"""
B0921FSF2F
[[16.413095238095234, 'B08KSB18PR', '2023-02-13', '2023-02-20'], [23.142857142857142, 'B09VCVNXZN', '2022-07-18', '2022-07-25'], [23.81428571428571, 'B087NWJX6D', '2022-07-17', '2022-07-18'], [24.77142857142857, 'B081437ZDY', '2022-07-17', '2022-07-18'], [25.933333333333334, 'B08143M8QW', '2022-07-17', '2022-07-18'], [26.13095238095238, 'B08142X9BJ', '2022-07-17', '2022-07-18'], [27.04060902373137, 'B01IRCAUXO', '2022-10-10', '2022-10-11'], [27.049999999999997, 'B08143GCL2', '2022-07-17', '2022-07-18'], [27.935714285714287, 'B087NWNDHC', '2022-07-17', '2022-07-18'], [51.16428571428571, 'B09VCT6WGM', '2023-02-06', '2023-02-13']]
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
#[[4.647058823529413, 'B08KSL5V96', '2022-11-21', '2022-11-28'], [4.670337922403003, 'B087NWNKLW', '2022-09-12', '2022-09-19'], [5.448949935992805, 'B0B5HM9F13', '2023-02-13', '2023-02-20'], [5.971428571428571, 'B08MFW2PF8', '2022-08-01', '2022-08-08'], [8.156722689075629, 'B0921TP88K', '2023-01-16', '2023-01-23'], [9.714285714285715, 'B075SKYCQ6', '2022-09-13', '2022-09-14'], [10.103571428571426, 'B0B1SFDGY7', '2023-01-09', '2023-01-16'], [15.857142857142856, 'B08KS957Z7', '2022-09-12', '2022-09-19'], [27.04060902373137, 'B01IRCAUXO', '2022-10-10', '2022-10-11'], [51.16428571428571, 'B09VCT6WGM', '2023-02-06', '2023-02-13']]
#[[7.065878111262216, 'B01IRCAUXO', '2023-04-17', '2023-04-24'], [7.261904761904761, 'B00RH6BC3O', '2022-09-12', '2022-09-19'], [8.156722689075629, 'B0921TP88K', '2023-01-16', '2023-01-23'], [8.834709240424383, 'B08MFWFMNH', '2022-09-19', '2022-09-26'], [9.642857142857142, 'B08VJN6NHM', '2022-08-29', '2022-09-05'], [9.714285714285715, 'B075SKYCQ6', '2022-09-13', '2022-09-14'], [10.103571428571426, 'B0B1SFDGY7', '2023-01-09', '2023-01-16'], [15.857142857142856, 'B08KS957Z7', '2022-09-12', '2022-09-19'], [27.04060902373137, 'B01IRCAUXO', '2022-10-10', '2022-10-11'], [51.16428571428571, 'B09VCT6WGM', '2023-02-06', '2023-02-13']]

#B01IRCAUXO, 