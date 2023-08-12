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


midrange_parent_asins = [['B00WUOYPFA', '2022-08-08', '2022-08-15'], ['B00WV8HBT2', '2023-05-06', '2023-05-07'], ['B06XWCZ52H', '2023-03-13', '2023-03-14'], ['B06ZZB844H', '2023-07-03', '2023-07-11'], ['B06ZZB879W', '2023-01-16', '2023-02-27'], ['B06ZZB879W', '2023-07-03', '2023-07-11'], ['B071DDFNZF', '2023-07-03', '2023-07-10'], ['B071R215FQ', '2022-11-21', '2022-12-02'], ['B071XSQR4L', '2023-01-16', '2023-03-14'], ['B071XSQR4L', '2023-07-03', '2023-07-11'], ['B075SHNG22', '2022-09-12', '2022-09-19'], ['B081482M3X', '2022-10-10', '2022-10-11'], ['B081482M3X', '2023-07-03', '2023-07-11'], ['B0BG8Y8RDZ', '2022-11-08', '2022-11-28'], ['B0BG8Y8RDZ', '2023-01-16', '2023-02-20'], ['B0BG8Y8RDZ', '2023-06-05', '2023-06-12'], ['B0CB9HCPK8', '2022-10-10', '2022-10-11'], ['B0CB9HTJNX', '2023-07-03', '2023-07-08'], ['B0CB9JY7ZY', '2023-01-09', '2023-02-16'], ['B0CB9KKV2D', '2022-12-26', '2023-01-02'], ['B0CBD5MWSN', '2022-08-08', '2022-08-15'], ['B0CBHVS17Z', '2022-08-08', '2022-08-15'], ['B0CBKZWRCZ', '2023-02-01', '2023-02-02'], ['B0CBL8GGCP', '2023-03-23', '2023-03-24'], ['B0CBL9HLFC', '2022-11-08', '2022-12-02'], ['B0CBL9HLFC', '2023-06-05', '2023-06-12'], ['B0CBLBFJ32', '2023-02-20', '2023-02-27']]

def graphingAsins(parentAsin,start_date,end_date):
    parentChildDf = pd.read_csv('parent_asins.csv')
    parentChildDf = parentChildDf[parentChildDf['parent_asin'] == parentAsin].dropna()
    profitBefore, profitDuring, profitAfter = parentInfoByChild(parentChildDf,start_date,end_date)
    profitBeforeLine = rootFunctions.line_function(profitBefore['profit'])
    profitDuringLine = rootFunctions.line_function(profitDuring['profit'])
    profitAfterLine = rootFunctions.line_function_extend(profitAfter['profit'].truncate(after=(profitAfter.index.min()+timedelta(days=14))))

    fig3 = mp.figure(figsize=(12, 8))
    mp.figure(fig3.number) 
    fig3.patch.set_facecolor(TRELLIS_CREAM)  # You can change 'lightgray' to any color you prefer for the outside of the plot
    mp.plot(profitBefore.index,profitBeforeLine, color=TRELLIS_PURPLE_DARK)
    mp.plot(profitDuring.index,profitDuringLine, color=TRELLIS_PURPLE_DARK)
    mp.plot(profitAfter.truncate(after=(profitAfter.index.min()+timedelta(days=14))).index,profitAfterLine[:15], color=TRELLIS_PURPLE_DARK)
    mp.plot(profitAfter.truncate(before=(profitAfter.index.min()+timedelta(days=14)), after = (profitAfter.index.min()+timedelta(days=30))).index,profitAfterLine[14:], color=TRELLIS_PURPLE_DARK,linestyle='--')
    mp.show()
def parentInfoByChild(parentChild,start_date,end_date):
    start = "2022-07-01"
    end = "2023-07-14"
    idx = pd.date_range(start,end)
    data = {'profit':[]}
    parentProfitBefore = pd.DataFrame(data)
    parentProfitBefore  = parentProfitBefore.reindex(idx,fill_value=0) 
    parentProfitDuring = pd.DataFrame(data)
    parentProfitDuring  = parentProfitDuring.reindex(idx,fill_value=0) 
    parentProfitAfter = pd.DataFrame(data)
    parentProfitAfter  = parentProfitAfter.reindex(idx,fill_value=0) 

    for i in range (len(parentChild)):
        asinValue = parentChild.iloc[i, parentChild.columns.get_loc('asin')]
        grouped_df = rootFunctions.filterByAsin(asinValue)[1]
        unit_df = rootFunctions.filterByAsin(asinValue)[0]
        unitBeforeDeal = unit_df.truncate(after=start_date,before=(datetime.strptime(start_date, '%Y-%m-%d')-timedelta(days=30)))
        unitBeforeDeal.drop(unitBeforeDeal.tail(0).index,inplace=True) 
        unitDuringDeal = unit_df.truncate(before=start_date,after = end_date)
        unitAfterDeal = unit_df.truncate(before = end_date)
        unitAfterDeal.drop(unitAfterDeal.head(0).index,inplace=True) 
        beforeDeal = grouped_df.truncate(after=start_date,before=(datetime.strptime(start_date, '%Y-%m-%d')-timedelta(days=30)))
        beforeDeal.drop(beforeDeal.tail(0).index,inplace=True) 
        duringDeal = grouped_df.truncate(before=start_date,after = end_date)
        afterDeal = grouped_df.truncate(before = end_date)
        afterDeal.drop(afterDeal.head(0).index,inplace=True)

        if rootFunctions.find_most_frequent_value(beforeDeal,unitBeforeDeal) == None and rootFunctions.find_most_frequent_value(afterDeal,unitAfterDeal) == None:
            cost_per_unit = 0
        elif (rootFunctions.find_most_frequent_value(beforeDeal,unitBeforeDeal) == None):
            cost_per_unit = (rootFunctions.find_most_frequent_value(afterDeal,unitAfterDeal))/3 
        else:
            cost_per_unit = (rootFunctions.find_most_frequent_value(beforeDeal,unitBeforeDeal))/3 
        profitBefore= rootFunctions.profit_dataframe(beforeDeal,unitBeforeDeal, cost_per_unit)
        profitDuring= rootFunctions.profit_dataframe(duringDeal,unitDuringDeal, cost_per_unit)
        profitAfter= rootFunctions.profit_dataframe(afterDeal,unitAfterDeal, cost_per_unit)
        parentProfitBefore = parentProfitBefore.add(profitBefore)

        # Add profitDuring to parentProfitDuring using .add() with fill_value=0
        parentProfitDuring = parentProfitDuring.add(profitDuring)

        # Add profitAfter to parentProfitAfter using .add() with fill_value=0
        parentProfitAfter = parentProfitAfter.add(profitAfter)
    parentProfitBefore = parentProfitBefore.truncate(after=start_date,before=(datetime.strptime(start_date, '%Y-%m-%d')-timedelta(days=30)))
    
    parentProfitBefore.drop(parentProfitBefore.tail(1).index,inplace=True) 
    parentProfitDuring = parentProfitDuring.truncate(before=start_date,after = end_date)
    parentProfitAfter = parentProfitAfter.truncate(before = end_date) #after=(datetime.strptime(end_date, '%Y-%m-%d')+timedelta(days=30))
    parentProfitAfter.drop(parentProfitAfter.head(1).index,inplace=True) 
    return parentProfitBefore, parentProfitDuring,parentProfitAfter 

def main():
    graphingAsins(midrange_parent_asins[0][0], midrange_parent_asins[0][1], midrange_parent_asins[0][2])


if __name__ == "__main__":
    main()