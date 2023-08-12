import rootFunctions
import pandas as pd
from datetime import datetime, timedelta

#This code seperates combines grouped deals (within a 14 day period), 
#then analyzes each of these grouped deals and by certain metrics, determines certain statistics related to each deal
#This code currently measures the uplift and splits it into three buckets of bad, neutral, and good, then further subdivides the positive by how positive they are and the deal type


def childAsins():
    goodDealBucket = 0
    badDealBucket = 0
    neutralDealBucket = 0
    allParentAsins = pd.read_csv('parent_asins.csv').drop_duplicates(subset='parent_asin')
    parentChildDfOG = pd.read_csv('parent_asins.csv')
    # 5-20, 20-40, 40-60, 60-80,80-100, 100+
    list_of_positives = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    midrange_asins = []
    print(len(allParentAsins))
    combinedDeals = {}
    positive_deal_types = {'COMBINED':0,'BEST_DEAL':0,'LIGHTNING_DEAL':0}
    for j in range (len(allParentAsins)):
        print(j)
        parentAsinValue = allParentAsins.iloc[j, allParentAsins.columns.get_loc('parent_asin')]
        parentChildDf = parentChildDfOG[parentChildDfOG['parent_asin'] == parentAsinValue].dropna()
        allDeals = allDealDates(parentChildDf)
        if (len(allDeals) > 0):
            groupedDeals,combinedDealLoopCounter = groupDeals(allDeals)
            combinedDeals = combine_and_sum_dicts(combinedDealLoopCounter,combinedDeals)
            columns = ['Group Of Deal','Start Date','End Date','Asins On Deal', 'Units Before Deal (14) Days)', 'Units During Deal', 'Units After Deal (14 Days)']
            parentAsinDf = pd.DataFrame(columns=columns)
            for i in range (len(groupedDeals)):
                list_of_info = ["Group " + str(i+1),groupedDeals[i][0], groupedDeals[i][-1], groupedDeals[i][1:-2]]
                list_of_numbers = getGroupedInfo(groupedDeals[i],parentChildDf)
                
                if ((list_of_numbers[0]*0.9) < list_of_numbers[1] < (list_of_numbers[0]*1.1)) and ((list_of_numbers[0]*0.9) < list_of_numbers[2] < (list_of_numbers[0]*1.1)):
                    neutralDealBucket +=1
                elif (list_of_numbers[0] < list_of_numbers[1] or list_of_numbers[0] < list_of_numbers[2]):
                    positive_deal_types[groupedDeals[i][-2]]+=1
                    goodDealBucket +=1
                    if -1 < list_of_numbers[0] < 1:
                        positivity_of_number = max(list_of_numbers[2],list_of_numbers[1])
                    elif list_of_numbers[0] <0:
                        positivity_of_number = max(-((list_of_numbers[1]-list_of_numbers[0])/list_of_numbers[0]),-((list_of_numbers[2]-list_of_numbers[0])/list_of_numbers[0]))
                    else:
                        positivity_of_number = max((list_of_numbers[1]-list_of_numbers[0])/list_of_numbers[0],(list_of_numbers[2]-list_of_numbers[0])/list_of_numbers[0])
                    print(positivity_of_number)
                    if positivity_of_number > 10:
                        list_of_positives[30] +=1
                    elif positivity_of_number > 9:
                        list_of_positives[29] +=1
                    elif positivity_of_number > 8:
                        list_of_positives[28] +=1
                    elif positivity_of_number > 7:
                        list_of_positives[27] +=1
                    elif positivity_of_number > 6:
                        list_of_positives[26] +=1
                    elif positivity_of_number > 5:
                        list_of_positives[25] +=1  
                    elif positivity_of_number > 4.8:
                        list_of_positives[24] +=1  
                    elif positivity_of_number > 4.6:
                        list_of_positives[23] +=1 
                    elif positivity_of_number > 4.4:
                        list_of_positives[22] +=1
                    elif positivity_of_number > 4.2:
                        list_of_positives[21] +=1    
                    elif positivity_of_number > 4:
                        list_of_positives[20] +=1                
                    elif positivity_of_number > 3.8:
                        list_of_positives[19] +=1                   
                    elif positivity_of_number > 3.6:
                        list_of_positives[18] +=1                    
                    elif positivity_of_number > 3.4:
                        list_of_positives[17] +=1
                    elif positivity_of_number > 3.2:
                        list_of_positives[16] +=1                    
                    elif positivity_of_number > 3:
                        list_of_positives[15] +=1

                    elif positivity_of_number > 2.8:
                        list_of_positives[14] +=1
                        midrange_asins.append([parentAsinValue,groupedDeals[i][0],groupedDeals[i][-1]])
                    elif positivity_of_number > 2.6:
                        list_of_positives[13] +=1
                        midrange_asins.append([parentAsinValue,groupedDeals[i][0],groupedDeals[i][-1]])
                    elif positivity_of_number > 2.4:
                        list_of_positives[12] +=1
                        midrange_asins.append([parentAsinValue,groupedDeals[i][0],groupedDeals[i][-1]])
                    elif positivity_of_number > 2.2:
                        list_of_positives[11] +=1  
                        midrange_asins.append([parentAsinValue,groupedDeals[i][0],groupedDeals[i][-1]])                  
                    elif positivity_of_number > 2:
                        list_of_positives[10] +=1
                        midrange_asins.append([parentAsinValue,groupedDeals[i][0],groupedDeals[i][-1]])

                    elif positivity_of_number > 1.8:
                        list_of_positives[9] +=1
                    elif positivity_of_number > 1.6:
                        list_of_positives[8] +=1
                    elif positivity_of_number > 1.4:
                        list_of_positives[7] +=1
                    elif positivity_of_number > 1.2:
                        list_of_positives[6] +=1
                    elif positivity_of_number > 1:
                        list_of_positives[5] +=1
                    elif positivity_of_number > 0.8:
                        list_of_positives[4] +=1
                    elif positivity_of_number > 0.6:
                        list_of_positives[3] +=1
                    elif positivity_of_number > 0.4:
                        list_of_positives[2] +=1
                    elif positivity_of_number > 0.2:
                        list_of_positives[1] +=1
                    elif positivity_of_number > 0:
                        list_of_positives[0] +=1
                else:
                    badDealBucket+=1
                parentAsinDf.loc[i] = list_of_info + list_of_numbers
                print("Distributions Of Positive Results: ",list_of_positives)
                print("List of Midrange Asins: ", midrange_asins)


    print("Distributions Of Positive Results: ",list_of_positives)
    print("List of Combined Deals: ", combinedDeals)
    print("Positive Deal Types: ", positive_deal_types)
    print("List of Midrange Asins: ", midrange_asins)
    return (goodDealBucket,neutralDealBucket,badDealBucket)

def getGroupedInfo (groupedDeals,parentChildDf):
    # beforeProfitAvg = 0
    # duringProfitAvg = 0
    # afterProfitAvg = 0
    beforeUnitAvg = 0
    duringUnitAvg = 0
    afterUnitAvg = 0
    priceBefore = 0
    priceDuring = 0
    priceAfter = 0
    start_date  = groupedDeals[0] 
    end_date  = groupedDeals[-1] 
    badAsins = groupedDeals[1:-1]
    count = 0
    for i in range (len(parentChildDf)):
        asinValue = parentChildDf.iloc[i, parentChildDf.columns.get_loc('asin')]
        if not(asinValue in badAsins):
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
            beforeDealNormalize = rootFunctions.normalizeAfterDealUnits(unit_df,unitBeforeDeal)
            duringDealNormalize = rootFunctions.normalizeAfterDealUnits(unit_df,unitDuringDeal)
            afterDealNormalize = rootFunctions.normalizeAfterDealUnits(unit_df,unitAfterDeal)
            # if rootFunctions.find_most_frequent_value(beforeDeal,unitBeforeDeal) == None and rootFunctions.find_most_frequent_value(afterDeal,unitAfterDeal) == None:
            #     cost_per_unit = 0
            # elif (rootFunctions.find_most_frequent_value(beforeDeal,unitBeforeDeal) == None):
            #     cost_per_unit = (rootFunctions.find_most_frequent_value(afterDeal,unitAfterDeal))/3 
            # else:
            #     cost_per_unit = (rootFunctions.find_most_frequent_value(beforeDeal,unitBeforeDeal))/3 
            # profitBefore= rootFunctions.profit_dataframe(beforeDeal,unitBeforeDeal, cost_per_unit)
            # profitDuring= rootFunctions.profit_dataframe(duringDeal,unitDuringDeal, cost_per_unit)
            # profitAfter= rootFunctions.profit_dataframe(afterDeal,unitAfterDeal, cost_per_unit)
            # list_profits = rootFunctions.profitNumbers14Days(profitBefore['profit'],start_date,profitAfter['profit'],end_date,profitDuring['profit'])
            priceBefore  = rootFunctions.find_most_frequent_value(beforeDeal,unitBeforeDeal)
            priceDuring = rootFunctions.find_most_frequent_value(duringDeal,unitDuringDeal)
            priceAfter =  rootFunctions.find_most_frequent_value(afterDeal,unitAfterDeal)
            if priceBefore == None:
                priceBefore = 0
            if priceDuring == None:
                priceDuring = 0
            if priceAfter == None:
                priceAfter = 0
            list_units = rootFunctions.unitsAt14Days(unitBeforeDeal,start_date,end_date,unitAfterDeal,unitDuringDeal)
            list_units_normal = rootFunctions.unitsAt14Days(beforeDealNormalize,start_date,end_date,afterDealNormalize,duringDealNormalize)
            if not((list_units[0] == 0) or (priceBefore > priceAfter) or (priceBefore > priceDuring)):
                # beforeProfitAvg += list_profits[0]/14
                # duringProfitAvg += list_profits[1]/profitDuring['profit'].count()
                # afterProfitAvg += list_profits[2]/14
                beforeUnitAvg += list_units_normal[0]/14
                duringUnitAvg += list_units_normal[1]/unitDuringDeal.count()
                afterUnitAvg += list_units_normal[2]/14

    # divConstant = (len(parentChildDf)-len(badAsins)-count)
    # if divConstant <= 0:
    #     return [0,0,0,0,0,0]
    #beforeProfitAvg,duringProfitAvg,afterProfitAvg
    
    new_row = [beforeUnitAvg,duringUnitAvg,afterUnitAvg]
    new_row[-6:] = [round(element, 2) for element in new_row[-6:]]
    return new_row


def combine_and_sum_dicts(dict1, dict2):
    combined_dict = dict1.copy()  
    for key, value in dict2.items():
        if key in combined_dict:
            combined_dict[key] += value
        else:
            combined_dict[key] = value
    return combined_dict

def allDealDates(parentChildDf):
    dealsDf = pd.read_csv('LUXE_DEALS.csv')
    columns = ['asin','start_date','end_date','promotion_type']
    allChildDeals = pd.DataFrame(columns=columns)
    for i in range (len(parentChildDf)):
        asinValue = parentChildDf.iloc[i, parentChildDf.columns.get_loc('asin')]
        asinDealDf = dealsDf[dealsDf['asin'] == asinValue].dropna()
        for j in range(len(asinDealDf)):
            new_row = [asinValue,asinDealDf.iloc[j, asinDealDf.columns.get_loc('start_datetime')][:10],asinDealDf.iloc[j, asinDealDf.columns.get_loc('end_datetime')][:10],asinDealDf.iloc[j, asinDealDf.columns.get_loc('promotion_type')]]
            allChildDeals.loc[len(allChildDeals)] = new_row
    allChildDeals = allChildDeals.sort_values(by=['start_date'], ascending=True)
    return allChildDeals

def groupDeals(allDealDatesOG):
    allDealDates = allDealDatesOG.copy(deep=True)
    allDealDates.loc[len(allDealDates)] = allDealDates.loc[len(allDealDates)-1]
    groupAsins = {}
    combinedDealsLength = {}
    count = 0
    i = 0
    boolCombined = False
    while  i < len(allDealDates)-1:
        groupAsins[count] = [allDealDates.iloc[i, allDealDates.columns.get_loc('start_date')], allDealDates.iloc[i, allDealDates.columns.get_loc('asin')]]
        if overlappedDates(allDealDates.iloc[i, allDealDates.columns.get_loc('start_date')],allDealDates.iloc[i+1, allDealDates.columns.get_loc('end_date')]):       
            while overlappedDates(allDealDates.iloc[i, allDealDates.columns.get_loc('start_date')],allDealDates.iloc[i+1, allDealDates.columns.get_loc('end_date')]) and (i < len(allDealDates)-2):
                groupAsins[count].append(allDealDates.iloc[i+1,allDealDates.columns.get_loc('asin')])
                i+=1
            boolCombined = True
        if(boolCombined):
             groupAsins[count].append("COMBINED")
             boolCombined = False
        elif (allDealDates.iloc[i, allDealDates.columns.get_loc('promotion_type')] == 'BEST_DEAL'):
             groupAsins[count].append("BEST_DEAL")
        elif (allDealDates.iloc[i, allDealDates.columns.get_loc('promotion_type')] == 'LIGHTNING_DEAL'):
            groupAsins[count].append("LIGHTNING_DEAL")

        groupAsins[count].append(allDealDates.iloc[i, allDealDates.columns.get_loc('end_date')])
        if (datetime.strptime(groupAsins[count][-1],'%Y-%m-%d') - datetime.strptime(groupAsins[count][0],'%Y-%m-%d')).days > 30:
            key = len(groupAsins[count])-2
            if key in combinedDealsLength:
                combinedDealsLength[key] += 1
            else:
                combinedDealsLength[key] = 1

        count+=1
        i+=1
    return groupAsins, combinedDealsLength
    
        
def overlappedDates(date1,date2):
    day1 = datetime.strptime(date1, '%Y-%m-%d')
    day2 = datetime.strptime(date2, '%Y-%m-%d')
    dateDifference = (day2-day1).days
    if(dateDifference < 21):
        return True
    return False




def main():
   print("Effects On Parent Asins (Positive, Neutral, Bad)",childAsins())



if __name__ == "__main__":
    main()

