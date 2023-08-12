
def main():
    #assigning variables
    asinValue = input()
    start_date = input()
    end_date = input()  
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
    print(cost_per_unit)
    #getting mean averages
    list_of_averages = averageDates(asinValue,start_date,end_date)
    
    #Calculate profits
    profitBefore= profit_dataframe(beforeDeal,unitBeforeDeal, cost_per_unit)
    profitDuring= profit_dataframe(duringDeal,unitDuringDeal, cost_per_unit)
    profitAfter= profit_dataframe(afterDeal,unitAfterDeal, cost_per_unit)
    printSimpleTable(profitAfter,profitBefore,unitBeforeDeal,start_date,end_date,unitAfterDeal)
    recovery_date = timeUntilProfit(normalizeAfterDeal(profitBefore,profitAfter)['profit'],discountLoss(unitBeforeDeal,beforeDeal,duringDeal,unitDuringDeal))
    profit_date = profitableTime(profitAfter,profitBefore) 

    user_choice = int(input("\nEnter a case number (1 for Profit Tables, 2 for Growth Charts, 3 for Straight to Graphs): "))
    switch_case(user_choice,beforeDeal,afterDeal,duringDeal,unitDuringDeal,unitBeforeDeal,unitAfterDeal,cost_per_unit,start_date,end_date,profit_date,profitAfter,recovery_date,profitBefore)
    fig1 = mp.figure(figsize=(12, 8))
    fig2 = mp.figure(figsize=(12, 8))

    #Plot graph with linear regression of revenue
    mp.figure(fig1.number) #mp.subplot(3,1,1)
    grouped_df.truncate(before=beforeDeal.index.min()).plot(x="date_trunc", y=["sales"],grid = True, kind="line")
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
    grouped_df.truncate(before=beforeDeal.index.min()).plot(x="date_trunc", y=["sales"],grid = True, kind="line")
    mp.hlines(y=list_of_averages[0], xmin=beforeDeal.index.min(), xmax=beforeDeal.index.max(), colors='#301934', linestyles='-', lw=2, label='Average 1')
    mp.hlines(y=list_of_averages[1], xmin=duringDeal.index.min(), xmax=duringDeal.index.max(), colors='#301934', linestyles='-', lw=2, label='Average 2')
    mp.hlines(y=list_of_averages[2], xmin=afterDeal.index.min(), xmax=afterDeal.index.max(), colors='#301934', linestyles='-', lw=2, label='Average 3')    
    mp.axvspan(duringDeal.index.min(),duringDeal.index.max(), facecolor='0.2', alpha=0.5)
    graphLabels(recovery_date,afterDeal,profit_date)
    mp.title("Average Sales per Day")

    #Plot trend lines for profit
    # mp.figure(fig3.number) 
    profit_dataframe(grouped_df, unit_df,cost_per_unit).truncate(before=profitBefore.index.min()).plot(y=["profit"],grid = True, kind="line", figsize=(12, 8))
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
