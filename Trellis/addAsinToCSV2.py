import rootFunctions
import gspread
import pandas as pd
import csv
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime, timedelta




def main():
    #assigning variables
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    credentials = ServiceAccountCredentials.from_json_keyfile_name('credentials.json', scope)
    gc = gspread.authorize(credentials)
    df = pd.read_csv("LUXE_DEALS.csv")
    df_restraints = df.copy(deep=True)
    start_date = input()
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = start_date + timedelta(days=30)
    df_restraints['start_datetime'] = pd.to_datetime(df_restraints['start_datetime']).dt.tz_localize(None)
    truncated_df = df_restraints[(df_restraints["start_datetime"] >= start_date) & (df_restraints["start_datetime"] <= end_date)]
    columns = ['Asin', 'Deal Type', 'Uplift', 'Was there Incremental Profit During Deal', 'Incremental Profit/Loss During Deal', 'Deal Discount Costs','Profit Before Deal (30 Days)', 'Profit During Deal', 'Profit After Deal (30 Days)','Units Before Deal (30 Days)', 'Units During Deal', 'Units After Deal (30 Days)', 'Price Of Unit Before Deal','Price Of Unit During Deal','Price Of Unit After Deal']
    dealInfo = pd.DataFrame(columns=columns)
    sheet_name = 'Trellis Data April'
    sheet = gc.open(sheet_name)
    for i in range(len(truncated_df)):
        asinValue = truncated_df.iloc[i, truncated_df.columns.get_loc('asin')]
        start_date = str(truncated_df.iloc[i, truncated_df.columns.get_loc('start_datetime')])[:10]
        end_date = str(truncated_df.iloc[i, truncated_df.columns.get_loc('end_datetime')])[:10]  
        discount = 0
        grouped_df = rootFunctions.filterByAsin(asinValue)[1]
        unit_df = rootFunctions.filterByAsin(asinValue)[0]
        subsheet_name = asinValue
        try:
            subsheet = sheet.worksheet(subsheet_name)
        except gspread.WorksheetNotFound:
            subsheet = sheet.add_worksheet(title=subsheet_name, rows="100", cols="20")  # Adjust the rows and cols as needed


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
        if rootFunctions.find_most_frequent_value(beforeDeal,unitBeforeDeal) == None and rootFunctions.find_most_frequent_value(afterDeal,unitAfterDeal) == None:
            cost_per_unit = 0
        elif (rootFunctions.find_most_frequent_value(beforeDeal,unitBeforeDeal) == None):
            cost_per_unit = (rootFunctions.find_most_frequent_value(afterDeal,unitAfterDeal))/3 
        else:
            cost_per_unit = (rootFunctions.find_most_frequent_value(beforeDeal,unitBeforeDeal))/3
        list_of_averages = rootFunctions.averageDates(asinValue,start_date,end_date)
        
        profitBefore= rootFunctions.profit_dataframe(beforeDeal,unitBeforeDeal, cost_per_unit)
        profitDuring= rootFunctions.profit_dataframe(duringDeal,unitDuringDeal, cost_per_unit)
        profitAfter= rootFunctions.profit_dataframe(afterDeal,unitAfterDeal, cost_per_unit)
        if (rootFunctions.find_most_frequent_value(beforeDeal,unitBeforeDeal) >= rootFunctions.find_most_frequent_value(afterDeal,unitAfterDeal) and (rootFunctions.unitsAtInterval(unitBeforeDeal,start_date,end_date,unitAfterDeal)[0])/30 < (rootFunctions.unitsAtInterval(unitBeforeDeal,start_date,end_date,unitAfterDeal)[1])/30):
            Uplift = 'Y'
        else:
            Uplift = 'N'

        new_row = [asinValue, df.iloc[i, df.columns.get_loc('promotion_type')], Uplift, 'Y' if rootFunctions.netLoss(unitBeforeDeal,beforeDeal,unitDuringDeal,duringDeal,cost_per_unit) > 0 else 'N', rootFunctions.netLoss(unitBeforeDeal,beforeDeal,unitDuringDeal,duringDeal,cost_per_unit),
                   rootFunctions.discountLoss(unitBeforeDeal,beforeDeal,duringDeal,unitDuringDeal,afterDeal,unitAfterDeal), rootFunctions.profitNumbers(profitBefore['profit'],start_date,profitAfter['profit'],end_date,profitDuring['profit'])[0]/30, 
                   rootFunctions.profitNumbers(profitBefore['profit'],start_date,profitAfter['profit'],end_date,profitDuring['profit'])[1]/profitDuring['profit'].count(),rootFunctions.profitNumbers(profitBefore['profit'],start_date,profitAfter['profit'],end_date,profitDuring['profit'])[2]/30,
                   (rootFunctions.unitsAtInterval(unitBeforeDeal,start_date,end_date,unitAfterDeal)[0])/30, unitDuringDeal.sum()/unitDuringDeal.count(), 
                   (rootFunctions.unitsAtInterval(unitBeforeDeal,start_date,end_date,unitAfterDeal)[1])/30, rootFunctions.find_most_frequent_value(beforeDeal,unitBeforeDeal),rootFunctions.find_most_frequent_value(duringDeal,unitDuringDeal), rootFunctions.find_most_frequent_value(afterDeal,unitAfterDeal)]
        #rootFunctions.profitNumbers(profitAfter['profit'],end_date)[0],rootFunctions.profitNumbers(profitAfter['profit'],end_date)[1]
        new_row[-9:] = [round(element, 2) for element in new_row[-9:]]

        dealInfo.loc[i] = new_row
        df1 = rootFunctions.getSimpleInfo2(beforeDeal,unitBeforeDeal,duringDeal,unitDuringDeal,cost_per_unit,start_date,end_date,afterDeal,unitAfterDeal)
        df2 = rootFunctions.get7DayProfitInfo(beforeDeal,unitBeforeDeal,duringDeal,unitDuringDeal,afterDeal,unitAfterDeal,cost_per_unit,start_date,end_date)
        df3 = rootFunctions.moreInformation(profitBefore,profitAfter,unitBeforeDeal,beforeDeal,duringDeal,unitDuringDeal,afterDeal,unitAfterDeal)
        subsheet.clear()
        csv_file = 'newCsv.csv'
        df1.to_csv(csv_file, index=False, mode='w')

        with open(csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([]) 
            writer.writerow(['7 Day Information'])  

            df2.to_csv(file, index=False)

        with open(csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([])
            writer.writerow(['More Information'])  

            df3.to_csv(file, index=False)
        with open(csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([])
            writer.writerow(["Deal Type: "+df.iloc[i, df.columns.get_loc('promotion_type')]])
        f = open(csv_file, "r")
        values = [r for r in csv.reader(f)]
        subsheet.update(values)

        with open(csv_file, 'w') as file:
            file.write("")
    overAllData = 'output.csv'
    print(dealInfo.to_string(index=False))
    dealInfo.to_csv(overAllData, index=False, mode='w')


if __name__ == "__main__":
    main()
