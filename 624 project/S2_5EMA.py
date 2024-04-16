import pandas as pd
import datetime
indexList = ["NIFTY50",
"NIFTY500",
"NIFTYAUTO",
"NIFTYBANK",
"NIFTYCOMMODITIES",
"ADANIPORTS",
"ASIANPAINT",
"AXISBANK",
"BAJAJFINSV",
"BAJFINANCE",
"BHARTIARTL",
"BPCL",
"BRITANNIA",
"CIPLA",
"COALINDIA",
"DIVISLAB",
"DRREDDY",
"EICHERMOT",
"GAIL",
"GRASIM",
"HCLTECH",
"HDFCAMC",
"HDFCBANK",
"HDFCLIFE",
"HEROMOTOCO",
"HINDALCO",
"HINDUNILVR",
"ICICIBANK",
"INDUSINDBK",
"INFY",
"IOC",
"ITC",
"JSWSTEEL",
"KOTAKBANK",
"LT",
"MARUTI",
"NESTLEIND",
"NTPC",
"ONGC",
"POWERGRID",
"RELIANCE",
"SBILIFE",
"SBIN",
"SHREECEM",
"SUNPHARMA",
"TATAMOTORS",
"TATASTEEL",
"TCS",
"TECHM",
"TITAN",
"ULTRACEMCO",
"UPL",
"WIPRO",
"AARTIIND",
"ABCAPITAL",
"ABFRL",
"ADANIENT",
"ATGL",
"AJANTPHARM",
"APLLTD",
"APOLLOHOSP",
"APOLLOTYRE",
"ASHOKLEY",
"AUBANK",
"BALKRISIND",
"BANKINDIA",
"BATAINDIA",
"BBTC",
"BEL",
"BHARATFORG",
"BHEL",
"CANBK",
"CASTROLIND",
"CESC",
"CHOLAFIN",
"COFORGE",
"COROMANDEL",
"CROMPTON",
"CUB",
"DALBHARAT",
"EDELWEISS",
"EMAMILTD",
"ENDURANCE",
"ESCORTS",
"EXIDEIND",
"FEDERALBNK",
"FORTIS",
"GMRINFRA",
"GODREJAGRO",
"GODREJIND",
"GODREJPROP",
"GSPL",
"GUJGASLTD",
"HUDCO",
"IBULHSGFIN",
"IDEA",
"IDFCFIRSTB",
"INDHOTEL",
"IPCALAB"]



wallet = 100
tradesDict = {"buytime":[],"buyprice":[],"stoploss":[],"target":[],"selltime":[],"profit":[],"open":[],"high":[],"close":[],"low":[],"5ema":[],"9ema":[],"21ema":[],"rsi":[]
              ,"openCloseDiff":[],"ema5_highDiff":[],"ema5_lowDiff":[]}

for index in indexList:
    print(index)
    df=pd.read_csv("./Data/{}.csv".format(index))
    totalProfit = 0; profitTrades = 0; lossTrades = 0; totalTrades = 0
    flag=False
    # raw_data['Mycol'] = pd.to_datetime(raw_data['Mycol'])
    for i in range(35,len(df["openTime"])-1):
        if flag==False:
            currentLow=df["low"][i]
            ema_5 = df["5ema"][i]
            ema_9 = df["9ema"][i]
            ema_21 = df["21ema"][i]
            rsi = df["rsi"][i]
            currentHigh=df["high"][i]
            TempcurrentHigh=df["high"][i]
            TempcurrentLow=df["low"][i]
            currentClose = df["close"][i]
            currentOpen = df["open"][i]
            nextHigh=df["high"][i+1]
            nextLow=df["low"][i+1]
            # rsi=df["rsi"][i]
            if currentLow>ema_5 and nextLow<currentLow :
                # 
                buyPrice=currentLow
                qty=wallet/buyPrice
                stopLoss=currentHigh
                target=buyPrice-((stopLoss-buyPrice)*8)
                # print("============================")
                buydate = datetime.datetime.fromtimestamp(df["openTime"][i])
                # print("buy",buydate,buyPrice,stopLoss,target)
                flag=True
        else:
            #sell
            low=df["low"][i]
            currentHigh=df["high"][i]

            if(currentHigh>stopLoss or low<target):
                if currentHigh>stopLoss:
                    exitPrice=stopLoss

                elif low<target:
                    exitPrice=target
                tradesDict["buytime"].append(buydate)
                tradesDict["buyprice"].append(buyPrice)
                tradesDict["stoploss"].append(stopLoss)
                tradesDict["target"].append(target)
                tradesDict["open"].append(currentOpen)
                tradesDict["high"].append(TempcurrentHigh)
                tradesDict["low"].append(currentLow)
                tradesDict["close"].append(currentClose)
                tradesDict["rsi"].append(rsi)
                tradesDict["5ema"].append(ema_5)
                tradesDict["9ema"].append(ema_9)
                tradesDict["21ema"].append(ema_21)
                tradesDict["openCloseDiff"].append(((currentOpen-currentClose)/currentOpen)*100)
                tradesDict["ema5_highDiff"].append(((TempcurrentHigh-ema_5)/TempcurrentHigh)*100)
                tradesDict["ema5_lowDiff"].append(((TempcurrentLow-ema_5)/TempcurrentLow)*100)
                date = datetime.datetime.fromtimestamp(df["openTime"][i])
                tradesDict["selltime"].append(date)
                # print("sell",date,exitPrice)
                profit = (buyPrice-exitPrice)*qty
                tradesDict["profit"].append(profit)
                # print("PROFIT : ",profit)
                if(profit > 0):
                    profitTrades = profitTrades+1
                else:
                    lossTrades = lossTrades+1
                totalProfit = totalProfit+profit
                flag=False

    print()
    print("=================================")
    print("totalProfit",totalProfit)
    print("profitTrades",profitTrades)
    print("lossTrades",lossTrades)
    print("totalTrades",profitTrades+lossTrades)
    print("=================================")
tradesDf = pd.DataFrame.from_dict(tradesDict)
tradesDf.to_csv("./S2_trades.csv")