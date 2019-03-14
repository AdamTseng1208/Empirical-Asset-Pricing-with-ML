#########################################################################################################
##           The Data Incubator - Capstone Project: Empirical Asset Pricing with Machine Learning
#########################################################################################################

########################### Load Packages and Environment Setup ########################### 

# Basic Packages
import numpy as np
import pandas as pd
import time
import math
import statsmodels.api as sm
import matplotlib.pyplot as plt
import datetime
from copy import copy
import pandas_datareader.data as web
# Machine Learning Package
import statsmodels.api as sm
# Spark Packages
import findspark
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.types import Row, StructField, StructType, StringType, IntegerType
import pyspark.sql.functions as F
from pyspark.sql.window import Window
# Spark Machine Learning Packages
from pyspark.ml.feature import VectorAssembler, OneHotEncoderEstimator, StringIndexer
from pyspark.ml.regression import LinearRegression, DecisionTreeRegressor, GBTRegressor, GeneralizedLinearRegression, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

# Get the Spark Environment Ready
sc = SparkContext(master="local[4]") #run spark locally on my computer and use four workers
print(sc)
sqlContext = SQLContext(sc)
print(sqlContext)

# Clear the cache
sqlContext.clearCache()

# General setting
# Specify date
start_date = pd.to_datetime("1957-03-01").date() #Specify start date
train_date = pd.to_datetime("1974-12-31").date() #Specify the training end date
validation_date = pd.to_datetime("1986-12-31").date() #Specify the validation end date
recent_date = pd.to_datetime("2010-01-01").date()
end_date = pd.to_datetime("2016-12-31").date() #Specify end date
today_date = datetime.datetime.today().date() #Specify today

#%%
########################################################################################
#                 FF-Factors Monthly Data Load In and Basic Cleaning
########################################################################################

# Load in value-weighted monthly market index and risk-free return
ff_factors = pd.read_csv("/Volumes/Transcend/1. Source Files/F-F_Research_Data_Factors.CSV")
ff_factors = ff_factors[:1100]
ff_factors["Mkt-RF"] = ff_factors["Mkt-RF"].astype(float) * 0.01
ff_factors["RF"] = ff_factors["RF"].astype(float) * 0.01
ff_factors["Date"] = pd.to_datetime(ff_factors["Date"], format = "%Y%m")
ff_factors["year"] = pd.to_datetime(ff_factors["Date"]).dt.year
ff_factors["month"] = pd.to_datetime(ff_factors["Date"]).dt.month
ff_factors['Mkt'] = ff_factors['Mkt-RF'] + ff_factors['RF'] 
ff_factors = ff_factors.rename(columns = {"Mkt-RF": "Mkt_RF"})
ff_factors = ff_factors.loc[(ff_factors["Date"] >= start_date)& (ff_factors["Date"] <= end_date)]
ff_factors_df = ff_factors.copy(deep = True)
ff_factors = sqlContext.createDataFrame(ff_factors) #convert pandas dataframe into spark dataframe
#print(type(ff_factors))
#print(ff_factors.count()) #718 rows
#print(ff_factors.show(10))
#print(ff_factors.printSchema())


#%%
########################################################################################
#                                 Macroeconomic Variables
########################################################################################
macro_variables_collect = ['CPIAUCSL', 'CPILFESL', 'GDP', 'GS1', 'GS10', 'T10Y2Y', 'T10Y3M', 'TEDRATE']
macro_data = web.DataReader(macro_variables_collect, 'fred', start_date, end_date)


#%%
########################################################################################
#                      Stock Monthly Data Load In and Basic Cleaning
#########################################################################################

# Load and check the monthly stock data
stock_monthly = sqlContext.read.format("com.databricks.spark.csv").options(header = "true", inferschema = "true").load("/Volumes/Transcend/1. Source Files/CRSP_Stocks_Data.csv")
#print(type(stock_monthly)) # data type: spark dataframe
#print(stock_monthly.count()) #check number of rows: 4,514,430
#print(stock_monthly.show(30)) #look at some rows
#print(stock_monthly.printSchema()) #look at the schema
#stock_monthly.select("SHRCD").describe().show() #generate basic statistics
# Column operations
stock_monthly = stock_monthly.withColumn("SICCD", 0 + stock_monthly.SICCD)
stock_monthly = stock_monthly.withColumn("RET", F.when((stock_monthly.RET == "-99") | (stock_monthly.RET == "-88") | (stock_monthly.RET == "-77") | (stock_monthly.RET == '-66') | (stock_monthly.RET == '-55') | (stock_monthly.RET == '-44') | (stock_monthly.RET == 'B') | (stock_monthly.RET == "C") , None).otherwise(stock_monthly.RET + 0))
stock_monthly = stock_monthly.withColumn("PRC", F.abs(stock_monthly.PRC)) #convert all prc into positive
stock_monthly = stock_monthly.withColumn("PRC_Lag", F.lag(stock_monthly.PRC).over(Window.partitionBy("PERMNO").orderBy("date")))
stock_monthly = stock_monthly.withColumn("ME", stock_monthly.PRC * stock_monthly.SHROUT) #Add the ME column
stock_monthly = stock_monthly.withColumn("date", F.to_date(stock_monthly.date, "yyyy/MM/dd")) #convert date
stock_monthly = stock_monthly.withColumn("fdate", F.add_months(stock_monthly.date, -18))
stock_monthly = stock_monthly.withColumn("year", F.year(stock_monthly.date)) #create year
stock_monthly = stock_monthly.withColumn("month", F.month(stock_monthly.date)) #create month
stock_monthly = stock_monthly.withColumn("fyear", F.year(stock_monthly.fdate)) #create fyear

# Filter operations
stock_monthly = stock_monthly.filter(stock_monthly.date >= start_date)
stock_monthly = stock_monthly.filter(stock_monthly.date <= end_date)
stock_monthly.count() #4,022,530
stock_monthly = stock_monthly.filter((stock_monthly.SHRCD == 10) | (stock_monthly.SHRCD == 11))
stock_monthly = stock_monthly.filter((stock_monthly.EXCHCD == 1) | (stock_monthly.EXCHCD == 2) | (stock_monthly.EXCHCD == 3)) 
#print(stock_monthly.count()) #now left 3,090,938

# Industry Segmentations and One-hot Encoding
stock_monthly = stock_monthly.withColumn("SICCD_Industry", F.when(stock_monthly.SICCD > 9000, "Public_Admin").otherwise(F.when(stock_monthly.SICCD >= 7000, "Services").otherwise(F.when(stock_monthly.SICCD >= 6000, "Finance").otherwise(F.when(stock_monthly.SICCD >= 5200, "Retail").otherwise(F.when(stock_monthly.SICCD >= 5000, "Wholesale").otherwise(F.when(stock_monthly.SICCD >= 4000, "Transportation").otherwise(F.when(stock_monthly.SICCD >= 2000, "Manufacturing").otherwise(F.when(stock_monthly.SICCD >= 1500, "Construction").otherwise(F.when(stock_monthly.SICCD >= 1000, "Mining").otherwise("Argriculture"))))))))))
string_encoder = StringIndexer(inputCol = "SICCD_Industry", outputCol = "SICCD_Index")
stock_monthly = string_encoder.fit(stock_monthly).transform(stock_monthly)
hot_encoder = OneHotEncoderEstimator(inputCols = ["SICCD_Index"], outputCols = ["Industry"])
stock_monthly = hot_encoder.fit(stock_monthly).transform(stock_monthly)
#stock_monthly['SICCD_Industry', 'SICCD_Index', 'Industry'].distinct().show() # take a look at the industry

# Create new columns
stock_monthly = stock_monthly.withColumn("DVOL", F.log(stock_monthly.PRC * stock_monthly.VOL))
stock_monthly = stock_monthly.withColumn("DVOL", F.lag(stock_monthly.DVOL).over(Window.partitionBy("PERMNO").orderBy("date")))
stock_monthly = stock_monthly.withColumn("TURN", F.log(stock_monthly.VOL / stock_monthly.SHROUT))
stock_monthly = stock_monthly.withColumn("TURN", F.lag(stock_monthly.TURN).over(Window.partitionBy("PERMNO").orderBy("date")))

#%%
########################################################################################
#                                  Momentum Factor - Building
########################################################################################

###################### Momentum 36, 12, 6 month (Long-term Momentum) ######################

# Add needed columns
stock_monthly = stock_monthly.withColumn("RET_Plus1", 1 + stock_monthly.RET) #RET_Plus1 column
stock_monthly = stock_monthly.withColumn("RET_Sign", F.when(stock_monthly.RET >= 0, 1).otherwise(0))

sqlContext.registerDataFrameAsTable(stock_monthly, "st_monthly") 
query = """
SELECT * , SUM(LOG(RET_Plus1)) OVER (PARTITION BY PERMNO ORDER BY date ROWS BETWEEN 12 PRECEDING AND 2 PRECEDING) AS Momentum_12, 
SUM(LOG(RET_Plus1)) OVER (PARTITION BY PERMNO ORDER BY date ROWS BETWEEN 7 PRECEDING AND 2 PRECEDING) AS Momentum_6,
SUM(LOG(RET_Plus1)) OVER (PARTITION BY PERMNO ORDER BY date ROWS BETWEEN 36 PRECEDING AND 2 PRECEDING) AS Momentum_36
FROM st_monthly
"""
stock_monthly = sqlContext.sql(query)
stock_monthly = stock_monthly.withColumn("Momentum_12", F.exp(stock_monthly.Momentum_12))
stock_monthly = stock_monthly.withColumn("Momentum_12", stock_monthly.Momentum_12 - 1)
stock_monthly = stock_monthly.withColumn("Momentum_6", F.exp(stock_monthly.Momentum_6))
stock_monthly = stock_monthly.withColumn("Momentum_6", stock_monthly.Momentum_6 - 1)
stock_monthly = stock_monthly.withColumn("Momentum_36", F.exp(stock_monthly.Momentum_36))
stock_monthly = stock_monthly.withColumn("Momentum_36", stock_monthly.Momentum_36 - 1)

###################### Momentum 1 month (Short-term reversal) ######################
stock_monthly = stock_monthly.withColumn("Reversal", F.lag(stock_monthly.RET).over(Window.partitionBy("PERMNO").orderBy("date")))


###################### Momentum Consistency ######################

sqlContext.registerDataFrameAsTable(stock_monthly, "st_monthly") 
query = """
SELECT * , SUM(RET_Sign) OVER (PARTITION BY PERMNO ORDER BY date ROWS BETWEEN 12 PRECEDING AND 2 PRECEDING) AS Momentum_Count
FROM st_monthly
"""
stock_monthly = sqlContext.sql(query)


#%%
########################################################################################
#                       Beta Factor - Building (Betting Against Beta)
########################################################################################

# Join the ff_factor table and CRSP dataset
sqlContext.registerDataFrameAsTable(ff_factors, "ff") 
sqlContext.registerDataFrameAsTable(stock_monthly, "st") 
query = """
SELECT st.* , ff.Mkt_RF, ff.RF
FROM st
LEFT JOIN ff
ON st.year = ff.year and st.month = ff.month
"""
stock_monthly = sqlContext.sql(query)

# Calculate stock_monthly correlation and volatilities
stock_monthly = stock_monthly.withColumn("Ex_RET", stock_monthly.RET - stock_monthly.RF)
sqlContext.registerDataFrameAsTable(stock_monthly, "st") 
query = """
SELECT * , STDDEV_SAMP(Ex_RET) OVER (PARTITION BY PERMNO ORDER BY date ROWS BETWEEN 11 PRECEDING AND CURRENT ROW) AS Stock_SD,
           STDDEV_SAMP(Mkt_RF) OVER (PARTITION BY PERMNO ORDER BY date ROWS BETWEEN 11 PRECEDING AND CURRENT ROW) AS Mkt_SD,
           CORR(Ex_RET, Mkt_RF) OVER (PARTITION BY PERMNO ORDER BY date ROWS BETWEEN 59 PRECEDING AND CURRENT ROW) AS Corr
FROM st
"""
stock_monthly = sqlContext.sql(query)

# Calculate Beta
stock_monthly = stock_monthly.withColumn("Beta", stock_monthly.Corr * stock_monthly.Stock_SD / stock_monthly.Mkt_SD)

# Shrink the Beta
stock_monthly = stock_monthly.withColumn("Beta_Shrink", 0.6 * stock_monthly.Beta + 0.4)

# Create the Lag Beta
stock_monthly = stock_monthly.withColumn("Beta_Lag", F.lag(stock_monthly.Beta, 1).over(Window.partitionBy("PERMNO").orderBy("date")))
stock_monthly = stock_monthly.withColumn("Beta_Shrink_Lag", F.lag(stock_monthly.Beta_Shrink, 1).over(Window.partitionBy("PERMNO").orderBy("date")))

# Create the square of the beta
stock_monthly = stock_monthly.withColumn("Beta_Lag_Square", stock_monthly['Beta_Lag'] ** 2)
stock_monthly = stock_monthly.withColumn("Beta_Shrink_Lag_Square", stock_monthly['Beta_Shrink_lag'] ** 2)

#%%
########################################################################################
#                                   Size Factor - Building
########################################################################################

# create size factor
stock_monthly = stock_monthly.withColumn("Size", F.lag(stock_monthly.ME, 1).over(Window.partitionBy("PERMNO").orderBy("date")))
stock_monthly = stock_monthly.withColumn("Log_Size", F.log(stock_monthly.Size))
#print(stock_monthly["date", "PERMNO", "Size", "Log_Size", "RET"].show(30))

# Value-Weighted Calculation
total_mktcap = stock_monthly.groupBy('date').agg({"Size": "sum"})
total_mktcap = total_mktcap.withColumnRenamed('sum(Size)', 'total_mktcap')
sqlContext.registerDataFrameAsTable(stock_monthly, "st") 
sqlContext.registerDataFrameAsTable(total_mktcap, "mktcap") 
query = """
SELECT st.* , mktcap.total_mktcap
FROM st LEFT JOIN mktcap
ON st.date = mktcap.date
"""
stock_monthly = sqlContext.sql(query)
stock_monthly = stock_monthly.withColumn("weight_mktcap", stock_monthly.Size / stock_monthly.total_mktcap)
## test
#stock_monthly.groupBy("date").agg({"weight_mktcap":"sum"}).show()


#%%
########################################################################################
#                                   Filter Stocks Once Again
########################################################################################

## Filter out Penny Stocks
#stock_monthly = stock_monthly.filter(stock_monthly.PRC >= 5)
#stock_monthly.count()

# Filter out Stocks without return
stock_monthly = stock_monthly.filter(stock_monthly.RET.isNotNull())
#stock_monthly.count()

## Test and take a closer look at the stock return
#test = stock_monthly.orderBy("RET", ascending=False)
#test.select('PERMNO', 'date', 'PRC', 'RET', 'DLRET').show(50)

#%%
########################################################################################
#                      Stock Fundamental Data Load In and Basic Cleaning
#########################################################################################

# Load in and Clean Linkage_CCM Data
Linkage_ccm = sqlContext.read.format("com.databricks.spark.csv").options(header = "true", inferschema = "true").load("/Volumes/Transcend/1. Source Files/Linkage_ccm.csv") #31,632
Linkage_ccm = Linkage_ccm.filter((Linkage_ccm.LINKTYPE == "LU") | (Linkage_ccm.LINKTYPE == "LC"))
Linkage_ccm = Linkage_ccm.filter((Linkage_ccm.LINKPRIM == "P") | (Linkage_ccm.LINKPRIM == "C")) #28,796

# Load in Compustat Data
Compustat = sqlContext.read.format("com.databricks.spark.csv").options(header = "true", inferschema = "true").load("/Volumes/Transcend/1. Source Files/Compustat_stocks_fundamental.csv") #542,386  

# Merge Compustat and Linkage_CCM
sqlContext.registerDataFrameAsTable(Linkage_ccm, "link") 
sqlContext.registerDataFrameAsTable(Compustat, "comp") 
query = """
SELECT comp. * , link.LINKPRIM, link.LIID, link.LINKTYPE, link.LPERMNO, link.LPERMCO, link.LINKDT, link.LINKENDDT, link.conm, link.naics, link.FYRC
FROM comp LEFT JOIN link
ON comp.GVKEY = link.GVKEY
"""
Compustat = sqlContext.sql(query) 
#Compustat.count() #636,655

# Compustat Cleaning
Compustat = Compustat.withColumn("datadate", F.to_date(Compustat.datadate, "yyyy/MM/dd"))
Compustat = Compustat.withColumn("LINKENDDT", F.to_date(Compustat.LINKENDDT, "yyyyMMdd"))
Compustat = Compustat.withColumn("LINKENDDT", F.when(Compustat.LINKENDDT.isNotNull(), Compustat.LINKENDDT).otherwise(today_date))
Compustat = Compustat.withColumn("LINKDT", Compustat["LINKDT"].cast("string"))
Compustat = Compustat.withColumn("LINKDT", F.to_date(Compustat.LINKDT, "yyyyMMdd"))
Compustat = Compustat.filter((Compustat.datadate >= Compustat.LINKDT) & (Compustat.datadate <= Compustat.LINKENDDT)) 
Compustat = Compustat.filter(Compustat.datafmt == "STD")
#Compustat['comp.GVKEY', 'datadate', 'fyear', 'cusip', 'LPERMNO', "LPERMCO", 'LINKDT', 'LINKENDDT'].show(100)
#Compustat.count() #327,236

## Merge Compustat with Pension data
#Pension = sqlContext.read.format("com.databricks.spark.csv").options(header = "true", inferschema = "true").load("/Volumes/Transcend/1. Source Files/data.pension.csv") #136,470   
#Pension = Pension.filter(Pension.datafmt == "STD") #136,470 
#Pension = Pension.withColumn("datadate", F.to_date(Pension.datadate, "yyyy/MM/dd"))
#sqlContext.registerDataFrameAsTable(Compustat, "comp")
#sqlContext.registerDataFrameAsTable(Pension, "pension")
#query = """
#SELECT comp.* , pension.*
#FROM comp 
#LEFT JOIN pension
#ON comp.GVKEY = pension.gvkey and comp.datadate = pension.datadate
#"""
#Compustat = sqlContext.sql(query) #295,300
#Compustat['comp.datadate', 'prba', "comp.GVKEY"].show(100)


#%%
########################################################################################
#                   Examine the CRSP and Compustat Dataset Individually
########################################################################################

# Examine the Compustat Dataset
Compustat.printSchema()
Compustat.count()
Test_Compustat = Compustat.filter(Compustat['gvkey'] == 10006)['gvkey', 'datadate', 'fyear', 'act', 'am', 'at', 'ci', 'ni', 'txdi', 'comp.sic', 'FYRC', 'LINKDT', 'LINKENDDT', 'at_lag', 'at_growth'].toPandas()

# Examine the CRSP Dataset
stock_monthly.printSchema()
stock_monthly.count()
Test_CRSP = stock_monthly.filter(stock_monthly['PERMNO'] == 10137).toPandas()
Test_CRSP2 = stock_monthly.filter(stock_monthly['PERMNO'] == 10006).toPandas()

#%%
######################################################################################################
#                Combine the Stock Monthly Data and Compustat And Construct the Value Factor
######################################################################################################

# Calculate Book Equity value in Compustat
Compustat = Compustat.withColumn("she", F.when(Compustat.seq.isNotNull(), Compustat.seq).otherwise(F.when((Compustat.ceq + Compustat.pstk).isNotNull(), (Compustat.ceq + Compustat.pstk)).otherwise(Compustat.at - Compustat.lt)))
Compustat = Compustat.withColumn("dt", F.when(Compustat.txditc.isNotNull(), Compustat.txditc).otherwise(F.when((Compustat.itcb + Compustat.txdb).isNotNull(), (Compustat.itcb + Compustat.txdb)).otherwise(F.when(Compustat.itcb.isNotNull(), Compustat.itcb).otherwise(Compustat.txdb))))
Compustat = Compustat.withColumn("ps", F.when(Compustat.pstkrv.isNotNull(), Compustat.pstkrv).otherwise(F.when(Compustat.pstkl.isNotNull(), Compustat.pstkl).otherwise(Compustat.pstk)))
Compustat = Compustat.withColumn("be", Compustat.she - Compustat.ps + Compustat.dt)
Compustat = Compustat.withColumn("be", F.when(Compustat.she.isNotNull(), Compustat.be).otherwise(None)) #295,300
#Compustat['datadate', 'GVKEY', 'fyear', 'LPERMNO', 'LPERMCO'].show(150)
#Compustat.select([F.count(F.when(F.isnull(c), c)).alias(c) for c in ['seq', 'she', 'dt', 'ps', 'be', 'at', 'lt']]).show()

# Calculate Asset Growth
Compustat = Compustat.withColumn("at_lag", F.lag(Compustat.at).over(Window.partitionBy("gvkey").orderBy("datadate")))
Compustat = Compustat.withColumn("at_growth", (Compustat.at - Compustat.at_lag) / Compustat.at_lag)

# Calculate Revenue Growth
Compustat = Compustat.withColumn("revt_lag", F.lag(Compustat['revt']).over(Window.partitionBy("gvkey").orderBy('datadate')))
Compustat = Compustat.withColumn("revt_growth", (Compustat['revt'] - Compustat['revt_lag']) / Compustat['revt_lag'])

# Calculate Net Income Growth
Compustat = Compustat.withColumn("ni_lag", F.lag(Compustat['ni']).over(Window.partitionBy("gvkey").orderBy("datadate")))
Compustat = Compustat.withColumn("ni_growth", (Compustat['ni'] - Compustat['ni_lag']) / Compustat['ni_lag'])

# Calculate Current Ratio
Compustat = Compustat.withColumn("current_ratio", Compustat.act / Compustat.lct)

# Calculate R&D Growth
Compustat = Compustat.withColumn("xrd_lag", F.lag(Compustat.xrd).over(Window.partitionBy("gvkey").orderBy("datadate")))
Compustat = Compustat.withColumn("xrd_growth", (Compustat.xrd - Compustat.xrd_lag) / Compustat.xrd_lag)
Compustat = Compustat.withColumn("xrd_growth_lag1", F.lag(Compustat.xrd_growth).over(Window.partitionBy("gvkey").orderBy("datadate")))
Compustat = Compustat.withColumn("xrd_growth_lag2", F.lag(Compustat.xrd_growth_lag1).over(Window.partitionBy("gvkey").orderBy("datadate")))

#%%
######################################################################################################
#                                           Merge CRSP and Compustat
######################################################################################################

#stock_monthly['PERMNO', 'date', 'fdate', 'fyear', 'Momentum_12', 'Reversal', 'Size'].show(150)
sqlContext.registerDataFrameAsTable(stock_monthly, "st")
sqlContext.registerDataFrameAsTable(Compustat, "comp")
query = """
SELECT st.* , comp.*
FROM st LEFT JOIN comp
ON st.PERMCO = comp.LPERMCO and st.fyear = comp.fyear
"""
CRSP_Compustat = sqlContext.sql(query)
CRSP_Compustat = CRSP_Compustat.orderBy("PERMNO", "date")
#CRSP_Compustat.count() #3,391,113

# Create Value factor
CRSP_Compustat = CRSP_Compustat.withColumn("Value", CRSP_Compustat.be / CRSP_Compustat.Size)
#CRSP_Compustat['date', 'PERMNO', 'RET', 'Reversal', "Momentum_12", "she", "be", "ME", "Value"].show(100)
#CRSP_Compustat.printSchema()

## Examine the null values
#CRSP_Compustat.select([F.count(F.when(F.isnull(c), c)).alias(c) for c in ['be', 'ME', 'Value', 'PERMNO', 'she']]).show()
#
## Examine the CRSP_Compustat Dataset
#Test_CRSPCompustat = CRSP_Compustat.filter(CRSP_Compustat.PERMNO == 10137)['PERMNO', 'date', 'PERMCO', 'PRC', 'RET', 'VOL', 'ME', 'fdate', 'year', 'month', 'st.fyear', 'DVOL', 'Momentum_12', 'Beta', 'Size', 'at', 'ci', 'Value'].toPandas()


#%%
########################################################################################################
#                            Other Fundamental Valuations Factors - Building
########################################################################################################

# Build signal earnings-to-price (ep)
CRSP_Compustat = CRSP_Compustat.withColumn("ep", CRSP_Compustat.ni / CRSP_Compustat.PRC_Lag)

# Build signal sales-to-price (sp)
CRSP_Compustat = CRSP_Compustat.withColumn("sp", CRSP_Compustat.revt / CRSP_Compustat.PRC_Lag)



#%%
########################################################################################
#                              Merge with Macroeconomic Variables
########################################################################################



#%%
########################################################################################
#                   Final Examination of the CRSP_Compustat Data
########################################################################################

CRSP_Compustat.count() #3,391,113
CRSP_Compustat.printSchema()
CRSP_Compustat.select('date', 'PRC', 'RET', 'Ex_RET').describe().show()
look_list = ['PERMNO', 'date', 'EXCHCD', 'SICCD', 'PERMCO', 'PRC', 'RET', 'Ex_RET', 'VOL', 'fdate', 'st.fyear', 'SICCD_Industry', 'DVOL', 'Momentum_12',
             'Momentum_6', 'Reversal', 'Mkt_RF', 'Beta', 'Beta_Lag', 'ME', 'Size', 'Value', 'total_mktcap', 'weight_mktcap', 'at', 'lt', 'ep', 'sp', 'datadate']
Test_CRSP_Compustat = CRSP_Compustat.filter((CRSP_Compustat['PERMNO'] == 10006) | (CRSP_Compustat['PERMNO'] == 10007) | (CRSP_Compustat['PERMNO'] == 10008)).select(look_list).toPandas()


#%%
########################################################################################
#                              Prepare dataset for the models
########################################################################################

# Build the training dataset
CRSP_Compustat_train = CRSP_Compustat.filter(CRSP_Compustat.date <= train_date) 
#CRSP_Compustat_train.count() # 472,471
#CRSP_Compustat_train.select([F.count(F.when(F.isnull(c), c)).alias(c) for c in CRSP_Compustat_train.columns]).show() # count null values in each column

# Build the validation dataset
CRSP_Compustat_validation = CRSP_Compustat.filter((CRSP_Compustat.date > train_date) & (CRSP_Compustat.date <= validation_date)) #740,374
#CRSP_Compustat_validation.count() #769,393

# Build the out-sample dataset
CRSP_Compustat_out = CRSP_Compustat.filter((CRSP_Compustat.date > validation_date))
#CRSP_Compustat_out.count() #2,213,330

# Build the recent dataset
CRSP_Compustat_recent = CRSP_Compustat.filter((CRSP_Compustat.date > recent_date))
#CRSP_Compustat_recent.count() #375,947

################################## Standard OLS Dataset #################################

# Build the vector Assembler
vectorAssembler = VectorAssembler(inputCols = ["Momentum_12", "Log_Size", "Value"], outputCol = "features")

# Select depdendent variables
select_variables_x = ['Momentum_12', 'Log_Size', 'Value']
select_variables = copy(select_variables_x)
select_variables = select_variables + ['RET', 'PERMNO', 'date', 'year', 'month', 'Size', 'total_mktcap', 'weight_mktcap']

# build the training dataset
train_data = CRSP_Compustat_train.select(select_variables)
#train_data.select([F.count(F.when(F.isnull(c), c)).alias(c) for c in train_data.columns]).show()
train_data = train_data.na.drop()
#train_data.count() #261,074
train_data = vectorAssembler.transform(train_data)
#train_data = train_data.select(['features', 'RET'])
#train_data.show(5)

# build the validation dataset
validation_data = CRSP_Compustat_validation.select(select_variables)
#validation_data.select([F.count(F.when(F.isnull(c), c)).alias(c) for c in validation_data.columns]).show()
validation_data = validation_data.na.drop()
#validation_data.count() #548,858
#validation_data.show(5)
validation_data = vectorAssembler.transform(validation_data)
#validation_data = validation_data.select(['features', 'RET'])
#validation_data.count()

# build the out-sample dataset
outsample_data = CRSP_Compustat_out.select(select_variables)
outsample_data = outsample_data.na.drop()
#outsample_data.count() #1,685,828
outsample_data = vectorAssembler.transform(outsample_data)

# build the recent data
recent_data = CRSP_Compustat_recent.select(select_variables)
recent_data = recent_data.na.drop()
#recent_data_ml.count() #295,672
recent_data = vectorAssembler.transform(recent_data)

################################## Machine Learning Dataset ##################################

# Select depdendent variables
select_variables_x = ['Momentum_12', 'Momentum_Count', 'Momentum_6', "Momentum_36", 'Reversal', 
                      'Beta_Lag', 'Beta_Shrink_Lag', 'Beta_Lag_Square', 'Beta_Shrink_Lag_Square',
                      'Size', 'Log_Size', 'revt', 'be', 'Value', 'DVOL', 'TURN', 'ep', 'sp',
                      'at_growth', 'ni_growth', 'revt_growth', 'current_ratio']
                      #'quick_ratio', Industry

select_variables = copy(select_variables_x)
select_variables = select_variables + ['RET', 'PERMNO', 'date', 'year', 'month', 'Size', 'total_mktcap', 'weight_mktcap']

# Define Vector Assembler_ml
vectorAssembler_ml = VectorAssembler(inputCols = select_variables_x, outputCol = "features")

# Define the Performance Evaluator
ml_evaluator_r2 = RegressionEvaluator(labelCol = "RET", predictionCol = "prediction", metricName = "r2")

# build the training dataset - machine learning
train_data_ml = CRSP_Compustat_train.select(select_variables)
#missing_value = train_data_ml.select([F.count(F.when(F.isnull(c), c)).alias(c) for c in train_data_ml.columns]).toPandas()
train_data_ml = train_data_ml.na.drop()
#train_data_ml.count() #219,216
train_data_ml = vectorAssembler_ml.transform(train_data_ml)
#train_data_ml = train_data_ml.select(['features', 'RET'])
#train_data_ml.show(4)

# build the validation dataset - machine learning
validation_data_ml = CRSP_Compustat_validation.select(select_variables)
validation_data_ml = validation_data_ml.na.drop()
#validation_data_ml.count() #403,158
validation_data_ml = vectorAssembler_ml.transform(validation_data_ml)
#validation_data_ml = validation_data_ml.select(['features', 'RET'])
#validation_data_ml.show(5)

# build the out-sample dataset - machine learning
outsample_data_ml = CRSP_Compustat_out.select(select_variables)
outsample_data_ml = outsample_data_ml.na.drop()
#outsample_data_ml.count() #1,577,816
outsample_data_ml = vectorAssembler_ml.transform(outsample_data_ml)

# build the recent dataset - machine learning
recent_data_ml = CRSP_Compustat_recent.select(select_variables)
recent_data_ml = recent_data_ml.na.drop()
#recent_data_ml.count() #295,672
recent_data_ml = vectorAssembler_ml.transform(recent_data_ml)

#%%
########################################################################################
#                          Simple OLS with Size, Value, and Momentum Model
########################################################################################

# Train the model
lr = LinearRegression(featuresCol = 'features', labelCol = "RET")
lr_model = lr.fit(train_data)

# Examine the model - in sample
lr_model.coefficients
lr_model.intercept
print("RMSE: %f" % lr_model.summary.rootMeanSquaredError)
print("R^2: %f" % lr_model.summary.r2)

# Examine the model
lr_predictions_val = lr_model.transform(validation_data)
lr_predictions_out = lr_model.transform(outsample_data)
lr_predictions_recent = lr_model.transform(recent_data)
#lr_predictions_val.select("prediction","RET","features").show(50)
lr_evaluator = RegressionEvaluator(predictionCol = "prediction", labelCol = "RET", metricName = "r2")
print("R Square on test data = %g" % lr_evaluator.evaluate(lr_predictions_val))

## Test with Statsmodel Pacakge
#AAA = CRSP_Compustat_train.toPandas()
#x_variable = AAA[['Momentum_12', 'Log_Size', 'Value']]
#x_variable = sm.add_constant(x_variable, prepend = False)
#y_variable = AAA[['RET']]
#mod = sm.OLS(y_variable, x_variable)
#res = mod.fit()
#print(res.summary())

#%%
########################################################################################
#                   Penalized OLS with Size, Value, and Momentum Model
########################################################################################

# Train the model
lr_reg = LinearRegression(featuresCol = 'features', labelCol = "RET", regParam = 0.3)
lr_reg_model = lr_reg.fit(train_data)

# Examine the model - in sample
lr_reg_model.coefficients
lr_reg_model.intercept
print("RMSE: %f" % lr_reg_model.summary.rootMeanSquaredError)
print("R Square: %f" % lr_reg_model.summary.r2)

# Examine the model - out sample
lr_reg_prediction = lr_reg_model.transform(validation_data)
lr_reg_evaluator = RegressionEvaluator(predictionCol = "prediction", labelCol = "RET", metricName = "r2")
print("R Square on test data = %g" % lr_reg_evaluator.evaluate(lr_reg_prediction))

# Grid Search on the hyper-parameters
plr_r2 = []
plr_rmse = []
niter = 500
elastic_reg = 0.0
tolerance = 1e-5
regs = [1e-4, 1e-2, 1.0, 1.e1] #control the penalty
intercept = True
evaluator = RegressionEvaluator(predictionCol = "prediction", labelCol = "RET")
for reg in regs:
    # Build the model
    model = LinearRegression(featuresCol = 'features',
                             labelCol = "RET",
                             maxIter = niter,
                             regParam = reg,
                             elasticNetParam = elastic_reg).fit(train_data)
    # Build the predictions
    prediction = model.transform(validation_data)
    # Document the model accuracy
    plr_r2.append(evaluator.evaluate(prediction, {evaluator.metricName: "r2"}))
    plr_rmse.append(evaluator.evaluate(prediction, {evaluator.metricName: "rmse"}))
    

#%%
########################################################################################
#                             Generalized Linear Regression Model
########################################################################################

# Sparse Model: Using only signals of momentum, size, and value
glr = GeneralizedLinearRegression(featuresCol = 'features', labelCol = "RET", family = "gaussian", link = "identity", maxIter = 10, regParam = 0.3)
glr_model = glr.fit(train_data)

# Examine the model - in sample
glr_model.coefficients
glr_model.intercept

# Examine the model - out sample
glr_model_predictions = glr_model.transform(validation_data)
glr_model_evaluator = RegressionEvaluator(predictionCol = "prediction", labelCol = "RET", metricName = "r2")
print("R Square on test data = %g" % glr_model_evaluator.evaluate(glr_model_predictions))

# Grid Search on the hyper-parameters
families = ['gaussian']
niter = 10
reg = 0.5
glr_r2 = []
glr_rmse = []
for option in families:
    # Build the model
    model = GeneralizedLinearRegression(featuresCol = 'features',
                                        labelCol = "RET",
                                        family = option, 
                                        link = 'identity',
                                        maxIter = niter,
                                        regParam = reg).fit(train_data)
    # Build the predictions
    prediction = model.transform(validation_data)
    # Document the model accuracy
    glr_r2.append(evaluator.evaluate(prediction, {evaluator.metricName: "r2"}))
    glr_rmse.append(evaluator.evaluate(prediction, {evaluator.metricName: "rmse"}))


#%%
########################################################################################
#                                   Decision Tree Regression
########################################################################################

# Sparse Model: Using only signals of momentum, size, and value
dt = DecisionTreeRegressor(featuresCol = "features", labelCol = "RET")
dt_model = dt.fit(train_data)
# Examine the model - out sample
dt_predictions = dt_model.transform(validation_data)
dt_evaluator = RegressionEvaluator(labelCol = "RET", predictionCol = "prediction", metricName = "r2")
dt_evaluator2 = RegressionEvaluator(labelCol = "RET", predictionCol = "prediction", metricName = "rmse")
print("R Square on test data = %g" % dt_evaluator.evaluate(dt_predictions))
print("RMSE on test data = %g" % dt_evaluator2.evaluate(dt_predictions))

# Sophisticated machine learning model
dt_ml_model = dt.fit(train_data_ml)
# Examine the model - out sample
dt_predictions_ml = dt_ml_model.transform(validation_data_ml)
dt_ml_evaluator = RegressionEvaluator(labelCol = "RET", predictionCol = "prediction", metricName = "r2")
print("R Square on test data = %g" % dt_ml_evaluator.evaluate(dt_predictions_ml))


#%%
########################################################################################
#                Gradient-Boosted Tree Regressor (Boosting method of Trees)
########################################################################################

## Sparse Model: Using only signals of momentum, size, and value
#gbt = GBTRegressor(featuresCol = 'features', labelCol = "RET", maxIter = 15)
#gbt_model = gbt.fit(train_data)
#gbt_predictions = gbt_model.transform(validation_data)
#gbt_evaluator = RegressionEvaluator(labelCol = "RET", predictionCol = "prediction", metricName = "r2")
#gbt_evaluator2 = RegressionEvaluator(labelCol = "RET", predictionCol = "prediction", metricName = "rmse")
#print("R Square on test data = %g" % gbt_evaluator.evaluate(gbt_predictions))
#print("RMSE on test data = %g" % gbt_evaluator2.evaluate(gbt_predictions))

# Sophisticated machine learning model
gbt = GBTRegressor(featuresCol = 'features', labelCol = "RET", maxIter = 15)
gbt_ml_model = gbt.fit(train_data_ml)
gbt_ml_model.featureImportances
gbt_ml_predictions_val = gbt_ml_model.transform(validation_data_ml)
gbt_ml_predictions_out = gbt_ml_model.transform(outsample_data_ml)
gbt_ml_predictions_recent = gbt_ml_model.transform(recent_data_ml)

# Examine the model - out sample
print("R Square on validation data = %g" % ml_evaluator_r2.evaluate(gbt_ml_predictions_val))



#%%
########################################################################################
#                        Random Forest (Bagging method of Trees)
########################################################################################

#################### Sophisticated machine learning model ####################

# Build the model and make predictions
rf = RandomForestRegressor(featuresCol = "features", labelCol = "RET")
rf_ml_model = rf.fit(train_data_ml)
rf_ml_model.featureImportances
rf_predictions_ml = rf_ml_model.transform(validation_data_ml)
rf_ml_pred_out = rf_ml_model.transform(outsample_data_ml)
rf_ml_pred_recent = rf_ml_model.transform(recent_data_ml)

# Examine the model 
print("R Square on validation dataset = %g" % ml_evaluator_r2.evaluate(rf_predictions_ml)) #validation sample
print("R Square on outsample data = %g" % ml_evaluator_r2.evaluate(rf_ml_pred_out)) #outsample dataset
print("R Square on recent sample data = %g" % ml_evaluator_r2.evaluate(rf_ml_pred_recent)) #recent dataset


#%%
##################################################################################################
#                                       Benchmark Portfolios
##################################################################################################

# Define Cumulative Portfolio Function
def make_cumulative(dataframe2):
    dataframe = copy(dataframe2)
    dataframe += 1
    for i in range(10):
        string = "Group" + str(i + 1)
        dataframe[string] = dataframe.iloc[:, i].cumprod()
    dataframe = dataframe.iloc[:,10:]
    return dataframe

# Build Portfolios
def make_portfolio(df):
    
    # Equal-Weighted and Value-Weighted Portfolio
    df = df.withColumn("RET_weight", df.RET * df.weight_mktcap)
    df_portfolio = df.groupBy("Group", "year", "month").agg(F.avg(df['Ex_RET']), F.sum(df['RET_weight']), F.sum(df['Size']) / F.avg(df['total_mktcap'])).toPandas()
    df_portfolio['RET'] = df_portfolio['sum(RET_weight)'] / df_portfolio['(sum(Size) / avg(total_mktcap))']
    df_portfolio_equal = df_portfolio.pivot_table(index = ['year', 'month'], columns = 'Group', values = 'avg(Ex_RET)')
    df_portfolio_equal_recent = copy(df_portfolio_equal.iloc[634:, :])
    df_portfolio_value = df_portfolio.pivot_table(index = ['year', 'month'], columns = 'Group', values = "RET")
    df_portfolio_recent_value = copy(df_portfolio_value.iloc[634:, :])
    df_portfolio_out_equal = copy(df_portfolio_equal.iloc[357:, :])
    df_portfolio_out_value = copy(df_portfolio_value.iloc[357:, :])
    
    # Put into dictionary
    final_result = dict()
    final_result['full_portfolio_equal'] = df_portfolio_equal
    final_result['recent_portfolio_equal'] = df_portfolio_equal_recent
    final_result['full_portfolio_equal_cum'] = make_cumulative(df_portfolio_equal)
    final_result['recent_portfolio_equal_cum'] = make_cumulative(df_portfolio_equal_recent)
    final_result['full_portfolio_value'] = df_portfolio_value
    final_result['recent_portfolio_value'] = df_portfolio_recent_value
    final_result['full_portfolio_value_cum'] = make_cumulative(df_portfolio_value)
    final_result['recent_portfolio_value_cum'] = make_cumulative(df_portfolio_recent_value)
    final_result['out_portfolio_equal'] = df_portfolio_out_equal
    final_result['out_portfolio_value'] = df_portfolio_out_value
    final_result['out_portfolio_equal_cum'] = make_cumulative(df_portfolio_out_equal)
    final_result['out_portfolio_value_cum'] = make_cumulative(df_portfolio_out_value)
    return(final_result)

# Market Portfolio - Based on Fama French
ff_mkt_portfolio = dict()
ff_mkt_benchmark = copy(ff_factors_df[['Date', 'year', 'month', 'Mkt_RF', 'RF', 'Mkt']])
ff_mkt_benchmark['Mkt_Plus1'] = 1 + ff_mkt_benchmark['Mkt']
ff_mkt_benchmark_recent = copy(ff_mkt_benchmark.iloc[634:, :])
ff_mkt_benchmark['Cumulative'] = ff_mkt_benchmark['Mkt_Plus1'].cumprod()
ff_mkt_benchmark_recent['Cumulative'] = ff_mkt_benchmark_recent['Mkt_Plus1'].cumprod()
ff_mkt_portfolio['full_portfolio'] = ff_mkt_benchmark
ff_mkt_portfolio['recent_portfolio'] = ff_mkt_benchmark_recent

# Market Portfolio
benchmark_mkt = CRSP_Compustat.select("PERMNO", "date", 'year', 'month', 'PRC', 'Ex_RET', 'RET', 'weight_mktcap')
# Equal-weighted
benchmark_mkt_equal = benchmark_mkt.groupBy("year", "month").agg({"RET": "mean"}).toPandas()
benchmark_mkt_equal['RET_Plus1'] = benchmark_mkt_equal['avg(RET)'] + 1
benchmark_mkt_equal = benchmark_mkt_equal.sort_values(["year", "month"])
benchmark_mkt_equal_recent = copy(benchmark_mkt_equal.iloc[634:, :])
benchmark_mkt_equal['Cumulative'] = benchmark_mkt_equal['RET_Plus1'].cumprod()
benchmark_mkt_equal_recent['Cumulative'] = benchmark_mkt_equal_recent['RET_Plus1'].cumprod()
# Value-weighted
benchmark_mkt_value = benchmark_mkt.withColumn("RET_weight", benchmark_mkt.RET * benchmark_mkt.weight_mktcap)
benchmark_mkt_value = benchmark_mkt_value.groupBy("year", "month").agg({"RET_weight": "sum"}).toPandas()
benchmark_mkt_value['RET_Plus1'] = benchmark_mkt_value['sum(RET_weight)'] + 1
benchmark_mkt_value = benchmark_mkt_value.sort_values(['year', 'month'])
benchmark_mkt_value_recent = copy(benchmark_mkt_value.iloc[634:, :])
benchmark_mkt_value['Cumulative'] = benchmark_mkt_value['RET_Plus1'].cumprod()
benchmark_mkt_value_recent['Cumulative'] = benchmark_mkt_value_recent['RET_Plus1'].cumprod()
# Put into Dictionary
mkt_portfolio = dict()
mkt_portfolio['full_portfolio_equal'] = benchmark_mkt_equal
mkt_portfolio['recent_portfolio_equal'] = benchmark_mkt_equal_recent
mkt_portfolio['full_portfolio_value'] = benchmark_mkt_value
mkt_portfolio['recent_portfolio_value'] = benchmark_mkt_value_recent

# Momentum Benchmark Portfolio
benchmark_mom12 = CRSP_Compustat.select('PERMNO', 'date', 'year', 'month', 'PRC', 'Ex_RET','RET', 'Momentum_12', "Size", "total_mktcap", "weight_mktcap")
sqlContext.registerDataFrameAsTable(benchmark_mom12, "mom12")
query = """
SELECT *, ntile(10) OVER (PARTITION BY date ORDER BY Momentum_12) AS Group
FROM mom12
"""
benchmark_mom12 = sqlContext.sql(query)
momentum_portfolio = make_portfolio(benchmark_mom12)

# Size Benchmark Portfolio
benchmark_size = CRSP_Compustat.select('PERMNO', 'date', 'year', 'month', 'PRC', 'Ex_RET', 'RET', 'Size', 'total_mktcap', 'weight_mktcap')
sqlContext.registerDataFrameAsTable(benchmark_size, "size")
query = """
SELECT *, ntile(10) OVER (PARTITION BY date ORDER BY Size) AS Group
FROM size
"""
benchmark_size = sqlContext.sql(query)
size_portfolio = make_portfolio(benchmark_size)

# Value Benchmark Portfolio
benchmark_value = CRSP_Compustat.select('PERMNO', 'date', 'year', 'month', 'PRC', 'Ex_RET', 'RET', 'Value', 'Size', 'total_mktcap', 'weight_mktcap')
sqlContext.registerDataFrameAsTable(benchmark_value, "value")
query = """
SELECT *, ntile(10) OVER (PARTITION BY date ORDER BY Value) AS Group
FROM value
"""
benchmark_value = sqlContext.sql(query)
value_portfolio = make_portfolio(benchmark_value)


#%%
##################################################################################################
#                             Portfolio Construction and Back-test
##################################################################################################

# Define the back-test function
def back_test(df):
    '''
    Define function for Cumulative Return of Portfolios
    '''
    # Make cumulative calculation function
    def make_cumulative(dataframe):
        dataframe += 1
        for i in range(10):
            string = "Group" + str(i + 1)
            dataframe[string] = dataframe.iloc[:, i].cumprod()
        dataframe = dataframe.iloc[:,10:]
        return dataframe
    
    #Build portfolios
    sqlContext.registerDataFrameAsTable(df, "rf")
    query = """
    SELECT * , ntile(10) OVER (PARTITION BY date ORDER BY prediction) AS Group
    FROM rf
    """
    df = sqlContext.sql(query)
    
    # Prediction Group
    portfolios_prediction = df.groupBy('date', 'Group').avg("prediction").toPandas()
    portfolios_prediction = portfolios_prediction.pivot_table(index = 'date', columns = 'Group', values = 'avg(prediction)')
    
    # Equal-Weighted and Value-Weighted Portfolios
    df = df.withColumn("RET_weight", df.RET * df.weight_mktcap)
    df_portfolio = df.groupBy("Group", "year", "month").agg(F.avg(df['RET']), F.sum(df['RET_weight']), F.sum(df['Size']) / F.avg(df['total_mktcap'])).toPandas()
    df_portfolio['RET_value'] = df_portfolio['sum(RET_weight)'] / df_portfolio['(sum(Size) / avg(total_mktcap))']
    df_portfolio_equal = df_portfolio.pivot_table(index = ['year', 'month'], columns = 'Group', values = 'avg(RET)')
    df_portfolio_value = df_portfolio.pivot_table(index = ['year', 'month'], columns = 'Group', values = "RET_value")
    
    # Result Dict
    result = dict()
    result['prediction'] = portfolios_prediction
    result['equal_return'] = df_portfolio_equal
    result['value_return'] = df_portfolio_value
    result['equal_return_cum'] = make_cumulative(copy(df_portfolio_equal))
    result['value_return_cum'] = make_cumulative(copy(df_portfolio_value))
    return result

# OLS with Momentum, Value, and Size Factors
lr_ols_result_val = back_test(lr_predictions_val)
lr_ols_result_out = back_test(lr_predictions_out)
lr_ols_result_recent = back_test(lr_predictions_recent)

# Random Forest Portfolio Backtest
rf_ml_result_val = back_test(rf_predictions_ml) # validation dataset
rf_ml_result_out = back_test(rf_ml_pred_out) # outsample dataset
rf_ml_result_recent = back_test(rf_ml_pred_recent) #recent sample dataset

# Gradient-Boosted Tree Regressions
gbt_ml_result_val = back_test(gbt_ml_predictions_val)
gbt_ml_result_out = back_test(gbt_ml_predictions_out)
gbt_ml_result_recent = back_test(gbt_ml_predictions_recent)



#%%
##################################################################################################
#                             Visualization and Investment Graph
##################################################################################################

# Create Portfolio Return, Volatility, and Sharpe
def return_vol_sharpe(df, name, time_period):
    pic = plt.figure()
    bar_width = 0.3
    bar1 = plt.bar(np.arange(1, 11), df.mean(axis = 0) * 12, bar_width, alpha = 0.8, color = "darkblue", label = "return")
    bar2 = plt.bar(np.arange(1, 11) + bar_width, df.std(axis = 0) * math.sqrt(12), bar_width, alpha = 0.8, color = "darkgreen", label = "volatility")
    bar3 = plt.bar(np.arange(1, 11) + bar_width*2, df.mean(axis = 0) * 12 / (df.std(axis = 0) * math.sqrt(12)), bar_width, alpha = 0.8, color = "orange", label = "Sharpe")
    plt.xlabel(name + " Group"); plt.ylabel("Annualized Return and Volatility"); plt.title(name + " Portfolio Group Performance " + time_period)
    plt.legend()
    plt.tight_layout()
    return pic

def return_vol_sharpe2(df, name, time_period):
    pic = plt.figure()
    bar_width = 0.3
    bar1 = plt.bar(['Momentum', 'Size', 'Value', 'RF', 'GBT'], df.mean(axis = 0) * 12, bar_width, alpha = 0.8, color = "darkblue", label = "return")
    bar2 = plt.bar(np.arange(0, 5) + bar_width, df.std(axis = 0) * math.sqrt(12), bar_width, alpha = 0.8, color = "darkgreen", label = "volatility")
    bar3 = plt.bar(np.arange(0, 5) + bar_width*2, df.mean(axis = 0) * 12 / (df.std(axis = 0) * math.sqrt(12)), bar_width, alpha = 0.8, color = "orange", label = "Sharpe")
    plt.xlabel(name + " Group"); plt.ylabel("Annualized Return and Volatility"); plt.title(name + " Portfolio Group Performance " + time_period)
    plt.legend()
    plt.tight_layout()
    return pic

# Out-Period Performance Evaluation
# Performance Statistics Charts
momentum_portfolio['out_portfolio_value_statistics_chart'] = return_vol_sharpe(momentum_portfolio['out_portfolio_value'], "Momentum", "1987 - 2016")
size_portfolio['out_portfolio_value_statistics_chart'] = return_vol_sharpe(size_portfolio['out_portfolio_value'], "Size", "1987 - 2016")
value_portfolio['out_portfolio_value_statistics_chart'] = return_vol_sharpe(value_portfolio['out_portfolio_value'], "Value", "1987 - 2016")
rf_ml_result_out['out_portfolio_value_statistics_chart'] = return_vol_sharpe(rf_ml_result_out['value_return'], "RF_Portfolio", "1987 - 2016")
gbt_ml_result_out['out_portfolio_value_statistics_chart'] = return_vol_sharpe(gbt_ml_result_out['value_return'], "GBT_Portfolio", "1987 - 2016")

# Cumulative Perforamce Chart
momentum_portfolio['out_portfolio_value_cum'].plot()
size_portfolio['out_portfolio_value_cum'].plot()
value_portfolio['out_portfolio_value_cum'].plot()
rf_ml_result_out['value_return_cum'].plot()
gbt_ml_result_out['value_return_cum'].plot()

# Joint Best Portfolios
# Portfolio Return
join_best_portfolios = pd.concat([momentum_portfolio['out_portfolio_value'].iloc[:, 9], size_portfolio['out_portfolio_value'].iloc[:, 0], 
                                  value_portfolio['out_portfolio_value'].iloc[:, 0], rf_ml_result_out['value_return'].iloc[:, 9], 
                                  gbt_ml_result_out['value_return'].iloc[:, 9]],
                                  axis = 1)
join_best_portfolios.columns = ['Momentum', 'Size', 'Value', 'RandomForest', 'GradientBoostTree']
return_vol_sharpe2(join_best_portfolios, "Portfolio Comparison", "1987 - 2016")
# Portfolio Cumulative Return
join_best_portfolios_cum = pd.concat([momentum_portfolio['out_portfolio_value_cum'].iloc[:, 9], size_portfolio['out_portfolio_value_cum'].iloc[:, 0], 
                                  value_portfolio['out_portfolio_value_cum'].iloc[:, 0], rf_ml_result_out['value_return_cum'].iloc[:, 9], 
                                  gbt_ml_result_out['value_return_cum'].iloc[:, 9]],
                                  axis = 1)
join_best_portfolios_cum.columns = ['Momentum', 'Size', 'Value', 'RandomForest', 'GradientBoostTree']
join_best_portfolios_cum.plot()

# Compare the time series of returns
x = np.arange(994)
ts1 = portfolio_return.iloc[:, 0]
ts2 = portfolio_return.iloc[:, 9:10]
plt.subplot(1, 2, 1)
plt.plot(x, ts1.values)
plt.ylim((-50, 70)); plt.xlabel("time"); plt.ylabel("return");plt.title("Group1")
plt.subplot(1, 2, 2)
plt.plot(x, ts2.values)
plt.ylim((-50, 70)); plt.xlabel("time"); plt.ylabel("return");plt.title("Group10")
# Plot the cumulative return
cumulative_return.plot()
plt.ylabel('Cumulative Return')
plt.xlabel('Year')
plt.title('Cumulative Return Group Sorted by Beta')
plt.show()
# Compare the sub_period portfolio returns time series plot
# Note: would rather print it seperately and join them manually
# Plot them together
fig, axes = plt.subplots(nrows=2, ncols=2)
cumulative_return_sub1.plot(ax=axes[0,0])
cumulative_return_sub2.plot(ax = axes[0, 1])
cumulative_return_sub3.plot(ax = axes[1, 0])
cumulative_return_sub4.plot(ax = axes[1, 1])
# Plot them seperately
cumulative_return_sub4.plot()
cumulative_return_sub3.plot()
cumulative_return_sub2.plot()
cumulative_return_sub1.plot()
# Compare the sub_period cumulative portfolio returns
bar_width = 0.4
plt.subplot(2, 2, 1)
plt.bar(np.arange(1, 11), cumulative_return_sub1.iloc[-1], bar_width, alpha = 0.8, color = "orange", label = "return")
plt.xticks(np.arange(1, 11), avg_group_return['Beta_Group'])
plt.xlabel("Beta_Group"); plt.ylabel("Return"); plt.title("1930 - 1950 Cumulative Return")
plt.subplot(2, 2, 2)
plt.bar(np.arange(1, 11), cumulative_return_sub2.iloc[-1], bar_width, alpha = 0.8, color = "blue", label = "return")
plt.xticks(np.arange(1, 11), avg_group_return['Beta_Group'])
plt.xlabel("Beta_Group"); plt.ylabel("Return"); plt.title("1950 - 1970 Cumulative Return")
plt.subplot(2, 2, 3)
plt.bar(np.arange(1, 11), cumulative_return_sub3.iloc[-1], bar_width, alpha = 0.8, color = "green", label = "return")
plt.xticks(np.arange(1, 11), avg_group_return['Beta_Group'])
plt.xlabel("Beta_Group"); plt.ylabel("Return"); plt.title("1970 - 1990 Cumulative Return")
plt.subplot(2, 2, 4)
plt.bar(np.arange(1, 11), cumulative_return_sub4.iloc[-1], bar_width, alpha = 0.8, color = "red", label = "return")
plt.xticks(np.arange(1, 11), avg_group_return['Beta_Group'])
plt.xlabel("Beta_Group"); plt.ylabel("Return"); plt.title("1990 - 2010 Cumulative Return")






