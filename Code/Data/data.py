import pandas as pd
import numpy as np
import yfinance as yf
import os
import argparse
from datetime import datetime
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
# grid search hyperparameters for lasso regression
from numpy import arange
from scipy import stats
from pandas import read_csv
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def my_log(x):
    return np.log(x.clip(lower=0.01))
def my_sqrt(x):
    return np.sqrt(x.clip(lower=0.01))
def parse_arguments():
    parser = argparse.ArgumentParser(description="langchain prompting")
    parser.add_argument("--start_date", type=str, default="2013-01-31", help="startdate")
    parser.add_argument("--path_to_file", type=str, default="./", help="startdate")
    parser.add_argument("--end_date", type=str, default="2017-12-31", help="enddate")
    parser.add_argument("--feature_names", type=str, default="Price/Earnings,Price/Book Value,Return on Assets,Return on Equity ,\
Free Cash Flow per Share,Price/Cash Flow,Dividend Yield (%),Enterprise Value/EBIT,Enterprise Value/EBITDA,Dividend Payout Ratio (%)", help="feature_names")
#     "Price/Earnings,Price/Book Value,Return on Assets,Return on Equity ,\
# Free Cash Flow per Share,Price/Cash Flow,Dividend Yield (%),Enterprise Value/EBIT,Enterprise Value/EBITDA,Dividend Payout Ratio (%)"
    parser.add_argument("--new_feature", type=str, default="012345678", help="newfeature")
    parser.add_argument("--y", type=str, default="Return", help="return")


    args = parser.parse_args()
    return args
#todo
dates = [
         '2012-12-31', 
         '2013-01-31', '2013-02-28', '2013-03-31', '2013-04-30', '2013-05-31', '2013-06-30', '2013-07-31',
         '2013-08-31', '2013-09-30', '2013-10-31', '2013-11-30', 
         '2013-12-31',
         '2014-01-31', '2014-02-28', '2014-03-31', '2014-04-30', '2014-05-31', '2014-06-30', '2014-07-31',
         '2014-08-31', '2014-09-30', '2014-10-31', '2014-11-30', 
         '2014-12-31',
        '2015-01-31', '2015-02-28', '2015-03-31', '2015-04-30', '2015-05-31', '2015-06-30', '2015-07-31',
         '2015-08-31', '2015-09-30', '2015-10-31', '2015-11-30', 
         '2015-12-31',
         '2016-01-31', '2016-02-29', '2016-03-31', '2016-04-30', '2016-05-31', '2016-06-30', '2016-07-31',
         '2016-08-31', '2016-09-30', '2016-10-31', '2016-11-30', 
         '2016-12-31',
         '2017-01-31', '2017-02-28', '2017-03-31', '2017-04-30', '2017-05-31', '2017-06-30', '2017-07-31',
         '2017-08-31', '2017-09-30', '2017-10-31', '2017-11-30', 
         '2017-12-31'
         ]
def get_historical_returns(ticker, start_date, end_date, frequency="monthly"):
    'Function to fetch Historical Price data and compute returns'
    if os.path.exists("historical_return/"+ticker+start_date+"-"+end_date+".csv"):
        data = pd.read_csv("historical_return/"+ticker+start_date+"-"+end_date+".csv")
        monthly_data = data.copy()
        monthly_data['Return'] = monthly_data['Close']
        monthly_data['Date']=pd.to_datetime(monthly_data['Date'])
        monthly_data = monthly_data.set_index('Date')
    else:
        data = yf.download(ticker,start=start_date, end=end_date)
        data.to_csv("historical_return/"+ticker+start_date+"-"+end_date+".csv")
        monthly_data = data.copy()
        monthly_data['Return'] = monthly_data['Close']
        
    # Calculate Daily Returns
    # daily_data = data.copy()
    # #print("daily_data:", daily_data)
    # daily_data['Return'] = daily_data['Close'].pct_change()
    # daily_returns = daily_data[['Return']].dropna()
    # Calculate Monthly Returns

    monthly_data = monthly_data['Return'].resample('M').last()
    #print("monthly data after:", monthly_data)
    monthly_returns = monthly_data.pct_change()
    monthly_returns = monthly_returns.dropna()

    if frequency == "daily": return daily_returns
    if frequency == "monthly": return monthly_returns
    return monthly_data

def get_historical_returns_by_day(ticker, start_date, end_date, frequency="monthly"):
    'Function to fetch Historical Price data and compute returns'
    if os.path.exists("historical_return/"+ticker+start_date+"-"+end_date+".csv"):
        data = pd.read_csv("historical_return/"+ticker+start_date+"-"+end_date+".csv")
    else:
        data = yf.download(ticker,start=start_date, end=end_date)
        data.to_csv("historical_return/"+ticker+start_date+"-"+end_date+".csv")

    #print("This ticker is:", ticker)
    #print(data)
    first_close = data['Close'].iloc[0]
    last_close = data['Close'].iloc[-1]
    # print(data.iloc[0])
    # print(data.iloc[-1])
    # 计算月度回报率
    monthly_return = (last_close - first_close) / first_close
    return monthly_return

def resample_quaterly_data(quaterly_data, target_data):
    'Repeat the quaterly available ratios to same frequency as target return'
    quaterly_data.index = pd.to_datetime(quaterly_data.index)
    target_data.index = pd.to_datetime(target_data.index)
    # target_data_after_1_month.index = pd.to_datetime(target_data_after_1_month.index)
    #print(target_data.index)
    # Resample the quaterly data to daily frequency using Forward Fill
    quaterly_data.index = quaterly_data.index + pd.DateOffset(days=1)
    aligned_quaterly_data = quaterly_data.reindex(target_data.index, method='ffill')

    #aligned_quaterly_data = aligned_quaterly_data.dropna()
    return aligned_quaterly_data

def load_data(args,ticker):
    # print(ticker)
    file_path = args.path_to_file + ticker + '.xlsx'
    sheet_name = ticker + '-US'
    data = pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl')
    data = data.reset_index(drop=True)
    data = data.set_index('Date').T
    data.index = pd.to_datetime(data.index, format='%b \'%y')
    data.index = data.index + pd.offsets.MonthEnd()
    ratio_data = data.apply(pd.to_numeric)
    for col in args.feature_names.split(","): #if there's no such column, fill with 0
            if col not in ratio_data.columns:
                ratio_data.loc[:, col] = 0
                print("ticke is:", ticker)
                print("missing column is:", col)
    ratio_data = ratio_data[args.feature_names.split(",")]
    ratio_data = ratio_data.apply(lambda x: x.fillna(x.mean()), axis=0)
    returns_data = get_historical_returns(ticker, args.start_date, args.end_date)
    adjusted_ratio_data = resample_quaterly_data(ratio_data, returns_data)
    features = pd.concat([adjusted_ratio_data, returns_data],axis=1)
    return features

def generate_betas(ticker_list, args):
    beta_values = {}
    all_X = None
    all_Y = None
    for ticker in ticker_list:
        current_ticker = load_data(args, ticker)
        ######################################### NEW FEATURES #################################################
        current_ticker['0'] =  current_ticker['Price/Earnings'] / (current_ticker['Return on Equity '] * current_ticker['Free Cash Flow per Share']) #VPR
        current_ticker['1'] = my_log(current_ticker['Price/Earnings']) * my_log(current_ticker['Return on Equity ']) * my_log(current_ticker["Dividend Yield (%)"] + 1) #PEI 
        current_ticker['2'] = current_ticker['Return on Equity '] / current_ticker['Price/Earnings'] #PVS
        beta = 2
        current_ticker['3'] = current_ticker['Return on Equity '] / (current_ticker['Price/Earnings']**beta) #RAPS
        current_ticker['4']= (current_ticker['Return on Equity '] * current_ticker['Free Cash Flow per Share'] / (current_ticker['Price/Earnings']+1)) * (1.0 / (current_ticker['Enterprise Value/EBITDA']+1)) #IOS
        current_ticker['5'] = (1.0/(current_ticker["Return on Assets"]+1)) * (1.0/(current_ticker['Enterprise Value/EBITDA']+1)) * (1.0/(current_ticker['Price/Cash Flow']+1)) #EVC
        current_ticker['6'] = current_ticker["Dividend Yield (%)"] * (1.0 / current_ticker["Price/Earnings"]) * current_ticker["Return on Equity "] #NF
        current_ticker['7'] = (current_ticker['Return on Equity '] * current_ticker["Return on Assets"]) * (my_sqrt(current_ticker['Price/Earnings'] \
            * current_ticker['Price/Book Value'] * current_ticker['Price/Cash Flow'])) * (current_ticker['Free Cash Flow per Share'] \
                * current_ticker["Dividend Yield (%)"]) * (1.0 / (current_ticker['Enterprise Value/EBITDA']+1.0)) + (current_ticker['Price/Earnings'])**2 #FinancialHealthScore
        current_ticker['8'] = (1.0 / current_ticker['Price/Earnings']) * current_ticker['Return on Equity '] * current_ticker['Dividend Yield (%)'] #Non-Linear Feature
        current_ticker['9'] = ( current_ticker['Price/Earnings'] + current_ticker['Return on Equity '] + current_ticker['Free Cash Flow per Share'])/3.0

        ######### AUTOMATE #########
        X = current_ticker[args.feature_names.split(",") + list(args.new_feature)]
        Y = current_ticker[args.y]
        X_col = list(X.columns)
        X_col.append(args.y)
        if all_X is None:
            all_X = X
            all_Y = Y
        else:
            all_X = np.vstack([all_X, X])
            all_Y = np.concatenate([all_Y,Y])
        # Z-Score Normalization (Standardization)
        standard_scaler = StandardScaler()
        X_standardized = standard_scaler.fit_transform(X)
        # 转换回DataFrame以便进一步使用
        X_standardized_df = pd.DataFrame(X_standardized, columns=X.columns)
        correlation_matrix = X_standardized_df.corr()
        #print(correlation_matrix)
        X_standardized = sm.add_constant(X_standardized)
        #print(X_standardized)
        model = sm.OLS(Y, X_standardized, missing='drop')
        model_result = model.fit()
        model_result = model.fit()
        # print(model_result)
        #print(beta_df)
        print("this is Beta adj rsquare of {} and the value is {}:".format(ticker, model_result.rsquared_adj))
        coefficients = model_result.params
        #######################################Add New Feature Here###################################
        beta_values[ticker] = dict(zip(['Constant'] + args.feature_names.split(",") + list(args.new_feature), coefficients))
    standard_scaler = StandardScaler()
    all_standardized = standard_scaler.fit_transform(np.c_[all_X,all_Y])
    all_df = pd.DataFrame(all_standardized, columns=X_col)
    correlation_matrix = all_df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Correlation Matrix Heatmap')
    plt.show()
    return beta_values

def load_features1(args, ticker):
    print("Loading features for ticker:", ticker)
    file_path = args.path_to_file + ticker + '.xlsx'
    sheet_name = ticker + '-US'
    data = pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl', index_col='Date')
    # Transpose the DataFrame so that features become columns
    data_transposed = data.T
    # Convert the index to datetime, assuming it's in the format 'Mon YY'
    data_transposed.index = pd.to_datetime(data_transposed.index, format='%b \'%y')
    # Adjust to end of the month since the data is monthly
    data_transposed.index = data_transposed.index + pd.offsets.MonthEnd()
    # Select only the required date range
    #data_transposed = data_transposed.loc[:, start_date:end_date]
    # Convert all data to numeric, errors='coerce' will set non-convertible data to NaN
    data_transposed = data_transposed.apply(pd.to_numeric, errors='coerce')
    # Fill NaN values with column mean
    data_transposed = data_transposed.fillna(data_transposed.mean())
    return data_transposed

def find_common_features(args, tickers):
    common_features = None

    for ticker in tickers:
        data_transposed = load_features1(args,ticker)

        if common_features is None:
            common_features = set(data_transposed.columns)
        else:
            common_features = common_features.intersection(set(data_transposed.columns))

    return list(common_features)

args = parse_arguments()
print(args)
stock_list = ['AAPL', 'PG', 'VZ', 'AMAT', 'MSFT', 'JNJ', 'V', 'PG', 'MA', 
              'TRGP', 'OKE', 'SON', 'DVN', 'HD', 'INTC']
#AMZN, GOOGL, JLL
betas_dict = generate_betas(stock_list, args)
beta_df = pd.DataFrame(betas_dict).transpose()

print(stock_list)
print(dates)
financial_signals = ['Constant'] + args.feature_names.split(",") + list(args.new_feature)
X = beta_df[args.feature_names.split(",") + list(args.new_feature)]

# Z-Score Normalization
standard_scaler = StandardScaler()
X_standardized = standard_scaler.fit_transform(X)
# 转换回DataFrame以便进一步使用
X_standardized_df = pd.DataFrame(X_standardized, columns=X.columns)
X_standardized = sm.add_constant(X_standardized)
gammas = {factor: [] for factor in financial_signals}
adjusted_r_squared = []

for date in range(0, len(dates)-1):
    print("date now is:", dates[date])
    for company in betas_dict:
        monthly_return_data = get_historical_returns_by_day(company, dates[date], dates[date+1], frequency="monthly")
        #print(monthly_return_data)
        ######################################## cross sectional ###################################
        #return_value = monthly_return_data.iloc[0]  # 假设这是1月份的回报率
        column_name = f"Return_{dates[date + 1].replace('-', '')}"
        beta_df.at[company, column_name] = monthly_return_data

    Y = beta_df[column_name]
    model = sm.OLS(Y, X_standardized, missing='drop')
    model_result = model.fit()
    # print(model_result)
    #print(beta_df)
    print("this is adj rsquare:", model_result.rsquared_adj)
    adjusted_r_squared.append(model_result.rsquared_adj)
    coefficients = model_result.params
    for i, factor in enumerate(financial_signals):
        gammas[factor].append(coefficients[i])
# print(gammas)
t_values = {}
p_values = {}
average_gammas = {}
for factor in financial_signals:
    factor_gammas = gammas[factor]  # 这是该因子所有时间点上的γ系数列表
    gamma_mean = np.mean(factor_gammas)
    gamma_std = np.std(factor_gammas, ddof=1)
    gamma_se = gamma_std / np.sqrt(len(factor_gammas))  # 计算标准误差
    t_stat = gamma_mean / gamma_se  # 计算t统计量

    # 双尾t检验
    p_val = stats.t.sf(np.abs(t_stat), df=(len(factor_gammas) - 1)) * 2

    t_values[factor] = t_stat
    p_values[factor] = p_val
    average_gammas[factor] = gamma_mean
# 输出结果
print("T-values:", t_values)
print("P-values:", p_values)
#average_gammas = {factor: sum(values) / len(values) for factor, values in gammas.items()}
average_adj_rsquared = sum(adjusted_r_squared) / len(adjusted_r_squared)
    # print(model_result)
    # print(coefficients)
print(average_gammas)
print("this is the average adj r-square:", average_adj_rsquared)
