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
    parser.add_argument("--start_date", type=int, default=2016, help="startdate")
    parser.add_argument("--path_to_file", type=str, default="./Information Technology/", help="startdate")
    parser.add_argument("--end_date", type=int, default=2023, help="enddate")
    parser.add_argument("--feature_names", type=str, default="Price/Earnings,Price/Book Value,Return on Assets,Return on Equity ,\
Free Cash Flow per Share,Price/Cash Flow,Dividend Yield (%),Enterprise Value/EBITDA,Gross Margin", help="feature_names")
#Enterprise Value/EBIT,
    parser.add_argument("--new_feature", type=str, default="012345678", help="newfeature")
    parser.add_argument("--y", type=str, default="3M Return", help="return")

    args = parser.parse_args()
    return args

def get_historical_returns(ticker, start_date, end_date):
    'Function to fetch Historical Price data and compute returns'
    print(ticker, start_date, end_date)
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
    if args.y == "Return":
        monthly_data = monthly_data['Return'].resample('M').last()
    elif args.y=="3M Return":
        monthly_data = monthly_data['Return'].resample('3M').last()
    returns = monthly_data.pct_change()
    returns = returns.dropna()
    if args.y=="Return": return returns
    elif args.y=="3M Return": return returns
    return returns

def get_historical_returns_by_day(ticker, start_date, end_date, frequency="monthly"):
    'Function to fetch Historical Price data and compute returns'
    # data = yf.download(ticker,start=start_date, end=end_date) 
    print(ticker, start_date, end_date)
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
    # print("This ticker is:", ticker)
    #Exit here to test if the ticker has 对应时间的data
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
    for start in ["FEB start/","JAN start/","DEC start/"]:
        file_path = args.path_to_file + start + ticker + '.xlsx'
        if os.path.isfile(file_path):
            break
    for sn in ['-US', '-USA']:
        try:
            sheet_name = ticker + sn
            data = pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl')
            break
        except:
            pass
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
    returns_data = get_historical_returns(ticker, str(args.start_date-1)+"-12-31", str(args.end_date)+"-12-31")
    adjusted_ratio_data = resample_quaterly_data(ratio_data, returns_data)
    features = pd.concat([adjusted_ratio_data, returns_data],axis=1)
    return features

def generate_betas(ticker_list, args):
    all_corr = np.zeros((19,19))
    ii=1
    for i in range(ii):
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
            Y = current_ticker["Return"]
            
            
            if args.y=="Return":
                assert np.sum(np.abs(X.iloc[0,:]-X.iloc[1,:]))<0.00001 or np.sum(np.abs(X.iloc[2,:]-X.iloc[1,:]))<0.00001
                assert X.shape[0] % 12 == 0
            assert X.shape[0] % 4 == 0
            assert X.shape[0]==Y.shape[0]
            X_col = list(X.columns)
            X_col.append("Return")
            Y=Y.iloc[1:]
            X=X.iloc[:-1,:]
            print(X)
            print(Y)
            if all_X is None:
                all_X = X.iloc[i,:]
                all_Y = np.array([Y.iloc[i]])
            else:
                print("Y is:", Y.iloc[i])
                all_X = np.vstack([all_X, X.iloc[i,:]])
                all_Y = np.concatenate([all_Y,np.array([Y.iloc[i]])])
            # Z-Score Normalization (Standardization)
            standard_scaler = StandardScaler()
            X_standardized = standard_scaler.fit_transform(X)

            # 转换回DataFrame以便进一步使用
            X_standardized_df = pd.DataFrame(X_standardized, columns=X.columns)
            correlation_matrix = X_standardized_df.corr()
            #print(correlation_matrix)
            X_standardized = sm.add_constant(X_standardized)
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
        no_standardized = np.c_[all_X,all_Y]
        all_df = pd.DataFrame(no_standardized, columns=X_col)
        correlation_matrix = all_df.corr()
        spearman_rank_corr = all_df.corr(method='spearman')
        all_corr+=spearman_rank_corr
        plt.figure(figsize=(10, 8))
        sns.heatmap(spearman_rank_corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
        plt.title('Correlation Matrix Heatmap')
        # plt.show()
        plt.savefig("output/figure/corr_{}.svg".format(i))
        plt.close()
    plt.figure(figsize=(10, 8))
    all_corr/=ii
    sns.heatmap(all_corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Correlation Matrix Heatmap')
    # plt.show()
    plt.savefig("output/figure/corr_avg.svg")
    exit()
    return beta_values

def load_features1(args, ticker):
    print("Loading features for ticker:", ticker)
    for start in ["FEB start/","JAN start/","DEC start/"]:
        file_path = args.path_to_file + start + ticker + '.xlsx'
        if os.path.isfile(file_path):
            break
    for sn in ['-US', '-USA']:
        try:
            sheet_name = ticker + sn
            data = pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl', index_col='Date')
            break
        except:
            pass
    # Transpose the DataFrame so that features become columns
    data_transposed = data.T
    # Convert the index to datetime, assuming it's in the format 'Mon YY'
    data_transposed.index = pd.to_datetime(data_transposed.index, format='%b \'%y')
    # Adjust to end of the month since the data is monthly
    data_transposed.index = data_transposed.index + pd.offsets.MonthEnd()
    # Select only the required date range {'Constant': 0.029903637721214335, 'Price/Earnings': -0.03763798712237585, 'Price/Book Value': 0.03577050094427117, 'Return on Assets': -0.017846563104682377, 'Return on Equity ': 0.013856011643014973, 'Free Cash Flow per Share': 0.0045626668692036534, 'Price/Cash Flow': 0.018519080638970765, 'Dividend Yield (%)': -0.021059131504528275, 'Enterprise Value/EBIT': 0.023049280782905066, 'Enterprise Value/EBITDA': 0.023153282029693675, 'Dividend Payout Ratio (%)': 0.018961950417037535, '0': -0.03767389983583187},
    #data_transposed = data_transposed.loc[:, start_date:end_date]
    # Convert all data to numeric, errors='coerce' will set non-convertible data to NaN
    data_transposed = data_transposed.apply(pd.to_numeric, errors='coerce')
    # Fill NaN values with co 'OKE', 'SON', 'DVN', 'HD', 'INTC'lumn mean
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
dates = []
if args.y == "Return":
    dates.append(str(args.start_date-1)+"-12-31")
    for year in range(args.start_date,args.end_date+1):
        dates_sub = [
                '-01-31', '-02-28', '-03-31', '-04-30', '-05-31', '-06-30', '-07-31',
                '-08-31', '-09-30', '-10-31', '-11-30', '-12-31']
        dates = dates+[str(year)+d for d in dates_sub]
elif args.y == "3M Return":
    dates.append(str(args.start_date-1)+"-12-31")
    for year in range(args.start_date,args.end_date+1):
        dates_sub = ['-03-31', '-06-30', '-09-30','-12-31']
        dates = dates+[str(year)+d for d in dates_sub]
print(dates)
#"QRVO"
info_tech_dec_start = ['AAPL', 'AKAM', 'AMD', 'ANET', 'ANSS', 'APH', 'CDNS', 'CDW', 'CTSH', 'ENPH', 'EPAM', 'FFIV', 'FSLR', 'FTNT', 'GEN', 'GLW', 'IBM', 'INTC', 'IT', 'JNPR', 'KLAC', 'LRCX', 'MCHP', 'MPWR', 'MSFT', 'MSI', 'NOW', 'NXPI', 'ON', 'PTC', 'QCOM', 'ROP', 'STX', 'SWKS', 'TDY', 'TEL', 'TER', 'TRMB', 'TXN', 'TYL', 'VRSN', 'WDC', 'ZBRA']
info_tech_jan_start = ['ADI', 'ADSK', 'AMAT', 'AVGO', 'CRM', 'CSCO', 'HPE', 'HPQ', 'INTU', 'KEYS', 'NTAP', 'NVDA', 'PANW', 'SNPS']

info_tech_feb_start = ['ACN', 'ADBE', 'JBL', 'MU', 'ORCL']

Russel_2000_stock_list = ['AAPL', 'AME', 'AMGN', 'AMT',  'DTE', 'DVN', 'EMN', 'FBIN', 'FERG', 'FR', 'GIS', 'H', 'LMT', 'LNT', 'LSTR',  'MCK', 'MCO', 'NEM', 'NLY', 'NNN', 'NUE',  'PEP', 'PH', 'PHM',  'RRC',  'SLB', 'TMO', 'AA', 
'AAL', 'CF',  'CTRA','PG', 'VZ',  'MSFT', 'JNJ', 'V', 'MA', 'TRGP', 'OKE', 'SON',  'HD', 'INTC']
#AMZN, GOOGL, JLL, AMAT, 
lst = find_common_features(args,info_tech_dec_start+info_tech_jan_start+info_tech_feb_start)
# lst.sort()
# print(lst)
# exit()
betas_dict = generate_betas(info_tech_dec_start+info_tech_jan_start+info_tech_feb_start, args)
# print(len(betas_dict.keys()))
# print(len(betas_dict['AAPL']))
# print(len(betas_dict))
beta_df = pd.DataFrame(betas_dict).transpose()

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
# print(X.shape)
for date in range(0, len(dates)-1):
    Y = []
    for company in betas_dict:
        monthly_return_data = get_historical_returns_by_day(company, dates[date], dates[date+1])
        # print("GAMMA MONTHLY RETURN DATA:", monthly_return_data)
        Y.append(monthly_return_data)
    # print(X.shape)
    Y = np.array(Y)
    # print(X_standardized)
    # print(Y)
    model = sm.OLS(Y, X_standardized, missing='drop')
    model_result = model.fit()
    # print(model_result)
    #print(beta_df)
    coefficients = model_result.params
    # print(coefficients)
    print("this is adj rsquare:", model_result.rsquared_adj)
    adjusted_r_squared.append(model_result.rsquared_adj)
    for i, factor in enumerate(financial_signals):
        gammas[factor].append(coefficients[i])
print(len(gammas.keys()))
print(len(gammas['Constant']))
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
print("this is the median adj r-square:",np.median(adjusted_r_squared))
