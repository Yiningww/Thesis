import pandas as pd
import numpy as np
import yfinance as yf
import os
from datetime import datetime
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import statsmodels.api as sm
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
'''
beta_values = {
    "AAPL": {"PE_beta": 0.5, "PB_beta": 0.3, "ROI_beta": 0.2, "ROE_beta": 0.4},
    "MSFT": {"PE_beta": 0.6, "PB_beta": 0.4, "ROI_beta": 0.3, "ROE_beta": 0.5},
    "GOOG": {"PE_beta": 0.7, "PB_beta": 0.5, "ROI_beta": 0.4, "ROE_beta": 0.6}
}

# 转换成DataFrame
beta_df = pd.DataFrame(beta_values).transpose()
# 查看DataFrame
print(beta_df)
'''
def get_historical_returns(ticker, start_date, end_date, frequency="monthly"):
    'Function to fetch Historical Price data and compute returns'
    data = yf.download(ticker,start=start_date, end=end_date)
    # Calculate Daily Returns
    daily_data = data.copy()
    #print("daily_data:", daily_data)
    daily_data['Return'] = daily_data['Close'].pct_change()
    daily_returns = daily_data[['Return']].dropna()
    # Calculate Monthly Returns
    monthly_data = data.copy()
    monthly_data['Return'] = monthly_data['Close']
    monthly_data = monthly_data['Return'].resample('M').last()
    #print("monthly data after:", monthly_data)
    monthly_returns = monthly_data.pct_change()
    monthly_returns = monthly_returns.dropna()

    if frequency == "daily": return daily_returns
    if frequency == "monthly": return monthly_returns
    return monthly_data

def get_historical_returns_by_day(ticker, start_date, end_date, frequency="monthly"):
    'Function to fetch Historical Price data and compute returns'
    data = yf.download(ticker,start=start_date, end=end_date)
    print("This ticker is:", ticker)
    #print(data)
    #Exit here to test if the ticker has 对应时间的data
    first_close = data['Close'].iloc[0]
    last_close = data['Close'].iloc[-1]
    # print(data.iloc[0])
    # print(data.iloc[-1])
    # 计算月度回报率
    monthly_return = (last_close - first_close) / first_close
    return monthly_return

# 假设beta_df是你之前创建的包含β值的DataFrame
# 获取2020年1月的AAPL回报率
# for company in beta_values:
#     monthly_return_aapl_200430 = get_historical_returns(company, "2020-03-31", "2020-04-30", frequency="monthly")
#     print(monthly_return_aapl_200430)
#     # 假设我们只关心2020年1月的数据
#     return_value = monthly_return_aapl_200430.iloc[0]  # 假设这是1月份的回报率
#     beta_df.at[company, "Return_2020-04-30"] = return_value

# 查看更新后的DataFramefor company in beta_values:
#     monthly_return_aapl_200430 = get_historical_returns(company, "2020-03-31", "2020-04-30", frequency="monthly")
#     print(monthly_return_aapl_200430)
#     # 假设我们只关心2020年1月的数据
#     return_value = monthly_return_aapl_200430.iloc[0]  # 假设这是1月份的回报率
#     beta_df.at[company, "Return_2020-04-30"] = return_value
#
# # 查看更新后的DataFrame

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

def load_features(path_to_file, ticker, start_date, end_date):
    print("ticker:", ticker)
    'Function to Load all features for a single company'
    # Load the Excel file and read Data from the file
    file_path = path_to_file + ticker + '.xlsx'
    sheet_name = ticker + '-US'
    #sheet_name = ticker
    data = pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl')
    # Remove rows with any NaN values
    # Because time frame is longer, cannot apply this
    # data = data.dropna()
    # Reset the index of the DataFrame and drop the old index
    data = data.reset_index(drop=True)
    data = data.set_index('Date').T
    data.index = pd.to_datetime(data.index, format='%b \'%y')
    data.index = data.index + pd.offsets.MonthEnd()
    ratio_data = data.apply(pd.to_numeric)
    # Select a few columns
    pe_column = 'Price/Earnings'
    pb_column = 'Price/Book Value'
    roa_column = 'Return on Assets'
    roe_column = 'Return on Equity '
    fcf_column = 'Free Cash Flow per Share'
    pcf_column = 'Price/Cash Flow'
    dy_column = 'Dividend Yield (%)'
    ebit_column = 'Enterprise Value/EBIT'
    ebitda_column = 'Enterprise Value/EBITDA'
    divpr_column = 'Dividend Payout Ratio (%)'
    # inv_turnover = 'Inventory Turnover'
    # asset_turnover = 'Asset Turnover'
    # cash_ratio = 'Cash Ratio'
    # de_column = 'Total Debt/Equity (%)'
    ratio_data = ratio_data[['Price/Earnings', 'Price/Book Value', 'Return on Assets', 'Return on Equity ',
                             'Free Cash Flow per Share', 'Price/Cash Flow']]

    # print(ratio_data[expected_column[1]])
    #expected_column = data[[pe_column, pb_column, roa_column, roe_column, fcf_column, pcf_column]]
    # ratio_data = data[[pe_column, pb_column, roa_column, roe_column, fcf_column, pcf_column, dy_column, ebit_column,
    #                     ebit_column, divpr_column]]
    # Replace N/A dates
    #print("Before N/A:", ratio_data)
    # for column in ratio_data.select_dtypes(include=[np.number]).columns:
    #     ratio_data[column] = ratio_data[column].fillna(ratio_data[column].mean())
    ###ChatGPT
    # # Reindex the dataframe to ensure all expected columns are present
    #
    # ratio_data = ratio_data.reindex(expected_column, axis=1)
    # # print(ratio_data)
    # # Fill missing PE ratio (or any other missing ratios) with 0
    # ratio_data = ratio_data.fillna(0)
    #
    # # Now, select only the required columns
    # ratio_data = ratio_data[['Price/Earnings', 'Price/Book Value', 'Return on Assets', 'Return on Equity ',
    #                          'Free Cash Flow per Share']]
    ###ChatGPT END

    # Process Return Data
    returns_data = get_historical_returns(ticker, start_date, end_date)
    adjusted_ratio_data = resample_quaterly_data(ratio_data, returns_data)
    features = pd.concat([adjusted_ratio_data, returns_data],axis=1)
    # print(adjusted_ratio_data.index)
    # print(returns_data.index)
    # print("rrr:", ratio_data)
    # print("fff:", features)
    return features

def load_features2(path_to_file, ticker, start_date, end_date):
    print("ticker:", ticker)
    'Function to Load all features for a single company'
    # Load the Excel file and read Data from the file
    file_path = path_to_file + ticker + '.xlsx'
    sheet_name = ticker + '-US'
    data = pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl')
    # Remove rows with any NaN values
    # Because time frame is longer, cannot apply this
    # data = data.dropna()
    # Reset the index of the DataFrame and drop the old index
    data = data.reset_index(drop=True)
    data = data.set_index('Date').T
    data.index = pd.to_datetime(data.index, format='%b \'%y')
    data.index = data.index + pd.offsets.MonthEnd()
    ratio_data = data.apply(pd.to_numeric)
    # Select a few columns
    pe_column = 'Price/Earnings'
    pb_column = 'Price/Book Value'
    roa_column = 'Return on Assets'
    roe_column = 'Return on Equity '
    fcf_column = 'Free Cash Flow per Share'
    pcf_column = 'Price/Cash Flow'
    dy_column = 'Dividend Yield (%)'
    ebit_column = 'Enterprise Value/EBIT'
    ebitda_column = 'Enterprise Value/EBITDA'
    divpr_column = 'Dividend Payout Ratio (%)'
    # inv_turnover = 'Inventory Turnover'
    # asset_turnover = 'Asset Turnover'
    # cash_ratio = 'Cash Ratio'
    # de_column = 'Total Debt/Equity (%)'
    expected_columns = [pe_column, pb_column, roa_column, roe_column, fcf_column, pcf_column, dy_column,
                        ebit_column, ebitda_column, divpr_column]
    # print(ratio_data[expected_column[1]])
    # Replace N/A dates
    #print("Before N/A:", ratio_data)
    #for column in expected_column.select_dtypes(include=[np.number]).columns:
    #     expected_column[column] = expected_column[column].fillna(expected_column[column].mean())
    ###ChatGPT
    # Reindex the dataframe to ensure all expected columns are present
    #ratio_data = ratio_data.reindex(expected_columns, axis=1)
    # Fill missing values with the mean (or max/min) for each feature
    # Fill missing PE ratio (or any other missing ratios) with 0 !!!!!!!!!
    # ratio_data = ratio_data.fillna(20)
    # 设置Pandas以显示所有列（根据你的DataFrame的列数）
    # Set Pandas to display all columns (None means no limit)
    pd.set_option('display.max_columns', None)
    # Set Pandas to display all rows (None means no limit)
    pd.set_option('display.max_rows', None)
    # Now, select only the required columns
    #ratio_data = ratio_data.copy()
    for col in expected_columns:
         if col not in ratio_data.columns:
             #ratio_data[col] = np.nan
             #ratio_data.loc[:, col] = np.nan
             ratio_data.loc[:, col] = 0 ## fill non-existing columns with 0
    ratio_data = ratio_data[['Price/Earnings', 'Price/Book Value', 'Return on Assets', 'Return on Equity ', 'Free Cash Flow per Share',
                             'Price/Cash Flow', 'Dividend Yield (%)', 'Enterprise Value/EBIT', 'Enterprise Value/EBITDA'
                             , 'Dividend Payout Ratio (%)']]
                             # 'Free Cash Flow per Share', 'Price/Cash Flow', 'Dividend Yield (%)',
                             # 'Enterprise Value/EBIT', 'Enterprise Value/EBITDA', 'Dividend Payout Ratio (%)']]
    ratio_data = ratio_data.apply(lambda x: x.fillna(x.mean()), axis=0)

    # ratio_data = ratio_data.reindex(expected_columns, axis=1)
    ###ChatGPT END
    # Process Return Data
    returns_data = get_historical_returns(ticker, start_date, end_date)
    #print("check time consistency:", returns_data)
    adjusted_ratio_data = resample_quaterly_data(ratio_data, returns_data)
    features = pd.concat([adjusted_ratio_data, returns_data], axis=1)
    return features

def load_features2_contd(ratio_data, returns_data):
    print("this is mean:", ratio_data.mean())
    # Fill missing values with the mean (or max/min) for each feature
    for column in ratio_data:
        # Replace missing values with the mean (or max/min) from the feature_stats
        ratio_data[column].fillna(ratio_data.mean()[column], inplace=True)
    adjusted_ratio_data = resample_quaterly_data(ratio_data, returns_data)
    features = pd.concat([adjusted_ratio_data, returns_data],axis=1)
    return features

def load_df(company_code):
    # Load the Excel file and read Data from the file
    file_path = company_code + '.xlsx'
    sheet_name = company_code
    data = pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl')

    # Remove rows with any NaN values
    data = data.dropna()
    # Reset the index of the DataFrame and drop the old index
    data = data.reset_index(drop=True)
    return data

## Multi data
def multi_stock_df(path_to_file, ticker_list, start_date, end_date):
    company_data = {}
    for ticker in ticker_list:
        company_data[ticker] = load_features2(path_to_file, ticker, start_date, end_date)
    # Initialize a list to hold DataFrames with the new multi-index
    multi_index_dfs = []
    # Set Pandas to display all columns (None means no limit)
    pd.set_option('display.max_columns', None)
    # Set Pandas to display all rows (None means no limit)
    pd.set_option('display.max_rows', None)
    for company, df in company_data.items():
        # Set the company name as an additional level in the index
        df_multi_index = df.copy()
        df_multi_index['Company'] = company
        df_multi_index.set_index(['Company', df_multi_index.index], inplace=True)
        print(company,df_multi_index.head(3))
        # Append to the list
        multi_index_dfs.append(df_multi_index)
    # Concatenate all DataFrames into a single multi-index DataFrame
    # print(multi_index_dfs)
    final_df = pd.concat(multi_index_dfs)
    #print(company_data['AAPL']['Price/Earnings'].mean())
    for column in final_df:
        # Replace missing values with the mean (or max/min) from the feature_stats
        final_df[column].fillna(final_df[column].mean(), inplace=True)
    # for column in final_df.select_dtypes(include=[np.number]).columns:
    #     final_df[column] = final_df[column].fillna(final_df[column].mean())
    return final_df

def get_future_return(row, months):
    # 获取结束日期
    end_date = (row['Date'] + pd.DateOffset(months=months)).date()
    end_date = pd.to_datetime(end_date) + pd.offsets.MonthEnd(0)
    # 使用提供的函数获取历史回报率
    # historical_returns = get_historical_returns(row['Company'], row['Date'], end_date, frequency="monthly")
    # Call the function to get historical returns
    historical_returns = get_historical_returns(row['Company'], row['Date'], end_date + pd.Timedelta(days=1),
                                                frequency="monthly")
    # The historical_returns should be a Series with a DateTimeIndex. We round the timestamp to remove the hour-minute-second part.
    end_date = end_date.round('D')

    # Now we can safely get the return for the end_date
    future_return = historical_returns.get(end_date)

    # 获取对应的回报率
    # future_return = historical_returns.loc[end_date]['Return'] if end_date in historical_returns.index else None
    return future_return

def load_features1(path_to_file, ticker):
    print("Loading features for ticker:", ticker)
    file_path = path_to_file + ticker + '.xlsx'
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

def find_common_features(path_to_file, tickers, start_date, end_date):
    common_features = None

    for ticker in tickers:
        data_transposed = load_features1(path_to_file, ticker)

        if common_features is None:
            common_features = set(data_transposed.columns)
        else:
            common_features = common_features.intersection(set(data_transposed.columns))

    return list(common_features)

def compile_data_for_all_companies(path_to_file, tickers, start_date, end_date):
    all_data = []

    for ticker in tickers:
        data = load_features2(path_to_file, ticker, start_date, end_date)
        all_data.append(data)

    # Concatenate all the dataframes along the columns (features)
    combined_data = pd.concat(all_data, axis=1)

    return combined_data

def calculate_statistics(combined_data):
    # Calculate the mean, max, and min for each feature
    statistics = {
        'mean': combined_data.mean(axis=1),
        'max': combined_data.max(axis=1),
        'min': combined_data.min(axis=1)
    }
    statistics_df = pd.DataFrame(statistics)

    return statistics_df

def generate_betas(path_to_file, ticker_list, start_date, end_date):
    beta_values = {}
    for ticker in ticker_list:
        current_ticker = load_features2(path_to_file, ticker, start_date, end_date)
        for column in current_ticker:
            # Replace missing values with the mean (or max/min) from the feature_stats
            current_ticker[column].fillna(current_ticker[column].mean(), inplace=True)
            ######################################### NEW FEATURES #################################################
            current_ticker['Investment Efficiency'] = (np.log(current_ticker['Price/Earnings']) + np.log(current_ticker['Return on Assets']) +
                                     np.log(current_ticker['Return on Equity '])) / 3
            # Calculate the new feature
            current_ticker['LARE'] = (np.sqrt(current_ticker['Return on Equity '] / current_ticker['Return on Assets']) /
                          (1 + np.abs(current_ticker['Return on Equity '] - current_ticker['Return on Assets'])))
            current_ticker['Integrated ROE/PE'] = current_ticker['Return on Equity ']/(current_ticker['Price/Earnings']**2)
            # Normalize the new feature to the range of 0 to 1
            scaler = MinMaxScaler()
            current_ticker['LARE_normalized'] = scaler.fit_transform(current_ticker[['LARE']])
            current_ticker['Integrated ROE/PE'] = scaler.fit_transform(current_ticker[['Integrated ROE/PE']])
            # The 'LARE_normalized' column now contains the normalized values of the new feature.
            # Display the new feature
            #current_ticker['Investment Efficiency'] = np.log(investment_efficiency)
            ######### AUTOMATE #########
            new_feature = ''
            Y = current_ticker['Return']
            if new_feature != '':
                X = current_ticker[['Price/Earnings', 'Price/Book Value', 'Return on Assets', 'Return on Equity ', 'Free Cash Flow per Share',
                             'Price/Cash Flow', 'Dividend Yield (%)', 'Enterprise Value/EBIT', 'Enterprise Value/EBITDA'
                             , 'Dividend Payout Ratio (%)', 'Integrated ROE/PE', new_feature]]
            else:
                X = current_ticker[['Price/Earnings', 'Price/Book Value', 'Return on Assets', 'Return on Equity ',
                                    'Free Cash Flow per Share',
                                    'Price/Cash Flow', 'Dividend Yield (%)', 'Enterprise Value/EBIT',
                                    'Enterprise Value/EBITDA', 'Dividend Payout Ratio (%)']]
            print("X is:", X)

            #X = sm.add_constant(X)
            min_max_scaler = MinMaxScaler()
            X_minmax = min_max_scaler.fit_transform(X)
            # Z得分标准化
            standard_scaler = StandardScaler()
            X_standardized = standard_scaler.fit_transform(X)
            # 转换回DataFrame以便进一步使用
            X_minmax_df = pd.DataFrame(X_minmax, columns=X.columns)
            X_standardized_df = pd.DataFrame(X_standardized, columns=X.columns)
            correlation_matrix = X_standardized_df.corr()
            #print(correlation_matrix)
            X_standardized = sm.add_constant(X_standardized)
            print(X_standardized)
            model = sm.OLS(Y, X_standardized, missing='drop')
            model_result = model.fit()
            # model_result = model.fit_regularized(method='elastic_net', alpha=0.1,
            # L1_wt=1.0, start_params=None, profile_scale=False, refit=False)
            #print(model_result.summary())
            coefficients = model_result.params
            #######################################Add New Feature Here###################################
            financial_signals = ['Constant', 'Price/Earnings', 'Price/Book Value', 'Return on Assets', 'Return on Equity ', 'Free Cash Flow per Share',
                             'Price/Cash Flow', 'Dividend Yield (%)', 'Enterprise Value/EBIT', 'Enterprise Value/EBITDA'
                             , 'Dividend Payout Ratio (%)', new_feature]
            beta_values[ticker] = dict(zip(financial_signals, coefficients))
    return beta_values
# Compile data for all companies
# combined_data = compile_data_for_all_companies(path_to_file, tickers, start_date, end_date)
#
# # Calculate statistics for each feature across all companies
# statistics_df = calculate_statistics(combined_data)
# print(statistics_df)

# Get the common features across the tickers
#common_features = find_common_features(path_to_file, tickers, start_date, end_date)
#print("Common features across tickers:", common_features)
# Define Input and Parameters
# 1. File Path
path_to_file = "./"
# 2. Ticker List
#ChatGPT_stocks = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'JNJ', 'V', 'PG', 'JPM', 'UNH', 'MA', 'INTC', 'VZ', 'GOOG', 'HD', 'T']
#stock_list = ['COTY', 'AR', 'EXAS', 'AMC'] #unavailable from 20130131 to 201712031: COTY,
stock_list = ['AAPL',
              'MSFT', 'AMZN', 'GOOGL', 'JNJ', 'V', 'PG', 'JPM', 'UNH', 'MA', 'LYV', 'JLL',
              'TRGP']# 'FTI', 'OKE', 'SON',  'DVN']


              #'FTI', 'OKE'] 'EQT', 'RRC',
              #, 'SON', 'RRC', 'DVN', 'EQT', 'NOV', 'OXY']
              # 'LYV', 'JLL', 'FTI', 'OKE', 'SON', 'RRC', 'DVN', 'COTY', 'AR', 'EQT', 'NOV']
#stock_list = ['AAPL', 'AMZN']
# 3. Target Time Frame
start_date = '2016-01-31'
end_date = '2017-12-31'
features_and_return = multi_stock_df(path_to_file, stock_list, start_date, end_date)
# print(features_and_return)
# exit()
multi_stock_return = pd.DataFrame(features_and_return['Return'])
final_df = features_and_return.reset_index()
betas_dict = generate_betas(path_to_file, stock_list, start_date, end_date)
exit()
beta_df = pd.DataFrame(betas_dict).transpose()
dates2 = ['2017-01-31', '2017-02-28', '2017-03-31', '2017-04-30', '2017-05-31', '2017-06-30', '2017-07-31',
         '2017-08-31', '2017-09-30', '2017-10-31', '2017-11-30', '2017-12-31']
dates = ['2013-01-31', '2013-02-28', '2013-03-31', '2013-04-30', '2013-05-31', '2013-06-30', '2013-07-31',
         '2013-08-31', '2013-09-30', '2013-10-31', '2013-11-30', '2013-12-31',
         '2014-01-31', '2014-02-28', '2014-03-31', '2014-04-30', '2014-05-31', '2014-06-30', '2014-07-31',
         '2014-08-31', '2014-09-30', '2014-10-31', '2014-11-30', '2014-12-31',
         '2015-01-31', '2015-02-28', '2015-03-31', '2015-04-30', '2015-05-31', '2015-06-30', '2015-07-31',
         '2015-08-31', '2015-09-30', '2015-10-31', '2015-11-30', '2015-12-31',
         '2016-01-31', '2016-02-29', '2016-03-31', '2016-04-30', '2016-05-31', '2016-06-30', '2016-07-31',
         '2016-08-31', '2016-09-30', '2016-10-31', '2016-11-30', '2016-12-31',
         '2017-01-31', '2017-02-28', '2017-03-31', '2017-04-30', '2017-05-31', '2017-06-30', '2017-07-31',
         '2017-08-31', '2017-09-30', '2017-10-31', '2017-11-30', '2017-12-31'
         ]
################# NEW FEATURES:  'Investment Efficiency'
# old: 'Return on Assets' 'Return on Equity '
new_feature = ''
financial_signals = ['Constant', 'Price/Earnings', 'Price/Book Value', 'Return on Assets', 'Return on Equity ', 'Free Cash Flow per Share',
                             'Price/Cash Flow', 'Dividend Yield (%)', 'Enterprise Value/EBIT', 'Enterprise Value/EBITDA'
                             , 'Dividend Payout Ratio (%)']
X = beta_df[['Price/Earnings', 'Price/Earnings', 'Price/Book Value', 'Return on Assets', 'Return on Equity ', 'Free Cash Flow per Share',
                             'Price/Cash Flow', 'Dividend Yield (%)', 'Enterprise Value/EBIT', 'Enterprise Value/EBITDA'
                             , 'Dividend Payout Ratio (%)']]
if new_feature != '':
    financial_signals.append(new_feature)
    X = beta_df[['Price/Earnings', 'Price/Earnings', 'Price/Book Value', 'Return on Assets', 'Return on Equity ', 'Free Cash Flow per Share',
                             'Price/Cash Flow', 'Dividend Yield (%)', 'Enterprise Value/EBIT', 'Enterprise Value/EBITDA'
                             , 'Dividend Payout Ratio (%)', new_feature]]

#X = sm.add_constant(X)
# Z得分标准化
standard_scaler = StandardScaler()
X_standardized = standard_scaler.fit_transform(X)
# 转换回DataFrame以便进一步使用
X_standardized_df = pd.DataFrame(X_standardized, columns=X.columns)
#correlation_matrix = X_standardized_df.corr()
#print(correlation_matrix)
X_standardized = sm.add_constant(X_standardized)
gammas = {factor: [] for factor in financial_signals}
adjusted_r_squared = []
#print("aaa")
#print(get_historical_returns_by_day('AAPL', '2017-04-30', '2017-05-31', frequency="monthly"))

for date in range(0, len(dates)-1):
    print("date now is:", dates[date])
    for company in betas_dict:
        monthly_return_data = get_historical_returns_by_day(company, dates[date], dates[date+1], frequency="monthly")
        #print(monthly_return_data)
        ######################################## cross sectional ###################################
        #return_value = monthly_return_data.iloc[0]  # 假设这是1月份的回报率
        column_name = f"Return_{dates[date + 1].replace('-', '')}"
        beta_df.at[company, column_name] = monthly_return_data
        print("aaaaaaaaaa")

    Y = beta_df[column_name]
    model = sm.OLS(Y, X_standardized, missing='drop')
    model_result = model.fit()
    print(model_result)
    #print(beta_df)
    print("this is adj rsquare:", model_result.rsquared_adj)
    adjusted_r_squared.append(model_result.rsquared_adj)
    coefficients = model_result.params
    for i, factor in enumerate(financial_signals):
        gammas[factor].append(coefficients[i])
print(gammas)
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
exit()

















#print(beta_df)
# Pre-processing
# Loading Phase: Took a while to run this
features_and_return = multi_stock_df(path_to_file, stock_list, start_date, end_date)
multi_stock_return = pd.DataFrame(features_and_return['Return'])
final_df = features_and_return.reset_index()
#final_df['1M Return'] = final_df.apply(lambda row: get_future_return(row, 1), axis=1)
#company_data['AAPL'].info()
########################################## Evaluation period ##########################################################
start_date_eval = '2017-09-30'
end_date_eval = '2018-09-30'
features_and_return_eval = multi_stock_df(path_to_file, stock_list, start_date_eval, end_date_eval)
multi_stock_return_eval = pd.DataFrame(features_and_return_eval['Return'])
final_df_eval = features_and_return_eval.reset_index()
# 设置Pandas以显示所有列（根据你的DataFrame的列数）
# Set Pandas to display all columns and rows (None means no limit)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
extracted = final_df[final_df['Date'] == '2016-11-30']
extracted_eval = final_df_eval[final_df_eval['Date'] == '2017-11-30']
# print(final_df[final_df['Date'] == '2016-11-30'].iterrows()
specific_date = extracted.copy()
specific_date_eval = extracted_eval.copy()
#specific_date['1M Return'] = specific_date.apply(lambda row: get_future_return(row, 1), axis=1)
# specific_date['3M Return'] = specific_date.apply(lambda row: get_future_return(row, 3), axis=1)
# print(specific_date.head())
#print(specific_date['1M Return'])
#Train and Eval:
Y = specific_date['Return']
print(Y)

Y_eval = specific_date_eval['Return']*100
# specific_date['log pe'] = np.log(specific_date['Price/Earnings'])
# specific_date['log pb'] = np.log(specific_date['Price/Book Value'])
# specific_date['log roa'] = np.log(specific_date['Return on Assets'])
# specific_date['log roe'] = np.log(specific_date['Return on Equity '])
# specific_date['log fcfps'] = np.log(specific_date['Free Cash Flow per Share'])
#specific_date['log pcf'] = np.log(specific_date['Price/Cash Flow'])

###### New Signals ######
investment_efficiency = (specific_date['Price/Earnings'] + specific_date['Return on Assets'] + \
                                         specific_date['Return on Equity '])/3
investment_efficiency_eval = (specific_date_eval['Price/Earnings'] + specific_date_eval['Return on Assets'] + \
                                         specific_date_eval['Return on Equity '])/3
specific_date['Investment Efficiency'] = np.log(investment_efficiency)
specific_date['Investment Efficiency'].fillna(specific_date['Investment Efficiency'].mean(), inplace=True)

specific_date_eval['Investment Efficiency'] = np.log(investment_efficiency_eval)
specific_date_eval['Investment Efficiency'].fillna(specific_date_eval['Investment Efficiency'].mean(), inplace=True)

#print(specific_date['Investment Efficiency'])
# print(specific_date['PE_ROE_Interaction'])
specific_date['PE_ROE_Interaction'] = np.log(specific_date['Price/Earnings'] + 1) * np.sqrt(specific_date['Return on Equity '])
maxval = max(specific_date['PE_ROE_Interaction'])
minval = min(specific_date['PE_ROE_Interaction'])

specific_date_eval['PE_ROE_Interaction'] = np.log(specific_date_eval['Price/Earnings'] + 1) * np.sqrt(specific_date_eval['Return on Equity '])
maxval_eval = max(specific_date_eval['PE_ROE_Interaction'])
minval_eval = min(specific_date_eval['PE_ROE_Interaction'])
specific_date['Normalized_PE_ROE_Interaction'] = (specific_date['PE_ROE_Interaction']-minval)/(maxval-minval)
specific_date['Normalized_PE_ROE_Interaction'].fillna(specific_date['Normalized_PE_ROE_Interaction'].mean(), inplace=True)
specific_date_eval['Normalized_PE_ROE_Interaction'] = (specific_date_eval['PE_ROE_Interaction']-minval_eval)/(maxval_eval-minval_eval)
specific_date_eval['Normalized_PE_ROE_Interaction'].fillna(specific_date_eval['Normalized_PE_ROE_Interaction'].mean(), inplace=True)
# X = specific_date[['log pe', 'Price/Earnings', 'Price/Book Value', 'log roa', 'Return on Assets',
#                    'log roe', 'Return on Equity ', 'log fcfps', 'Free Cash Flow per Share']]
X = specific_date[['Price/Earnings',  'Price/Book Value', 'Return on Assets',
                    'Return on Equity ', 'Free Cash Flow per Share', 'Dividend Yield (%)',
                    'Enterprise Value/EBIT', 'Enterprise Value/EBITDA',
                   'Dividend Payout Ratio (%)']]
min_max_scaler = MinMaxScaler()
X_minmax = min_max_scaler.fit_transform(X)

# Z得分标准化
standard_scaler = StandardScaler()
X_standardized = standard_scaler.fit_transform(X)

# 转换回DataFrame以便进一步使用
X_minmax_df = pd.DataFrame(X_minmax, columns=X.columns)
X_standardized_df = pd.DataFrame(X_standardized, columns=X.columns)

X_eval = specific_date_eval[['Price/Earnings',  'Price/Book Value', 'Return on Assets',
                    'Return on Equity ', 'Free Cash Flow per Share', 'Dividend Yield (%)',
                    'Enterprise Value/EBIT', 'Enterprise Value/EBITDA',
                   'Dividend Payout Ratio (%)', 'Investment Efficiency']]
correlation_matrix = X_standardized_df.corr()
print(correlation_matrix)
X_standardized = sm.add_constant(X_standardized)
X = sm.add_constant(X)

X_eval = sm.add_constant(X_eval)
# print(Y_eval)
# print(X_eval)
model = sm.OLS(Y, X_minmax, missing='drop')
model_result = model.fit()
#model_result = model.fit_regularized(method='elastic_net', alpha=0.1,
                                     #L1_wt=1.0, start_params=None, profile_scale=False, refit=False)
print(model_result.summary())
coefficients = model_result.params
print(coefficients)
exit()
# Predictions
# print(Y_eval.shape)
# print(X_eval.shape)
# print(Y.shape)
# print(X.shape)
model_eval = sm.OLS(Y_eval, X_eval, missing='drop')
#model_result_eval = model_eval.fit()
#print(model_result_eval.summary())
predictions = model_eval.predict(coefficients)
print("Y_EVAL:", Y_eval)
print("predictions:", predictions)
# Calculate the MSE
mse = np.mean((Y_eval - predictions) ** 2)
print("Mean Squared Error:", mse)
exit()

## LASSO try
# define model
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
model = LassoCV(alphas=arange(0, 1, 0.01), cv=cv, n_jobs=-1)
# fit model
model.fit(X, y)
# summarize chosen configuration
print('alpha: %f' % model.alpha_)


# for index, row in final_df[final_df['Date'] == '2016-11-30'].iterrows():
#     #print(index, row['Date'])
#     print(get_historical_returns(row['Company'], row['Date'], row['Date']))
'''
# Setup TimeFrame and Target Firm
ticker = 'GOOG'
company_code = 'GOOG-US'
start_date = '2017-08-30'
end_date = '2018-09-30'

# Process Ratio Data
data = load_df(company_code)
data = data.set_index('Date').T
data.index = pd.to_datetime(data.index, format='%b \'%y')
data.index = data.index + pd.offsets.MonthEnd()
ratio_data = data.apply(pd.to_numeric)

# Select a few columns
pe_column = 'Price/Earnings'
pb_column = 'Price/Book Value'
roa_column = 'Return on Assets'
roe_column = 'Return on Equity '
fcf_column = 'Free Cash Flow per Share'
ratio_data = data[[pe_column, pb_column, roa_column, roe_column, fcf_column]]

print("Ratio data:", ratio_data)

# Process Return Data
returns_data = get_historical_returns(ticker, start_date, end_date)
# print(returns_data.index) # DateTime Index(Check)

adjusted_ratio_data = repeat_quaterly_data(ratio_data, returns_data)
#print("Adjusted ratio data:", adjusted_ratio_data)
#print("Returns are:", returns_data)
features = pd.concat([adjusted_ratio_data, returns_data], axis=1)
print(features.head(20).to_string())

# Scale the data
# scaler = MinMaxScaler(feature_range=(0,1))
# scaled_data = scaler.fit_transform(data)

# Create Sequence: use previous 4 quarters to predict the next quarter
seq_length = 3
X, y = create_sequences(features, returns_data, seq_length)
#print(X)
'''
