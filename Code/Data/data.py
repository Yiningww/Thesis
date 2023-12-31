# use automatically configured the lasso regression algorithm
from numpy import arange
from pandas import read_csv
from sklearn.linear_model import LassoCV
from sklearn.model_selection import RepeatedKFold
# load the dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
dataframe = read_csv(url, header=None)
data = dataframe.values
X, y = data[:, :-1], data[:, -1]
# define model evaluation method
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# define model
model = LassoCV(alphas=arange(0, 1, 0.01), cv=cv, n_jobs=-1)
# fit model
model.fit(X, y)
# summarize chosen configuration
print('alpha: %f' % model.alpha_)
exit()


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
from pandas import read_csv
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.model_selection import RepeatedKFold

def get_historical_returns(ticker, start_date, end_date, frequency="monthly"):
    'Function to fetch Historical Price data and compute returns'

    data = yf.download(ticker,start=start_date, end=end_date)

    # Calculate Daily Returns
    daily_data = data.copy()
    daily_data['Return'] = daily_data['Close'].pct_change()
    daily_returns = daily_data[['Return']].dropna()

    # Calculate Monthly Returns
    monthly_data = data.copy()
    monthly_data['Return'] = monthly_data['Close']
    monthly_data = monthly_data['Return'].resample('M').last()
    monthly_returns = monthly_data.pct_change()
    monthly_returns = monthly_returns.dropna()

    if frequency == "daily": return daily_returns
    if frequency == "monthly": return monthly_returns

    return monthly_data

def resample_quaterly_data(quaterly_data, target_data):
    'Repeat the quaterly available ratios to same frequency as target return'

    quaterly_data.index = pd.to_datetime(quaterly_data.index)
    target_data.index = pd.to_datetime(target_data.index)
    # target_data_after_1_month.index = pd.to_datetime(target_data_after_1_month.index)

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
    print(adjusted_ratio_data.index)
    print(returns_data.index)
    exit()
    print("rrr:", ratio_data)
    print("fff:", features)
    exit()
    return features

def load_features2(path_to_file, ticker, start_date, end_date):
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
             print(ticker)
             print(col)
             #ratio_data[col] = np.nan
             ratio_data.loc[:, col] = np.nan

    ratio_data = ratio_data[['Price/Earnings', 'Price/Book Value', 'Return on Assets', 'Return on Equity ',
                             'Free Cash Flow per Share', 'Price/Cash Flow', 'Dividend Yield (%)',
                             'Enterprise Value/EBIT', 'Enterprise Value/EBITDA', 'Dividend Payout Ratio (%)']]

    # ratio_data = ratio_data.reindex(expected_columns, axis=1)
    ###ChatGPT END
    # Process Return Data
    returns_data = get_historical_returns(ticker, start_date, end_date)
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
        # Append to the list
        multi_index_dfs.append(df_multi_index)
    # Concatenate all DataFrames into a single multi-index DataFrame
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
path_to_file = "/Users/wangyining/Desktop/Thesis/Code/Data/"
# 2. Ticker List
#ChatGPT_stocks = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'JNJ', 'V', 'PG', 'JPM', 'UNH', 'MA', 'INTC', 'VZ', 'GOOG', 'HD', 'T']
#stock_list = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'JNJ', 'V', 'PG', 'JPM', 'UNH', 'MA']
stock_list = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'JNJ', 'V', 'PG', 'JPM', 'UNH', 'MA', 'TRGP', 'AMC', 'EXAS', 'OXY',
              'LYV', 'JLL', 'FTI', 'OKE', 'SON', 'RRC', 'DVN', 'COTY', 'AR', 'EQT', 'NOV']
#stock_list = ['AAPL', 'AMZN']
# 3. Target Time Frame
start_date = '2016-09-30'
end_date = '2017-09-30'
# Pre-processing
# Loading Phase: Took a while to run this
features_and_return = multi_stock_df(path_to_file, stock_list, start_date, end_date)
multi_stock_return = pd.DataFrame(features_and_return['Return'])
final_df = features_and_return.reset_index()
#final_df['1M Return'] = final_df.apply(lambda row: get_future_return(row, 1), axis=1)
print(final_df.info())
#company_data['AAPL'].info()
# 设置Pandas以显示所有列（根据你的DataFrame的列数）
# Set Pandas to display all columns (None means no limit)
pd.set_option('display.max_columns', None)
# Set Pandas to display all rows (None means no limit)
pd.set_option('display.max_rows', None)
extracted = final_df[final_df['Date'] == '2016-11-30']
# print(final_df[final_df['Date'] == '2016-11-30'].iterrows()
specific_date = extracted.copy()
specific_date['1M Return'] = specific_date.apply(lambda row: get_future_return(row, 1), axis=1)
# specific_date['3M Return'] = specific_date.apply(lambda row: get_future_return(row, 3), axis=1)
# print(specific_date.head())
print(specific_date['1M Return'])
y = specific_date['Return']
specific_date['log pe'] = np.log(specific_date['Price/Earnings'])
specific_date['log pb'] = np.log(specific_date['Price/Book Value'])
specific_date['log roa'] = np.log(specific_date['Return on Assets'])
specific_date['log roe'] = np.log(specific_date['Return on Equity '])
specific_date['log fcfps'] = np.log(specific_date['Free Cash Flow per Share'])
#specific_date['log pcf'] = np.log(specific_date['Price/Cash Flow'])

###### New Signals ######
investment_efficiency = (specific_date['Price/Earnings'] + specific_date['Return on Assets'] + \
                                         specific_date['Return on Equity '])/3
specific_date['Investment Efficiency'] = np.log(investment_efficiency)
print(specific_date['Investment Efficiency'])
# print(specific_date['PE_ROE_Interaction'])
specific_date['PE_ROE_Interaction'] = np.log(specific_date['Price/Earnings'] + 1) * np.sqrt(specific_date['Return on Equity '])
maxval = max(specific_date['PE_ROE_Interaction'])
minval = min(specific_date['PE_ROE_Interaction'])
specific_date['Normalized_PE_ROE_Interaction'] = (specific_date['PE_ROE_Interaction']-minval)/(maxval-minval)

# X = specific_date[['log pe', 'Price/Earnings', 'Price/Book Value', 'log roa', 'Return on Assets',
#                    'log roe', 'Return on Equity ', 'log fcfps', 'Free Cash Flow per Share']]
X = specific_date[['Price/Earnings',  'Price/Book Value', 'Return on Assets',
                    'Return on Equity ', 'Free Cash Flow per Share', 'Dividend Yield (%)',
                    'Enterprise Value/EBIT', 'Enterprise Value/EBITDA',
                   'Dividend Payout Ratio (%)']]
print(X.info())
#X = sm.add_constant(X)

## LASSO try
# define model
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
model = LassoCV(alphas=arange(0, 1, 0.01), cv=cv, n_jobs=-1)
# fit model
model.fit(X, y)
# summarize chosen configuration
print('alpha: %f' % model.alpha_)
exit()



model = sm.OLS(Y, X, missing='drop')
model_result = model.fit()
#model_result = model.fit_regularized(method='elastic_net', alpha=0.1,
                                     #L1_wt=1.0, start_params=None, profile_scale=False, refit=False)
print(model_result.summary())
coefficients = model_result.params
print(coefficients)
exit()
# Predictions
predictions = model.predict(X)
print(predictions)
# Calculate the MSE
mse = np.mean((Y - predictions) ** 2)
print("Mean Squared Error:", mse)




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