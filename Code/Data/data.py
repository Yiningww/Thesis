import pandas as pd
import numpy as np
import yfinance as yf
import os
from datetime import datetime
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

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
    # aligned_quaterly_data = quaterly_data.reindex(target_data_after_1_month.index, method='ffill')

    aligned_quaterly_data = aligned_quaterly_data.dropna()
    return aligned_quaterly_data


def load_features(path_to_file, ticker, start_date, end_date):
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
    # dy_column = 'Dividend Yield (%)'
    # ebit_column = 'Enterprise Value/EBIT'
    # ebitda_column = 'Enterprise Value/EBITDA'
    # divpr_column = 'Dividend Payout Ratio (%)'
    # inv_turnover = 'Inventory Turnover'
    # asset_turnover = 'Asset Turnover'
    # cash_ratio = 'Cash Ratio'
    # de_column = 'Total Debt/Equity (%)'
    ratio_data = data[[pe_column, pb_column, roa_column, roe_column, fcf_column]]
    # ratio_data = data[[pe_column, pb_column, roa_column, roe_column, fcf_column, pcf_column, dy_column, ebit_column,
    #                    ebit_column, divpr_column]]
    # Replace N/A dates
    #print("Before N/A:", ratio_data)
    
    for column in ratio_data.select_dtypes(include=[np.number]).columns:
        ratio_data[column] = ratio_data[column].fillna(ratio_data[column].mean())
    # Process Return Data
    returns_data = get_historical_returns(ticker, start_date, end_date)
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
        company_data[ticker] = load_features(path_to_file, ticker, start_date, end_date)

    # Initialize a list to hold DataFrames with the new multi-index
    multi_index_dfs = []
    for company, df in company_data.items():
        # Set the company name as an additional level in the index
        df_multi_index = df.copy()
        df_multi_index['Company'] = company
        df_multi_index.set_index(['Company', df_multi_index.index], inplace=True)
        # Append to the list
        multi_index_dfs.append(df_multi_index)
    # Concatenate all DataFrames into a single multi-index DataFrame
    final_df = pd.concat(multi_index_dfs)
    # print(final_df)
    # for column in final_df.select_dtypes(include=[np.number]).columns:
    #     final_df[column] = final_df[column].fillna(final_df[column].mean())
    return final_df

def create_sequences(features, targets, seq_length):
    'Function to create sequence'
    'Need to define the sequence length: e.g. using 4 quaters to predict the next quater'

    xs = []
    ys = []

    for i in range(len(features)-seq_length):
        x = features[i:(i+seq_length)]
        y = targets.iloc[i+seq_length]
        xs.append(x)
        ys.append(y)

    return np.array(xs), np.array(ys)

def get_future_return(row, months):
    # 获取结束日期
    end_date = (row['Date'] + pd.DateOffset(months=months)).date()
    end_date = pd.to_datetime(end_date) + pd.offsets.MonthEnd(0)
    # 使用提供的函数获取历史回报率
    historical_returns = get_historical_returns(row['Company'], row['Date'], end_date, frequency="monthly")
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


def load_features_from_file(file_path):
    # Load the Excel file and get the features
    data = pd.read_excel(file_path, engine='openpyxl')
    # Assuming the features are in the first column after transposing
    features = data.T.iloc[:, 0]
    return features

def find_common_features(directory_path):
    # Get a list of all Excel files in the directory
    file_paths = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith('.xlsx')]
    # Initialize a set with the features from the first file
    initial_features = load_features_from_file(file_paths[0])
    print(initial_features)
    common_features = set(initial_features)

    # Loop through the remaining files and find the common features
    for file_path in file_paths[1:]:
        features = load_features_from_file(file_path)
        common_features.intersection_update(features)

    return common_features

# Specify the directory where your Excel files are stored
# directory_path = "/Users/wangyining/Desktop/Thesis/Code/Data/"
#
# # Call the function to find common features
# common_features = find_common_features(directory_path)
# print(common_features)
# exit()
# Define Input and Parameters
# 1. File Path
path_to_file = "/Users/wangyining/Desktop/Thesis/Code/Data/"
# 2. Ticker List
#ChatGPT_stocks = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'JNJ', 'V', 'PG', 'JPM', 'UNH', 'MA', 'INTC', 'VZ', 'GOOG', 'HD', 'T']
#stock_list = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'JNJ', 'V', 'PG', 'JPM', 'UNH', 'MA']
stock_list = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'JNJ', 'V', 'PG', 'JPM', 'UNH', 'MA', 'TRGP', 'AMC', 'EXAS', 'OXY',
              'LYV', 'JLL', 'FTI', 'OKE', 'SON', 'RRC', 'DVN', 'COTY', 'AR', 'EQT', 'NOV']
# 3. Target Time Frame
start_date = '2016-09-30'
end_date = '2017-09-30'

# Pre-processing
# Loading Phase: Took a while to run this
features_and_return = multi_stock_df(path_to_file, stock_list, start_date, end_date)
multi_stock_return = pd.DataFrame(features_and_return['Return'])
final_df = features_and_return.reset_index()
print(final_df)
final_df['1M Return'] = final_df.apply(lambda row: get_future_return(row, 1), axis=1)
print(final_df)
exit()

extracted = final_df[final_df['Date'] == '2016-11-30']
# print(final_df[final_df['Date'] == '2016-11-30'].iterrows()
specific_date = extracted.copy()
specific_date['1M Return'] = specific_date.apply(lambda row: get_future_return(row, 1), axis=1)
#specific_date['3M Return'] = specific_date.apply(lambda row: get_future_return(row, 3), axis=1)
print(specific_date)
# print(specific_date.head())

import statsmodels.api as sm
Y = specific_date['Return']
specific_date['log pe'] = np.log(specific_date['Price/Earnings'])
specific_date['log pb'] = np.log(specific_date['Price/Book Value'])
specific_date['log roa'] = np.log(specific_date['Return on Assets'])
specific_date['log roe'] = np.log(specific_date['Return on Equity '])
specific_date['log fcfps'] = np.log(specific_date['Free Cash Flow per Share'])
### New Signals
investment_efficiency = (specific_date['Price/Earnings'] + specific_date['Return on Assets'] + \
                                         specific_date['Return on Equity '])/3
specific_date['Investment Efficiency'] = np.log(investment_efficiency)

# X = specific_date[['log pe', 'Price/Earnings', 'Price/Book Value', 'log roa', 'Return on Assets',
#                    'log roe', 'Return on Equity ', 'log fcfps', 'Free Cash Flow per Share']]
X = specific_date[['log pe',  'log pb', 'log roa',
                   'log roe',  'log fcfps', 'Investment Efficiency']]
X = sm.add_constant(X)
#print(X.head())
model = sm.OLS(Y, X, missing='drop')
model_result = model.fit()
print(model_result.summary())



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