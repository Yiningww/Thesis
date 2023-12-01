import pandas as pd
import numpy as np
import yfinance as yf
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

    # Resample the quaterly data to daily frequency using Forward Fill
    quaterly_data.index = quaterly_data.index + pd.DateOffset(days=1)
    aligned_quaterly_data = quaterly_data.reindex(target_data.index, method='ffill')

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
    dy_column = 'Dividend Yield (%)'
    ebit_column = 'Enterprise Value/EBIT'
    ebit_column = 'Enterprise Value/EBITDA'
    divpr_column = 'Dividend Payout Ratio (%)'
    inv_turnover = 'Inventory Turnover'
    asset_turnover = 'Asset Turnover'
    cash_ratio = 'Cash Ratio'
    de_column = 'Total Debt/Equity (%)'




    ratio_data = data[[pe_column, pb_column, roa_column, roe_column, fcf_column, pcf_column, dy_column, ebit_column,
                       ebit_column, divpr_column]]

    #print(ratio_data)

    # Replace N/A dates
    for column in ratio_data.select_dtypes(include=[np.number]).columns:
        #ratio_data.loc[:, column] = ratio_data.loc[:, column].fillna(ratio_data[column].mean())
        ratio_data[column] = ratio_data[column].fillna(ratio_data[column].mean())
    #print(ratio_data)
    #ratio_data = ratio_data.dropna()

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

# Define Input and Parameters
# 1. File Path
path_to_file = "/Users/wangyining/Desktop/Thesis/Code/Data/"
# 2. Ticker List
#ChatGPT_stocks = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'JNJ', 'V', 'PG', 'JPM', 'UNH', 'MA', 'INTC', 'VZ', 'GOOG', 'HD', 'T']
#stock_list = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'JNJ', 'V', 'PG', 'JPM', 'UNH', 'MA']
stock_list = ['AAPL', 'MSFT']
# 3. Target Time Frame
start_date = '2016-09-30'
end_date = '2017-09-30'

# Pre-processing
# Loading Phase: Took a while to run this
final_df = multi_stock_df(path_to_file, stock_list, start_date, end_date)
multi_stock_df = final_df
multi_stock_return = pd.DataFrame(multi_stock_df['Return'])
features_and_return = multi_stock_df
print("AFTER NA:", features_and_return.head(20).to_string())


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