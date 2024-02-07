from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
import pandas as pd
import argparse
import datetime
import yfinance as yf
import os

from langchain.prompts import PromptTemplate

def parse_arguments():
    parser = argparse.ArgumentParser(description="langchain prompting")

    parser.add_argument("--start_date", type=str, default="2016-01-31", help="startdate")
    parser.add_argument("--path_to_file", type=str, default="../Data/", help="startdate")
    parser.add_argument("--end_date", type=str, default="2017-12-31", help="enddate")
    parser.add_argument("--featurenames", type=str, default="Price/Earnings,Price/Book Value,Return on Assets,Return on Equity ,Free Cash Flow per Share,Price/Cash Flow", help="feature names")
    parser.add_argument("--method", type=str, default="zero_shot_cot", choices=["zero_shot", "zero_shot_cot"], help="method")
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo-1106", choices=["gpt-3.5-turbo-1106", "gpt-4-1106-preview"], help="model_name")
    args = parser.parse_args()
    args.answer_str = ""
    if args.method == "zero_shot_cot":
        args.answer_str = "Let us think step by step. "
    return args


args = parse_arguments()
# initialize the models
openai = OpenAI(
    #model_name="text-davinci-003",
    model_name = args.model_name,
    openai_api_key="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"
)

feature_defs = {"Price/Earnings":"""Price/Earnings (P/E) Ratio:
- **Definition**: The P/E ratio is a valuation metric that compares a company's current share price to its per-share earnings. It is calculated by dividing the market value per share by the earnings per share (EPS).
- **Effect on predicting stock returns**: The P/E ratio is often used to gauge if a stock is overvalued or undervalued compared to its earnings. A lower P/E might suggest a stock is undervalued, or that investors expect lower earnings growth in the future. Conversely, a higher P/E may imply overvaluation or high expectations for future growth.
- **Preferred tendency**: Investors typically prefer a lower P/E ratio, but this can depend on the industry, growth potential, and other factors. It's also beneficial if the P/E is lower relative to the market or industry average.
""",
"Price/Book Value":"""Price/Book Value (P/B) Ratio:
- **Definition**: The P/B ratio measures a company's market value relative to its book value (the value of its assets minus liabilities). It's calculated by dividing the stock price per share by the book value per share.
- **Effect on predicting stock returns**: The P/B ratio can indicate whether a stock is undervalued or overvalued in terms of its asset base. A lower P/B may imply a stock is undervalued, while a higher P/B might indicate overvaluation.
- **Preferred tendency**: Investors may prefer a lower P/B ratio, which could suggest that a stock is undervalued. However, the interpretation of P/B should be industry-specific, as some industries naturally have higher P/B ratios.
""",
"Return on Assets":"""Return on Assets (ROA):
- **Definition**: ROA is a profitability ratio that shows how efficiently a company uses its assets to generate earnings. It's calculated by dividing net income by total assets.
- **Effect on predicting stock returns**: Higher ROA indicates more efficient use of the company's assets to create profits, which could lead to higher stock returns.
- **Preferred tendency**: A higher ROA is generally preferred, as it suggests the company is generating more earnings from its assets.
""",
"Return on Equity ":"""Return on Equity (ROE):
- **Definition**: ROE measures a corporation's profitability by revealing how much profit a company generates with the money shareholders have invested. It's calculated by dividing net income by shareholder equity.
- **Effect on predicting stock returns**: A higher ROE often reflects a company's ability to use equity investments effectively, which can be attractive to investors and lead to higher stock returns.
- **Preferred tendency**: A higher ROE is usually preferred, indicating that the company is effectively using equity capital to grow profits.
""",
"Free Cash Flow per Share":"""Free Cash Flow per Share (FCF per Share):
- **Definition**: FCF per share is the amount of cash a company generates after accounting for capital expenditures, relative to its total number of outstanding shares.
- **Effect on predicting stock returns**: FCF per share indicates the company's ability to generate cash, which can be used for dividends, share buybacks, or reinvestment for growth. Investors tend to value companies with higher FCF per share.
- **Preferred tendency**: A higher FCF per share is preferred because it suggests the company has healthy cash flow available for distribution or reinvestment into the business.
""",
"Price/Cash Flow":"""Price/Cash Flow (P/CF) Ratio:
- **Definition**: The P/CF ratio is a valuation metric that compares a stock's market price to its cash flow per share. It is calculated by dividing the stock's per-share price by its operating cash flow per share.
- **Effect on predicting stock returns**: This ratio can help assess the value placed on the company's cash flow. A lower P/CF might indicate the stock is undervalued relative to its cash generation capability.
- **Preferred tendency**: Generally, a lower P/CF ratio is preferred, as it suggests that an investor is paying less for each dollar of cash flow generated by the company.
"""}

template1 = """I will give you some financial features like Price/Earnings  Price/Book Value  Return on Assets,   Return on Equity, and   
Free Cash Flow per Share, .

Question: {query}

Answer:  """


template2 = """I will give you some financial information(Price/Earnings  Price/Book Value  Return on Assets,   Return on Equity   
Free Cash Flow per Share  Price/Cash Flow ), including several rows of a financial dataset of multiple companies with 
some features, their expected returns, and their 1-month-later expected returns. Please give me the description of the features".

Context: 
{featuredef}
It's important to note that no single financial metric can provide a cPrice/Earnings  Price/Book Value  Return on Assetsomplete picture of a stock's potential. Investors often use a combination of metrics to analyze a company's financial health, profitability, and potential for returns. Additionally, industry norms and economic context can significantly affect the ideal values for these ratios.

Following are part of the data:
{data}

Question: {query}

Answer: {answer_str}
"""
prompt_template = PromptTemplate(
    input_variables=["featuredef","data","query"],
    template=template2
)

feature_def = ""
for i,f in enumerate(args.featurenames.split(",")):
    feature_def += str(i) + ". " + feature_defs[f] + "\n"

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
def load_data(args,stock_list):
    data_str = ""
    for ticker in stock_list:
        print(ticker)
        file_path = args.path_to_file + ticker + '.xlsx'
        sheet_name = ticker + '-US'
        data = pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl')
        data = data.reset_index(drop=True)
        data = data.set_index('Date').T
        data.index = pd.to_datetime(data.index, format='%b \'%y')
        data.index = data.index + pd.offsets.MonthEnd()
        ratio_data = data.apply(pd.to_numeric)
        for col in args.featurenames.split(","):
             if col not in ratio_data.columns:
                 ratio_data.loc[:, col] = 0
        ratio_data = ratio_data[args.featurenames.split(",")]
        ratio_data = ratio_data.apply(lambda x: x.fillna(x.mean()), axis=0)
        returns_data = get_historical_returns(ticker, args.start_date, args.end_date)
        adjusted_ratio_data = resample_quaterly_data(ratio_data, returns_data)
        features = pd.concat([adjusted_ratio_data, returns_data],axis=1)
        data_str += "data for company: " + ticker +"\n"+features.to_string() +"\n\n"
    return data_str
stock_list = ['AAPL','MSFT', 'AMZN', 'GOOGL', 'JNJ', 'V', 'PG', 'JPM']#, 'UNH', 'MA', 'LYV', 'JLL','TRGP','FTI', 'OKE', 'SON',  'DVN']
data_str = load_data(args,stock_list)

q1 = "Please give me the definitions of the provided features, explain their effect in predicting stock returns and the preferred tendency of the features"

q2 = "Please create a new nonlinear feature based on the provided context (existing features), and this" \
              "new feature should be correlated to the returns," \
              "explain how you created this feature and describe the meaning of this new feature. Note that don't" \
              "provide simple linear combination of other existing features" \
              "Please also provide the calculated values of this new feature" \
              "and normalize them in the range from 0 to 1"

result = openai(
    prompt_template.format(
        query = q2,featuredef=feature_def,data=data_str,answer_str=args.answer_str
        # query = "can you find the function that inputs are existing features and output is the return?"
    ))
output = {
        'template': [template2],
        'start_date':[args.start_date],
        'end_date':[args.start_date],
        'featurenames':[args.featurenames],
        'stocks':[stock_list],
        'answer':[result]
        }
print(result)

outfilepath = './output/'+args.model_name+'/'+args.method + '/'
file_name = f"out_{args.model_name}_{args.method}.csv"
filepath = os.path.join(outfilepath, file_name)
if not os.path.exists(outfilepath):
    os.makedirs(outfilepath)
if os.path.exists(filepath):
    time_tag = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filepath = filepath.replace(".csv", "_{}.csv".format(time_tag))

df = pd.DataFrame.from_dict(output)
df.to_csv(filepath)

print("output saved to {}".format(filepath))