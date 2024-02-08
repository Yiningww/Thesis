from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
import pandas as pd
import argparse
import datetime
import yfinance as yf
import os
import numpy as np

from langchain.prompts import PromptTemplate

def parse_arguments():
    parser = argparse.ArgumentParser(description="langchain prompting")

    parser.add_argument("--start_date", type=str, default="2016-01-31", help="startdate")
    parser.add_argument("--path_to_file", type=str, default="../Data/", help="startdate")
    parser.add_argument("--end_date", type=str, default="2017-12-31", help="enddate")
    parser.add_argument("--feature_names", type=str, default="Price/Earnings,Price/Book Value,Return on Assets,Return on Equity ,Free Cash Flow per Share,Price/Cash Flow,Dividend Yield (%),Enterprise Value/EBIT,Enterprise Value/EBITDA,Dividend Payout Ratio (%)", help="feature_names")
    parser.add_argument("--method", type=str, default="zero_shot_cot", choices=["zero_shot", "zero_shot_cot"], help="method")
    parser.add_argument("--model_name", type=str, default="gpt-4-1106-preview", choices=["gpt-3.5-turbo-1106", "gpt-4-1106-preview"], help="model_name")
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
""",
"Dividend Yield (%)":"""Dividend Yield (%):
Definition: Dividend Yield is a financial ratio that shows how much a company pays out in dividends each year relative to its stock price. It's calculated as the annual dividends per share divided by the price per share.
Effect in Predicting Stock Returns: A higher dividend yield can be attractive to investors looking for income from their investments, indicating the company returns a significant portion of its earnings to shareholders. However, an exceptionally high yield can also signal a company in distress, where the stock price has fallen due to underlying problems.
Preferred Tendency: A moderate to high dividend yield that is sustainable over time is generally preferred, but it should be evaluated within the context of the company's overall financial health and industry standards.
""",
"Enterprise Value/EBIT":"""Enterprise Value/EBIT:
Definition: This ratio compares the total value of the company (including debt and excluding cash) to its earnings before interest and taxes (EBIT). It gives an indication of how expensive the company is relative to its earnings capacity before the effect of capital structure and tax policies.
Effect in Predicting Stock Returns: A lower EV/EBIT ratio may suggest that the company is undervalued relative to its earnings power, potentially offering higher returns to investors. It's useful for comparing companies across different industries or regions.
Preferred Tendency: Lower values are generally preferred as they indicate the company might be undervalued.
""",
"Enterprise Value/EBITDA":"""Enterprise Value/EBITDA:
Definition: Similar to EV/EBIT, this ratio uses EBITDA (Earnings Before Interest, Taxes, Depreciation, and Amortization) as a proxy for the company's operating cash flow. It compares the company's total valuation to its cash earnings, excluding the non-cash expenses.
Effect in Predicting Stock Returns: A lower EV/EBITDA ratio may indicate a company is undervalued relative to its operational cash earnings, potentially offering a good buying opportunity. This metric is widely used for valuation comparisons, especially for capital-intensive or highly leveraged companies.
Preferred Tendency: Investors generally prefer a lower EV/EBITDA ratio, indicating the company is cheaper relative to its operational profitability.
""",
"Dividend Payout Ratio (%)":"""Dividend Payout Ratio (%):
Definition: The Dividend Payout Ratio measures the percentage of earnings paid to shareholders in dividends. It's calculated by dividing the total dividends paid by the company's net income.
Effect in Predicting Stock Returns: This ratio helps investors understand how much money a company is returning to shareholders versus reinvesting back into the business. A very high payout ratio may not be sustainable in the long term, whereas a too-low ratio might suggest the company is not returning enough value to shareholders.
Preferred Tendency: A balanced dividend payout ratio is preferred; it should be sustainable based on the company's earnings, allow for growth reinvestment, and align with the company's overall financial strategy.
"""}

template1 = """I will give you some financial features like Price/Earnings  Price/Book Value  Return on Assets,   Return on Equity, and   
Free Cash Flow per Share, .

Question: {query}

Answer:  """


template2 = """I will give you some financial information, including several rows of a financial dataset of multiple companies with 
some features(included in the context) and their expected returns. I will also give you the descriptions of these features".

Context: 
{featuredef}
It's important to note that no single financial metric can provide a complete picture of a stock's potential. Investors often use a combination of metrics to analyze a company's financial health, profitability, and potential for returns. Additionally, industry norms and economic context can significantly affect the ideal values for these ratios.

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
for i,f in enumerate(args.feature_names.split(",")):
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
        for col in args.feature_names.split(","):
             if col not in ratio_data.columns:
                 ratio_data.loc[:, col] = np.nan
        ratio_data = ratio_data[args.feature_names.split(",")]
        ratio_data = ratio_data.apply(lambda x: x.fillna(x.mean()), axis=0)
        returns_data = get_historical_returns(ticker, args.start_date, args.end_date)
        adjusted_ratio_data = resample_quaterly_data(ratio_data, returns_data)
        features = pd.concat([adjusted_ratio_data, returns_data],axis=1)
        data_str += "data for company: " + ticker +"\n"+features.to_string() +"\n\n"
    return data_str
stock_list = ['AAPL','MSFT', 'AMZN', 'GOOGL', 'JNJ', 'V', 'PG', 'JPM', 'UNH', 'MA', 'LYV', 'JLL','TRGP','FTI', 'OKE', 'SON', 'DVN']
data_str = load_data(args,stock_list)

q1 = "Please give me the definitions of the provided features, explain their effect in predicting stock returns and the preferred tendency of the features"

q2 = "Please create a new nonlinear feature based on the provided context (existing features), and this" \
              "new feature should be correlated to the returns," \
              "explain how you created this feature and describe the meaning of this new feature. Note that don't" \
              "provide simple linear combination of other existing features and focus on as many meaningful existing features as possible." \
              "Please also provide the calculated values of this new feature" \
              "and standardize them"

result = openai(
    prompt_template.format(
        query = q2,featuredef=feature_def,data=data_str,answer_str=args.answer_str
        # query = "can you find the function that inputs are existing features and output is the return?"
    ))
output = {
        'template': [template2],
        'start_date':[args.start_date],
        'end_date':[args.start_date],
        'feature_names':[args.feature_names],
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