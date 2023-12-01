from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
# initialize the models
openai = OpenAI(
    model_name="text-davinci-003",
    openai_api_key="KEY"
)
from langchain.prompts import PromptTemplate

template = """I will give you some financial information, including several rows of a financial dataset of GOOG with 
5 features and respected closings. I will also give you the description of the features".

Context: Price/Earnings (P/E) Ratio: Indicates how much investors are willing to pay per dollar of earnings; higher values suggest expectations of future growth.

Price/Book (P/B) Value Ratio: Compares a company's market value to its book value; helps assess if a stock is over or undervalued.

Return on Assets (ROA): Shows how efficiently a company uses its assets to generate earnings; higher ROA means better asset efficiency.

Return on Equity (ROE): Measures profitability relative to shareholder equity; higher ROE indicates effective use of investment capital.

Free Cash Flow per Share (FCF per Share): Reflects the amount of cash available to shareholders on a per-share basis; higher values suggest better cash generation and potential for growth.

5-row sample data of AAPL: 
            Price/Earnings  Price/Book Value  Return on Assets  Return on Equity   Free Cash Flow per Share     Close
Date                                                                                                                 
2017-09-30       32.049390          4.241759         12.026302          14.433230                  1.725199  0.021058
2017-12-31       58.262806          4.767289          6.942038           8.686346                  1.720909  0.024466
2018-03-31       43.794143          4.458494          8.763442          10.881893                  1.503969 -0.066025
2018-06-30       48.317453          4.792793          8.358126          10.511593                  1.513906  0.028258
2018-09-30       44.901053          4.890509          9.131203          11.481006                  1.625823 -0.020292

Question: {query}

Answer: """
prompt_template = PromptTemplate(
    input_variables=["query"],
    template=template
)
print(openai(
    prompt_template.format(
        query="Please create a new financial ratio (feature) based on the provided context, explain how you created this"
              "ratio and describe the meaning of this new feature. Please also provide the calculated values of this new feature"
              "and normalize them in the range from 0 to 1"
    )
))
