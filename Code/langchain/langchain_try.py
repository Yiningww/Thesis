import openai
import os
import dotenv
from dotenv import load_dotenv, find_dotenv
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
os.environ['OPENAI_API_KEY'] = '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'
'''
llm = OpenAI()
chat_model = ChatOpenAI()
llm.predict("hi!")
'''
_ = load_dotenv(find_dotenv()) # read local .env file
#openai.api_key = os.environ['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']
openai.api_key = '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'
# account for deprecation of LLM model
import datetime
# Get the current date
current_date = datetime.datetime.now().date()

# Define the date after which the model should be set to "gpt-3.5-turbo"
target_date = datetime.date(2024, 6, 12)

# Set the model variable based on the current date
if current_date > target_date:
    llm_model = "gpt-3.5-turbo"
else:
    llm_model = "gpt-3.5-turbo-0301"

def get_completion(prompt, model=llm_model):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message["content"]

#print(get_completion("What is 1+1?"))
chat = ChatOpenAI(temperature=0.0, model=llm_model)
template_string = """Translate the text \
that is delimited by triple backticks \
into a style that is {style}. \
text: ```{text}```
"""
from langchain.prompts import ChatPromptTemplate

prompt = """The following are exerpts from conversations with an AI
assistant. The assistant is typically sarcastic and witty, producing
creative  and funny responses to the users questions. Here are some
examples: 

User: How are you?
AI: I can't complain but sometimes I still do.

User: What time is it?
AI: It's time to get a watch.

User: What is the meaning of life?
AI: """

print(openai(prompt))









'''
prompt_template = ChatPromptTemplate.from_template(template_string)
prompt_template.messages[0].prompt
prompt_template.messages[0].prompt.input_variables
customer_style = """American English \
in a calm and respectful tone
"""
customer_email = """
Arrr, I be fuming that me blender lid \
flew off and splattered me kitchen walls \
with smoothie! And to make matters worse, \
the warranty don't cover the cost of \
cleaning up me kitchen. I need yer help \
right now, matey!
"""
customer_messages = prompt_template.format_messages(
                    style=customer_style,
                    text=customer_email)
print(type(customer_messages))
print(type(customer_messages[0]))
print(customer_messages[0])
# Call the LLM to translate to the style of the customer message
customer_response = chat(customer_messages)
print(customer_response.content)

model = ChatOpenAI()
prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")
chain = prompt | model
##Stream
#for s in chain.stream({"topic": "bears"}):
    #print(s.content, end="", flush=True)
##Invoke
print(chain.invoke({"topic": "bears"}))
#AIMessage(content="Why don't bears wear shoes?\n\nBecause they already have bear feet!")
chain.batch([{"topic": "bears"}, {"topic": "cats"}])
'''
