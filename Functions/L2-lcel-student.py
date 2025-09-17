#!/usr/bin/env python
# coding: utf-8

# # LangChain Expression Language (LCEL)

# # LangChain Expression Language (LCEL)

# # LangChain Expression Language (LCEL)

# In[1]:


import os
import openai

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']


# In[2]:


#!pip install pydantic==1.10.8


# In[3]:


from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser


# ## Simple Chain

# In[4]:


prompt = ChatPromptTemplate.from_template(
    "tell me a short joke about {topic}"
)
model = ChatOpenAI()
output_parser = StrOutputParser()


# In[5]:


chain = prompt | model | output_parser


# In[13]:


chain.invoke({"topic": "aspirapolvere"})


# ## More complex chain
# 
# And Runnable Map to supply user-provided inputs to the prompt.

# In[14]:


from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import DocArrayInMemorySearch


# In[15]:


vectorstore = DocArrayInMemorySearch.from_texts(
    ["harrison worked at kensho", "bears like to eat honey"],
    embedding=OpenAIEmbeddings()
)
retriever = vectorstore.as_retriever()


# In[16]:


retriever.get_relevant_documents("where did harrison work?")


# In[17]:


retriever.get_relevant_documents("what do bears like to eat")


# In[18]:


template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)


# In[19]:


from langchain.schema.runnable import RunnableMap


# In[20]:


chain = RunnableMap({
    "context": lambda x: retriever.get_relevant_documents(x["question"]),
    "question": lambda x: x["question"]
}) | prompt | model | output_parser


# In[21]:


chain.invoke({"question": "where did harrison work?"})


# In[22]:


inputs = RunnableMap({
    "context": lambda x: retriever.get_relevant_documents(x["question"]),
    "question": lambda x: x["question"]
})


# In[23]:


inputs.invoke({"question": "where did harrison work?"})


# ## Bind
# 
# and OpenAI Functions

# In[24]:


functions = [
    {
      "name": "weather_search",
      "description": "Search for weather given an airport code",
      "parameters": {
        "type": "object",
        "properties": {
          "airport_code": {
            "type": "string",
            "description": "The airport code to get the weather for"
          },
        },
        "required": ["airport_code"]
      }
    }
  ]


# In[25]:


prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}")
    ]
)
model = ChatOpenAI(temperature=0).bind(functions=functions)


# In[26]:


runnable = prompt | model


# In[27]:


runnable.invoke({"input": "what is the weather in sf"})


# In[28]:


functions = [
    {
      "name": "weather_search",
      "description": "Search for weather given an airport code",
      "parameters": {
        "type": "object",
        "properties": {
          "airport_code": {
            "type": "string",
            "description": "The airport code to get the weather for"
          },
        },
        "required": ["airport_code"]
      }
    },
        {
      "name": "sports_search",
      "description": "Search for news of recent sport events",
      "parameters": {
        "type": "object",
        "properties": {
          "team_name": {
            "type": "string",
            "description": "The sports team to search for"
          },
        },
        "required": ["team_name"]
      }
    }
  ]


# In[29]:


model = model.bind(functions=functions)


# In[30]:


runnable = prompt | model


# In[31]:


runnable.invoke({"input": "how did the patriots do yesterday?"})


# ## Fallbacks

# In[32]:


from langchain.llms import OpenAI
import json


# **Note**: Due to the deprication of OpenAI's model `text-davinci-001` on 4 January 2024, you'll be using OpenAI's recommended replacement model `gpt-3.5-turbo-instruct` instead.

# In[33]:


simple_model = OpenAI(
    temperature=0, 
    max_tokens=1000, 
    model="gpt-3.5-turbo-instruct"
)
simple_chain = simple_model | json.loads


# In[34]:


challenge = "write three poems in a json blob, where each poem is a json blob of a title, author, and first line"


# In[35]:


simple_model.invoke(challenge)


# <p style=\"background-color:#F5C780; padding:15px\"><b>Note:</b> The next line is expected to fail.</p>

# In[36]:


simple_chain.invoke(challenge)


# In[37]:


model = ChatOpenAI(temperature=0)
chain = model | StrOutputParser() | json.loads


# In[38]:


chain.invoke(challenge)


# In[39]:


final_chain = simple_chain.with_fallbacks([chain])


# In[40]:


final_chain.invoke(challenge)


# ## Interface

# In[41]:


prompt = ChatPromptTemplate.from_template(
    "Tell me a short joke about {topic}"
)
model = ChatOpenAI()
output_parser = StrOutputParser()

chain = prompt | model | output_parser


# In[42]:


chain.invoke({"topic": "bears"})


# In[43]:


chain.batch([{"topic": "bears"}, {"topic": "frogs"}])


# In[44]:


for t in chain.stream({"topic": "bears"}):
    print(t)


# In[45]:


response = await chain.ainvoke({"topic": "bears"})
response


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




