from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
prompt = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""

chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",prompt)
    ]
)

generation_chain = chat_prompt | llm | StrOutputParser()

# if __name__=="__main__":
