from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()

class AnswerGrader(BaseModel):
    binary_score:bool= Field("Checks whether the answer is correct or not")

system = """You are an answer grader agent. You check whether the generated answer addresses the question or is totally irrelevant to the question asked.
Answer 'yes' if it is relevant and 'no' if it is irrelevant.
"""

chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",system),
        ("human","User question: \n\n {question} \n\n LLM generation:{generation}"),
    ]
)

llm = ChatOpenAI(temperature=0)
structured_llm_grader = llm.with_structured_output(AnswerGrader)

answer_grader = chat_prompt | structured_llm_grader

