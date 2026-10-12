from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
load_dotenv()


class HallucinationGrader(BaseModel):
    binary_score:bool = Field("Test whether the answer is correct or not")

llm = ChatOpenAI(model="gpt-4o-mini")
structured_llm_output = llm.with_structured_output(HallucinationGrader)

system= """
You are a grader assessing whether an LLM generation is grounded in / supported by a score.
Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the documents.
'No' means it is not supported.
"""

hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",system),
        ("human","Set of Facts:\n\n{documents} \n\n LLM generation:{generation}")
    ]
)

hallucination_chain = hallucination_prompt | structured_llm_output
