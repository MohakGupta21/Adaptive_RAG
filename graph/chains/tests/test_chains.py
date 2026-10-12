from dotenv import load_dotenv
from pprint import pprint
load_dotenv()
from graph.chains import answer_grader
from graph.chains.answer_grader import AnswerGrader, answer_grader
from graph.chains.generation import generation_chain
from graph.chains.hallucination_grader import HallucinationGrader, hallucination_chain
from graph.chains.retriever_grader import GradeDocuments, retrieval_grader
from ingestion import retriever
from graph.chains.router import RouteQuery, question_router

def test_retrieval_grader_answer_yes()->None:
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content

    res:GradeDocuments = retrieval_grader.invoke(
        {"document":doc_txt, "question":question}
    )

    assert res.binary_score == "yes"

def test_retrieval_grader_answer_no()->None:
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"question":"how to make pizza", "document":doc_txt}
    )
    assert res.binary_score == "no"

def test_generation_chain() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({"context":docs,"question":question})
    pprint(generation)    

def test_hallucination_grader_yes()-> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({"question":question,"context":docs})
    hallucination_response:HallucinationGrader = hallucination_chain.invoke({"documents":docs,"generation":generation})
    assert hallucination_response.binary_score

def test_hallucination_grader_no()-> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    hallucination_response:HallucinationGrader = hallucination_chain.invoke({"documents":docs,"generation":"In order to make pizza, find a good dough"})
    assert not hallucination_response.binary_score

def test_answer_grader_yes()->None:
    question = "agent memory"
    docs = retriever.invoke(question)
    generated_ans = generation_chain.invoke({"question":question,"context":docs})
    answer_grader_res:AnswerGrader = answer_grader.invoke({"question":question,"generation":generated_ans})
    assert answer_grader_res.binary_score

def test_answer_grader_no()-> None:
    question = "agent memory"
    answer_grader_res:AnswerGrader = answer_grader.invoke({"question":question,"generation":"How to make a pizza is really simple."})
    assert not answer_grader_res.binary_score

def test_router_to_vectorstore()-> None:
    question ="agent memory"
    res:RouteQuery = question_router.invoke({"question":question})
    assert res.datasource == "vectorstore"

def test_router_to_websearch()-> None:
    question ="how to make a pizza"
    res:RouteQuery = question_router.invoke({"question":question})
    assert res.datasource == "websearch"