from dotenv import load_dotenv
load_dotenv()
from graph_compiled import compiled

if __name__=="__main__":
    answer=compiled.invoke({
        "question":"How is a pizza made?"
    })
    print(answer["generation"])