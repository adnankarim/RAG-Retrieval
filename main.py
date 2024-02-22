# import timeit
# import argparse
from llm.wrapper import query_embeddings,set_qa_prompt
from llm.llm import setup_llm
from langchain.chains import RetrievalQA
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from uvicorn import run
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import box
import yaml
from fastapi.middleware.cors import CORSMiddleware

# Import config vars
with open('config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
def build_retrieval_qa_chain(llm, prompt):
    embeddings = HuggingFaceEmbeddings(model_name=cfg.EMBEDDINGS,
                                       model_kwargs={'device': 'cuda'})
    vectordb = FAISS.load_local(cfg.DB_FAISS_PATH, embeddings)
    retriever = vectordb.as_retriever(search_kwargs={'k': cfg.VECTOR_COUNT})

    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                           chain_type='stuff',
                                           retriever=retriever,
                                           return_source_documents=cfg.RETURN_SOURCE_DOCUMENTS,
                                           chain_type_kwargs={'prompt': prompt})
    return qa_chain





@app.post("/chat")
async def chatMe(request: Request):
    data= await request.json()
    question=data['question']
    llm = setup_llm()
    qa_prompt = set_qa_prompt()
    qa_chain = build_retrieval_qa_chain(llm, qa_prompt)
    response = qa_chain({'query': question})
    answer=response["result"]
   
    
    return JSONResponse(content={"answer": answer   })

if __name__ == "__main__":
    run(app, host="127.0.0.1", port=8000)
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--input',
    #                     type=str,
    #                     default="Explain NLIs for Visual Data Analysis in 200 words",
    #                     help='Enter the query to pass into the LLM')
    # parser.add_argument('--semantic_search',
    #                     type=bool,
    #                     default=False,
    #                     help='Enter True if you want to run semantic search, else False')
    # args = parser.parse_args()

    # start = timeit.default_timer()
    # if args.semantic_search:
    #     semantic_search = query_embeddings(args.input)
    #     print(f'Semantic search: {semantic_search}')
    #     print('='*50)
    # else:
    #     qa_chain = setup_qa_chain()
    #     response = qa_chain({'query': args.input})
    #     print(f'\nAnswer: {response["result"]}')
    #     print('=' * 50)
    # end = timeit.default_timer()

    # print(f"Time to retrieve answer: {end - start}")
