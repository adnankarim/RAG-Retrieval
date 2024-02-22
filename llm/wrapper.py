import box
import yaml

from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from llm.prompts import qa_template


# Import config vars
with open('config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))


def set_qa_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """

    prompt = PromptTemplate(template=qa_template,
                            input_variables=['context', 'question'])

    return prompt




def query_embeddings(query):
    embeddings = HuggingFaceEmbeddings(model_name=cfg.EMBEDDINGS,
                                       model_kwargs={'device': 'cuda'})
    vectordb = FAISS.load_local(cfg.DB_FAISS_PATH, embeddings)
    retriever = vectordb.as_retriever(search_kwargs={'k': cfg.VECTOR_COUNT})
    semantic_search = retriever.get_relevant_documents(query)

    return semantic_search