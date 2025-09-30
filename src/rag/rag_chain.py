from src.base.llm_model import get_llama3_model, get_gemini_model
from src.rag.main import build_rag_chain
from src.chat.main import build_chat_chain
from src import config
# llm = get_llama3_model(model_name="llama3-70b-8192",temperature=1)

llm = get_llama3_model(model_name="meta-llama/llama-4-maverick-17b-128e-instruct",temperature=1)
# llm = get_gemini_model()
genai_docs = config.Data_health_path
ml_docs = config.Data_orther_path

# --------- Chains----------------
health_chain = build_rag_chain(llm, data_dir=genai_docs, data_type="csv", data_embedded="Yes",disease_list_file=config.disease_list_file)
# orther_chain = build_rag_chain(llm, data_dir=ml_docs, data_type="html")

chat_chain = build_chat_chain(llm, 
                              history_folder="./chat_histories",
                              max_history_length=6)

def get_answer_rag_chain(question):
    answer = health_chain.invoke(question)
    return answer