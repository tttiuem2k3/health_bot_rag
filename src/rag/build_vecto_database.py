import os
import pandas as pd
from typing import List
from tqdm import tqdm
import torch
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from pyvi import ViTokenizer


# Hàm tiền xử lý văn bản tiếng Việt
def tokenize_text(text: str):
    return ViTokenizer.tokenize(text.strip())

# ========================
# 1. CSV Loader có tiền xử lý
# ========================
class CSVLoader():
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, csv_files: List[str], **kwargs):
        documents = []
        for csv_file in csv_files:
            data = pd.read_csv(csv_file)
            data = data.fillna("")

            for _, row in data.iterrows():
                # Tiền xử lý văn bản
                processed_content = tokenize_text(row["Document"].strip())

                document = Document(
                    page_content=processed_content,
                    metadata={
                        "title": row["Title"].strip().lower(),
                        "source": row["Source"].strip(),
                    }
                )
                documents.append(document)
        print(f"📄 Số lượng document: {len(documents)}")
        return documents
# ========================
# 2. Tiền xử lý và chia đoạn
# ========================
def preprocess_documents(csv_file: str, chunk_size: int = 1024, chunk_overlap: int = 200):
    loader = CSVLoader()
    documents = loader([csv_file])

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    doc_split = splitter.split_documents(documents)
    print(f"✂️ Số lượng chunk sau khi chia: {len(doc_split)}")
    # In ra số lượng từ trong 5 chunk đầu tiên
    print("\n📊 Thống kê số lượng từ trong một số chunk:")
    for i, doc in enumerate(doc_split[:5]):
        word_count = len(doc.page_content.split())
        print(f"Chunk {i+1}: {word_count} từ — Metadata: {doc.metadata}")
        print(f"Nội dung: {doc.page_content[:100]}...\n")
    return doc_split

# ========================
# 3. Embedding + Lưu vào Chroma
# ========================
def build_chroma_index(documents: List[Document], embedding_model: str, persist_directory: str):
    # Sử dụng GPU nếu có
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"💻 Thiết bị đang dùng: {device}")
    print(f"Sample document: {documents[:1]}")
    embedding = HuggingFaceEmbeddings(model_name=embedding_model, model_kwargs={"device": device})

    print("🔍 Đang embed và lưu vào Chroma...")
    vectordb = Chroma.from_documents(
        documents=tqdm(documents, desc="⚙️ Đang xử lý embedding"),
        embedding=embedding,
        persist_directory=persist_directory
    )
    vectordb.persist()
    print(f"✅ Đã lưu index tại: {persist_directory}")
    return vectordb

# ========================
# 4. Load lại DB
# ========================
def load_chroma_index(embedding_model: str, persist_directory: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding = HuggingFaceEmbeddings(model_name=embedding_model, model_kwargs={"device": device})
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
    print("✅ Đã load lại index từ disk.")
    return vectordb


# ========================
# 5. Main
# ========================
Data_health_path ="./data_source/health_bot"
csv_path = Data_health_path + "/disease_cleaned_final.csv"  # Thay bằng file CSV của bạn
persist_dir = Data_health_path + "/chroma_db1"
embedding_model = "bkai-foundation-models/vietnamese-bi-encoder"

doc_chunks = preprocess_documents(csv_path)
build_chroma_index(doc_chunks, embedding_model, persist_dir)


from langchain_community.vectorstores import Chroma
from pyvi import ViTokenizer

def preprocess_query(query: str) -> str:
    return ViTokenizer.tokenize(query.strip())

def load_disease_list(disease_file):
    with open(disease_file, "r", encoding="utf-8") as f:
        return [line.strip().lower() for line in f.readlines() if line.strip()]

def detect_disease_in_query(query: str, disease_list: list):
    query_lower = query.lower()
    sorted_disease_list = sorted(disease_list, key=len, reverse=True)
    for disease in sorted_disease_list:
        if disease in query_lower:
            return disease
    return None

def get_relevant_documents(query: str, vectordb: Chroma, disease_list_file, k: int = 10,detected_disease: str ="None" ):
    if detected_disease:
        print(f"🔍 Phát hiện tên bệnh trong câu hỏi: {detected_disease}")
        retriever = vectordb.as_retriever(
            search_kwargs={"k": k, "filter": {"title": detected_disease}}
        )
    else:
        print("🔎 Không phát hiện tên bệnh, truy vấn toàn bộ.")
        retriever = vectordb.as_retriever(search_kwargs={"k": k})

    return retriever.get_relevant_documents(query)

# Load Chroma đã build sẵn
disease_list_file = Data_health_path +"/disease_list.txt"

vectordb = load_chroma_index(embedding_model, persist_dir)

# Truy vấn
query = "Gần đây tôi hay đi tiểu đêm"
disease_list = load_disease_list(disease_list_file)
detected_disease = detect_disease_in_query(query, disease_list)
k=10
query = preprocess_query(query)
docs = get_relevant_documents(query, vectordb,disease_list_file,k,detected_disease)
for doc in docs:
    print(doc.metadata["title"].capitalize(),"---",doc.metadata["source"],"\n--->",doc.page_content, "\n---------")
