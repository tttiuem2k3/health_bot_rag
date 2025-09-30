from typing import Union, List, Literal
import glob
from tqdm import tqdm
import multiprocessing
from langchain_community.document_loaders import PyPDFLoader, BSHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd
from langchain.schema import Document
from pyvi import ViTokenizer

# Hàm tiền xử lý văn bản tiếng Việt
def tokenize_text(text: str):
    return ViTokenizer.tokenize(text.strip())

def remove_non_utf8_characters(text):
    return ''.join(char for char in text if ord(char) < 128)

def load_pdf(pdf_file):
    docs = PyPDFLoader(pdf_file, extract_images=True).load()
    for doc in docs:
        doc.page_content = remove_non_utf8_characters(doc.page_content)
    return docs

def load_html(html_file):
    docs = BSHTMLLoader(html_file).load()
    for doc in docs:
        doc.page_content = remove_non_utf8_characters(doc.page_content)
    return docs


def get_num_cpu():
    return multiprocessing.cpu_count()


class BaseLoader:
    def __init__(self) -> None:
        self.num_processes = get_num_cpu()

    def __call__(self, files: List[str], **kwargs):
        pass


class PDFLoader(BaseLoader):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, pdf_files: List[str], **kwargs):
        num_processes = min(self.num_processes, kwargs["workers"])
        with multiprocessing.Pool(processes=num_processes) as pool:
            doc_loaded = []
            total_files = len(pdf_files)
            with tqdm(total=total_files, desc="Loading PDFs", unit="file") as pbar:
                for result in pool.imap_unordered(load_pdf, pdf_files):
                    doc_loaded.extend(result)
                    pbar.update(1)
        print(f"✅ Đã tải {len(doc_loaded)} documents từ PDF files!")
        return doc_loaded


class HTMLLoader(BaseLoader):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, html_files: List[str], **kwargs):
        num_processes = min(self.num_processes, kwargs["workers"])
        with multiprocessing.Pool(processes=num_processes) as pool:
            doc_loaded = []
            total_files = len(html_files)
            with tqdm(total=total_files, desc="Loading HTMLs", unit="file") as pbar:
                for result in pool.imap_unordered(load_html, html_files):
                    doc_loaded.extend(result)
                    pbar.update(1)
        print(f"✅ Đã tải {len(doc_loaded)} documents từ WEB!")
        return doc_loaded

class CSVLoader(BaseLoader):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, csv_files: List[str], **kwargs):
        """
        Load dữ liệu từ file CSV và chuyển đổi thành danh sách các đối tượng Document.
        :param csv_files: Danh sách đường dẫn tới các file CSV.
        :return: Danh sách các đối tượng Document.
        """
        documents = []
        for csv_file in csv_files:
            data = pd.read_csv(csv_file)
            data = data.fillna("")  # Xử lý các giá trị null
            for _, row in data.iterrows():
                # Tạo đối tượng Document từ mỗi hàng
                processed_content = tokenize_text(row["Document"].strip())

                document = Document(
                    page_content=processed_content,
                    metadata={
                        "title": row.get("Title", "").strip().lower(),
                        "source": row.get("Source", "").strip(),
                    }
                )
                documents.append(document)
        print(f"✅ Đã tải {len(documents)} documents từ {len(csv_files)} CSV files!")
        return documents
    
class TextSplitter:
    def __init__(self, 
                 separators: List[str] = ['\n\n', '\n', ' ', ''],
                 chunk_size: int = 1024,
                 chunk_overlap: int = 128
                 ) -> None:
        
        self.splitter = RecursiveCharacterTextSplitter(
            separators=separators,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    def __call__(self, documents):
        return self.splitter.split_documents(documents)



class Loader:
    def __init__(self, 
                 file_type: str = Literal["pdf", "html", "csv"],
                 split_kwargs: dict = {
                     "chunk_size": 1024,
                     "chunk_overlap": 128}
                 ) -> None:
        assert file_type in ["pdf", "html", "csv"], "file_type must be either pdf, html, or csv"
        self.file_type = file_type
        if file_type == "pdf":
            self.doc_loader = PDFLoader()
        elif file_type == "html":
            self.doc_loader = HTMLLoader()
        elif file_type == "csv":
            self.doc_loader = CSVLoader()
        else:
            raise ValueError("file_type must be either pdf, html, or csv")

        self.doc_spltter = TextSplitter(**split_kwargs)

    def load(self, files: Union[str, List[str]], workers: int = 1):
        if isinstance(files, str):
            files = [files]
        doc_loaded = self.doc_loader(files, workers=workers)
        doc_split = self.doc_spltter(doc_loaded)
        print(f"Đã chia thành {len(doc_split)} chunk!")
        return doc_split

    def load_dir(self, dir_path: str, workers: int = 1):
        if self.file_type == "pdf":
            files = glob.glob(f"{dir_path}/*.pdf")
            assert len(files) > 0, f"No {self.file_type} files found in {dir_path}"
        elif self.file_type == "html":
            files = glob.glob(f"{dir_path}/*.html")
            assert len(files) > 0, f"No {self.file_type} files found in {dir_path}"
        elif self.file_type == "csv":
            files = glob.glob(f"{dir_path}/*.csv")
            assert len(files) > 0, f"No {self.file_type} files found in {dir_path}"
        else:
            raise ValueError("Unsupported file type for directory loading.")
        return self.load(files, workers=workers)