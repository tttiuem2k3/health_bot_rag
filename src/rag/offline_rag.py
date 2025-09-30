import re
from langchain import hub
from langchain_core.runnables import RunnablePassthrough,RunnableLambda
from langchain_core.output_parsers import StrOutputParser


class Str_OutputParser(StrOutputParser):
    def __init__(self) -> None:
        super().__init__()
    
    def parse(self, text: str) -> str:
        return self.extract_answer(text)
    
    
    def extract_answer(self,
                       text_response: str,
                       pattern: str = r"Answer:\s*(.*)"
                       ) -> str:
        
        match = re.search(pattern, text_response, re.DOTALL)
        if match:
            answer_text = match.group(1).strip()
            return answer_text
        else:
            return text_response


class Offline_RAG:
    def __init__(self, llm) -> None:
        self.llm = llm
        self.prompt = hub.pull("rlm/rag-prompt")
        self.str_parser = Str_OutputParser()

    def debug_retriever_output(self, docs):
        print("\n📘🎯 Documents retrieved:\n")
        for i, doc in enumerate(docs):
            print(f"🚀 Document {i+1} --- {doc.metadata['title'].title()} --- {doc.metadata['source']}")
            print(f"🧾 Nội dung: {doc.page_content[:100]} .....(còn tiếp)\n")
            print()
        return docs
    
    def get_chain(self, retriever):
        input_data = {
            "context": retriever
                | RunnableLambda(self.debug_retriever_output)
                | RunnableLambda(lambda docs: self.format_docs(docs)),  
            "question": RunnablePassthrough()
        }
        rag_chain = (
            input_data
            | self.prompt
            | self.llm
            | self.str_parser
        )
        return rag_chain

    def format_docs(self, docs):
        # init_promt= f"Dựa vào dữ liệu lịch sử chat bot: {history}\n"
        formatted_docs = "\n\n".join(
            f"Dữ liệu y tế của {doc.metadata['title'].title()} tham khảo thêm ở đường link {doc.metadata['source']}: {doc.page_content}" for doc in docs
        )
        # formatted_docs = f"{init_promt}\n{formatted_docs}"
        # print(f"🧾 Nội dung promt: {formatted_docs}\n")
        return formatted_docs