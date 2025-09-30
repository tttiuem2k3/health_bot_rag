from groq import Groq  
import random
from src import config
api_keys_llama = config.api_keys_llama
# Hàm chọn API key ngẫu nhiên
def get_random_api_key(api_keys):
    return random.choice(api_keys)


class Llama3Model:
    def __init__(self, model_name: str, temperature: float, max_completion_tokens: int):
        self.model_name = model_name
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens

    def __call__(self, prompt: str, **kwargs):
        api_key = get_random_api_key(api_keys_llama)
        client = Groq(api_key=api_key)

        completion = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": "Bạn là một trợ lý AI thông minh tên là David được phát triển bởi Trần Tấn Thịnh. Bạn trả lời câu hỏi ngắn nhưng đảm bảo độ chính xác.(Câu trả dạng văn bản thuần tiếng việt để dễ dàng chuyển thành giọng nói, loại bỏ các ký tự đặc biệt, các ký tự khác chữ và số)"
                },
                {
                    "role": "user",
                    "content": str(prompt)
                }
            ],
            temperature=self.temperature,
            max_completion_tokens=self.max_completion_tokens,
            top_p=1,
            stream=True,
            stop=None,
        )
        response = ""
        for chunk in completion:
            response += chunk.choices[0].delta.content or ""
        return response

def get_llama3_model(model_name: str = "meta-llama/llama-4-scout-17b-16e-instruct", 
                     max_completion_tokens: int = 8192, temperature: float = 2, **kwargs):
    return Llama3Model(model_name=model_name, 
                       temperature=temperature, 
                       max_completion_tokens=max_completion_tokens)