import torch
from transformers import BitsAndBytesConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from groq import Groq  
import random
from src import config
from google import genai
from google.genai import types

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Hàm chọn API key ngẫu nhiên
def get_random_api_key(api_keys):
    return random.choice(api_keys)

class Llama3Model:
    def __init__(self, model_name: str, temperature: float, max_completion_tokens: int):
        self.model_name = model_name
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens

    def __call__(self, prompt: str, **kwargs):
        api_key = get_random_api_key(config.api_keys_llama)
        client = Groq(api_key=api_key)

        completion = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": "Bạn là một trợ lý AI thông minh tên là Lisa được phát triển bởi Trần Tấn Thịnh, bạn có nhiệm vụ hỗ trợ người dùng trong việc hỏi đáp các vấn đề liên quan đến bệnh tật dựa vào các thông tin tôi cung cấp, hãy trả lời câu hỏi một cách có chọn lọc, đầy đủ thông tin và chính xác với câu hỏi mà tôi đưa ra!(Câu trả dạng văn bản thuần tiếng việt để dễ dàng chuyển thành giọng nói, loại bỏ các ký tự đặc biệt, các ký tự khác chữ và số)"
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

class GeminiModel:
    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.api_key = api_key

    def __call__(self, prompt: str, **kwargs):
        if not isinstance(prompt, str):
            prompt = str(prompt)
        client = genai.Client(api_key=self.api_key)
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=prompt),
                ],
            ),
        ]
        generate_content_config = types.GenerateContentConfig(
            response_mime_type="text/plain",
        )

        response = ""
        for chunk in client.models.generate_content_stream(
            model=self.model_name,
            contents=contents,
            config=generate_content_config,
        ):
            response += chunk.text

        return response

def get_gemini_model(model_name: str = "gemini-2.0-flash", api_key: str = None):
    if api_key is None:
        api_key = get_random_api_key(config.api_keys_gemini)
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set in environment variables.")
    return GeminiModel(model_name=model_name, api_key=api_key)
