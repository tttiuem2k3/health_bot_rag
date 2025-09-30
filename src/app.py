from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes
from langchain.schema import HumanMessage
from src import config
from src.iot.iot_routes import router as iot_router 
from src.rag.rag_chain import health_chain, chat_chain
from src.rag.main import InputQA, OutputQA

# --------- App - FastAPI ----------------

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple api server using Langchain's Runnable interfaces",
)
app.include_router(iot_router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# --------- Routes - FastAPI ----------------
# Chuyển đổi HumanMessage thành dictionary
def human_message_to_dict(message: HumanMessage):
    return {
        "content": message.content
    }

@app.get("/check")
async def check():
    return {"status": "ok"}

import logging
logging.basicConfig(level=logging.INFO)

@app.post("/health_bot", response_model=OutputQA)
async def generative_ai(inputs: InputQA):
    logging.info(f"Payload received: {inputs}")
    question = inputs.question
    logging.info(f"    ❓  Question: {question}")
    answer = health_chain.invoke(question)
    logging.info(f"    ✅ Answer: {answer}")
    return {"answer": answer}


# @app.post("/machine_learning", response_model=OutputQA)
# async def machine_learning(inputs: InputQA):
#     answer = orther_chain.invoke(inputs.question)
#     return {"answer": answer}



# --------- Langserve Routes - Playground ----------------
add_routes(app, 
           health_chain, 
           playground_type="default",
           path="/health_bot")

# add_routes(app, 
#            orther_chain, 
#            playground_type="default",
#            path="/machine_learning")


add_routes(app,
           chat_chain,
           enable_feedback_endpoint=False,
           enable_public_trace_link_endpoint=False,
           playground_type="default",
           path="/chat")

"""
Run app - local: uvicorn src.app:app --host 127.0.0.1 --port 8000 --reload 
"""