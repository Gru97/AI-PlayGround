import json
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from langchain_intro.chatbot_agent import hospital_agent_executor
from pydantic import BaseModel


class QuestionQuery(BaseModel):
    query_text: str


app = FastAPI()
@app.post("/question")
async def search(query: QuestionQuery):
   result= hospital_agent_executor.invoke({"input": query.query_text})
   return JSONResponse(content=result['output'])


