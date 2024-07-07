from fastapi import FastAPI
from demo_gradio import classifier_fn
from pydantic import BaseModel

app = FastAPI()

class PostRequest(BaseModel):
    text:str

@app.get("/")
def root():
    return "OK"

@app.post("/prediction")
def predict(request: PostRequest):
    return {
        "prediction": classifier_fn(request.text)
    }