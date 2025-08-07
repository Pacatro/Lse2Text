from fastapi import FastAPI

from app.routers import predict, train, eval


app = FastAPI()
app.include_router(train.router)
app.include_router(predict.router)
app.include_router(eval.router)
