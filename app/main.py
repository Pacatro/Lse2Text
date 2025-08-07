from fastapi import FastAPI

from app.core.config import settings
from app.routers import predict, train, eval


app = FastAPI()
app.include_router(train.router)
app.include_router(predict.router)
app.include_router(eval.router)


@app.get("/")
def main():
    return {"img width": settings.img_width, "img height": settings.img_height}
