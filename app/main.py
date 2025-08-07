from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.routers import train, predict, eval

app = FastAPI()

app.include_router(train.router)
app.include_router(predict.router)
app.include_router(eval.router)

BASE_DIR = Path(__file__).resolve().parent

app.mount(
    "/static",
    StaticFiles(directory=BASE_DIR / "static"),
    name="static",
)

templates = Jinja2Templates(directory=BASE_DIR / "templates")


@app.get("/")
async def index(req: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request": req})
