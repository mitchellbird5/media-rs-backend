# app.py
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.views import router

from media_rs.utils.data_cache import DataCache

app = FastAPI(title="Media Recommender API")

@app.get("/")
def root():
    return {"status": "ok", "service": "MediaRS"}

if os.getenv("ENVIRONMENT") == "production":
    origins = os.getenv("CORS_ALLOWED_ORIGINS", "").split(",")
else:
    origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

# Startup event
@app.on_event("startup")
async def startup_event():
    global cache
    cache = DataCache(repo_id=os.getenv("HF_REPO_ID"))
    cache.warmup()  # if warmup is async; otherwise just cache.warmup()
    print("DataCache warmup finished")