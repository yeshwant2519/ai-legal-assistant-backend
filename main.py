from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ----------------------------
# Load data
# ----------------------------
law_df = pd.read_csv("combined_law_data.csv")

# ----------------------------
# NLP model
# ----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(
    law_df["full_text"].tolist(),
    convert_to_numpy=True
)

# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI(title="AI Legal Assistant API")

# CORS (for browser access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Request schema
# ----------------------------
class QuestionRequest(BaseModel):
    question: str

# ----------------------------
# Semantic search function
# ----------------------------
def semantic_search(query, top_k=3):
    query_embedding = model.encode([query], convert_to_numpy=True)
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    results = law_df.iloc[top_indices].copy()
    results["confidence"] = similarities[top_indices]
    return results

# ----------------------------
# Routes
# ----------------------------
@app.get("/")
def root():
    return {"message": "AI Legal Assistant backend is running"}

@app.post("/ask")
def ask_law(req: QuestionRequest):
    results = semantic_search(req.question)

    return {
        "query": req.question,
        "results": results[
            ["law", "section", "title", "description", "punishment", "confidence"]
        ].to_dict(orient="records")
    }