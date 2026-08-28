import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

BEAUTY_CATEGORIES = [
    "Makeup", "Natural", "Skin", "Personal Care",
    "Hair", "Fragrance", "Men's Store", "Nykaa Luxe", "Brand"
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "Nykaa_Product_Review.csv")


def load_and_clean_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)

    beauty_df = df[
        df["Product Category"].str.split(">").str[0].str.strip().isin(BEAUTY_CATEGORIES)
    ].copy()

    numeric_columns = ["Product Price", "Product Rating", "Product Reviews Count"]
    for col in numeric_columns:
        beauty_df[col] = pd.to_numeric(beauty_df[col], errors="coerce")

    text_columns = ["Product Name", "Product Category", "Product Description",
                     "Product Tags", "Product Contents", "Product Brand"]
    for col in text_columns:
        beauty_df[col] = beauty_df[col].fillna("")

    category_split = beauty_df["Product Category"].str.split(">", expand=True)
    beauty_df["category_level_1"] = category_split[0].str.strip()
    beauty_df["category_level_2"] = category_split[1].str.strip() if category_split.shape[1] > 1 else None
    beauty_df["category_level_3"] = category_split[2].str.strip() if category_split.shape[1] > 2 else None

    beauty_df["product_type"] = beauty_df["category_level_3"].fillna(beauty_df["category_level_2"])
    beauty_df["brand"] = beauty_df["Product Brand"].str.strip().str.lower()

    beauty_df["price_band"] = pd.cut(
        beauty_df["Product Price"],
        bins=[0, 300, 700, 1500, float("inf")],
        labels=["Budget", "Mid-range", "Premium", "Luxury"]
    )

    beauty_df["quality_score"] = (
        beauty_df["Product Rating"] * np.log1p(beauty_df["Product Reviews Count"])
    )
    scaler = MinMaxScaler()
    beauty_df["quality_score_norm"] = scaler.fit_transform(beauty_df[["quality_score"]].fillna(0))

    # dedup: same product appearing under multiple category rows -> merge
    beauty_df = (
        beauty_df.groupby("Product Name", as_index=False)
        .agg({
            "Product Brand": "first", "Product Price": "first", "Product Rating": "first",
            "Product Reviews Count": "first", "Product Category": lambda x: " | ".join(set(x)),
            "Product Description": "first", "Product Tags": "first", "Product Contents": "first",
            "category_level_1": "first", "category_level_2": "first", "product_type": "first",
            "brand": "first", "price_band": "first", "quality_score": "first",
            "quality_score_norm": "first",
        })
    )

    beauty_df["product_type_set"] = beauty_df["product_type"].apply(_type_set)

    beauty_df["recommendation_text"] = (
        beauty_df["Product Name"].fillna("") + " " +
        beauty_df["product_type"].fillna("") + " " +
        beauty_df["brand"].fillna("") + " " +
        beauty_df["Product Description"].fillna("") + " " +
        beauty_df["Product Tags"].fillna("")
    )

    return beauty_df.reset_index(drop=True)


def _type_set(value):
    if not isinstance(value, str) or not value.strip():
        return set()
    return {t.strip().lower() for t in value.split("|") if t.strip()}


def _type_overlap_score(target_set, candidate_set):
    if not target_set or not candidate_set:
        return 0.0
    return len(target_set & candidate_set) / len(target_set | candidate_set)


def _category_match_score(target_row, candidate_row):
    t2 = str(target_row["category_level_2"]).strip().lower()
    c2 = str(candidate_row["category_level_2"]).strip().lower()
    return 1.0 if t2 and c2 and t2 == c2 else 0.0


def build_tfidf(beauty_df: pd.DataFrame):
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    tfidf_matrix = vectorizer.fit_transform(beauty_df["recommendation_text"])
    similarity_matrix = cosine_similarity(tfidf_matrix)
    return vectorizer, tfidf_matrix, similarity_matrix


class Recommender:
    """Loads data + builds the TF-IDF index once; call this once and reuse it."""

    def __init__(self):
        self.df = load_and_clean_data()
        self.vectorizer, self.tfidf_matrix, self.similarity_matrix = build_tfidf(self.df)

    def hybrid_recommend(self, product_name, top_n=10, weights=None, min_text_similarity=0.03):
        weights = weights or {"text": 0.45, "type": 0.20, "category2": 0.20, "price": 0.10, "quality": 0.05}
        matches = self.df[self.df["Product Name"].str.contains(product_name, case=False, na=False)]
        if matches.empty:
            return None

        idx = matches.index[0]
        target = self.df.loc[idx]

        same_l1 = self.df["category_level_1"].astype(str).str.strip().str.lower() == str(target["category_level_1"]).strip().lower()
        pool = self.df[same_l1].copy()
        text_scores = self.similarity_matrix[idx][pool.index.to_numpy()]

        keep = text_scores >= min_text_similarity
        pool = pool[keep]
        if pool.empty:
            return None
        text_scores = text_scores[keep]

        type_scores = pool["product_type_set"].apply(lambda s: _type_overlap_score(target["product_type_set"], s)).to_numpy()
        category2_scores = pool.apply(lambda row: _category_match_score(target, row), axis=1).to_numpy()
        price_scores = (pool["price_band"] == target["price_band"]).astype(float).to_numpy()
        quality_scores = pool["quality_score_norm"].fillna(0).to_numpy()

        final_scores = (
            weights["text"] * text_scores + weights["type"] * type_scores +
            weights["category2"] * category2_scores + weights["price"] * price_scores +
            weights["quality"] * quality_scores
        )

        pool = pool.assign(final_score=final_scores, text_similarity=text_scores)
        pool = pool[pool.index != idx].sort_values("final_score", ascending=False).head(top_n)
        return pool

    def recommend_from_preferences(self, preference_text, top_n=10, category_filter=None, price_band_filter=None):
        query_vector = self.vectorizer.transform([preference_text])
        text_scores = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

        pool = self.df.copy()
        pool_scores = text_scores[pool.index.to_numpy()]

        if category_filter:
            keep = pool["category_level_1"].astype(str).str.lower() == category_filter.lower()
            pool, pool_scores = pool[keep], pool_scores[keep.to_numpy()]
        if price_band_filter:
            keep = pool["price_band"] == price_band_filter
            pool, pool_scores = pool[keep], pool_scores[keep.to_numpy()]

        if pool.empty:
            return None

        quality_scores = pool["quality_score_norm"].fillna(0).to_numpy()
        final_scores = 0.85 * pool_scores + 0.15 * quality_scores
        pool = pool.assign(text_similarity=pool_scores, final_score=final_scores)
        return pool.sort_values("final_score", ascending=False).head(top_n)

from google import genai

def rag_recommend(client, retriever: Recommender, preference_text, top_n=5,
                   category_filter=None, price_band_filter=None, model="gemini-3.6-flash"):
    retrieved = retriever.recommend_from_preferences(
        preference_text, top_n=top_n,
        category_filter=category_filter, price_band_filter=price_band_filter
    )
    if retrieved is None:
        return {"retrieved_products": None, "recommendation_text": "No matches found."}

    context = "\n".join(
        f"- {row['Product Name']} by {row['Product Brand']}, ₹{row['Product Price']}, "
        f"rated {row['Product Rating']}★, type: {row['product_type']}"
        for _, row in retrieved.iterrows()
    )

    prompt = f"""A user is looking for: "{preference_text}"

Top matching products from our catalog:
{context}

Write a short, friendly recommendation (3-5 sentences) explaining which 1-2 products
best fit what they asked for and why. Only reference products listed above."""

    response = client.models.generate_content(model=model, contents=prompt)
    return {"retrieved_products": retrieved, "recommendation_text": response.text}