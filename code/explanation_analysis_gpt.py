# -*- coding: utf-8 -*-

!pip install openai pandas numpy scikit-learn tqdm sentence-transformers

import torch
torch.cuda.is_available()

import pandas as pd
import numpy as np

from tqdm.auto import tqdm

from openai import OpenAI

from sklearn.cluster import HDBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# API CLIENT
client = OpenAI(api_key="...")

# LOAD DATA

# df должен содержать столбец с текстом
# например: df['text']

df = pd.read_csv("/content/all_models_translated.csv")

TEXT_COLUMN = "explanation"

texts = (
    df[TEXT_COLUMN]
    .astype(str)
    .fillna("")
    .tolist()
)

! pip install tiktoken
import tiktoken
from tqdm.auto import tqdm

# =========================
# EMBEDDINGS
# =========================


# делаем перимерную оценку токенов, чтобы не превышался лимит, иначе модель для эмбеддингов не будет работать

EMBEDDING_MODEL = "text-embedding-3-small"

# tokenizer OpenAI
encoding = tiktoken.encoding_for_model(EMBEDDING_MODEL)

def count_tokens(text):
    return len(encoding.encode(text))


def get_embeddings(
    texts,
    model=EMBEDDING_MODEL,
    max_tokens_per_batch=200000,   # запас относительно лимита 300k
    max_tokens_per_text=8000       # защита от сверхдлинных текстов
):

    embeddings = []

    current_batch = []
    current_tokens = 0

    for text in tqdm(texts):

        text = str(text)

        # считаем реальные токены
        n_tokens = count_tokens(text)

        # обрезаем слишком длинные тексты
        if n_tokens > max_tokens_per_text:

            encoded = encoding.encode(text)

            encoded = encoded[:max_tokens_per_text]

            text = encoding.decode(encoded)

            n_tokens = max_tokens_per_text

        # если batch переполнится -> отправляем
        if current_tokens + n_tokens > max_tokens_per_batch:

            response = client.embeddings.create(
                model=model,
                input=current_batch
            )

            batch_embeddings = [
                item.embedding
                for item in response.data
            ]

            embeddings.extend(batch_embeddings)

            # reset
            current_batch = []
            current_tokens = 0

        current_batch.append(text)
        current_tokens += n_tokens

    # последний batch
    if current_batch:

        response = client.embeddings.create(
            model=model,
            input=current_batch
        )

        batch_embeddings = [
            item.embedding
            for item in response.data
        ]

        embeddings.extend(batch_embeddings)

    return np.array(embeddings)

# убираем пустые explanation

df = df[
    df["explanation"]
    .fillna("")
    .astype(str)
    .str.strip() != ""
].copy()

df = df.reset_index(drop=True)

texts = (
    df["explanation"]
    .astype(str)
    .tolist()
)

print("Texts:", len(texts))

embeddings = get_embeddings(texts)

print(embeddings.shape)

# Без Umap слишком много мелких класетров получается

!pip install umap-learn

import umap

reducer = umap.UMAP(
    n_neighbors=30, # Сколько соседей UMAP использует, чтобы понять локальную структуру пространств; Если кластеров всё ещё слишком много, параметр надо увеличить
    n_components=10, # Сколько измерений останется после reduction; хорошее значение для topic modeling
    min_dist=0.0, # Насколько близко UMAP разрешает точкам сжиматься → плотные компактные кластеры (при увеличении размазываются)
    metric='cosine', # Как UMAP считает similarity между embeddings.
    random_state=42 # Фиксирует randomness.
)

reduced_embeddings = reducer.fit_transform(embeddings)

print(reduced_embeddings.shape)

"""Если хочешь ещё более “жёсткую типологию”:

min_cluster_size=150
min_samples=30

Если хочешь чуть больше структуры:

min_cluster_size=80
min_samples=15
"""

# =========================
# CLUSTERING
# =========================

clusterer = HDBSCAN(
    min_cluster_size=150, #было 100; чтобы кластеры были более крупные, увеличиваем параметр — это задает мнимальный размер готового кластера (если группа меньше 100 объектов, она не считается кластером)
    min_samples=30, #было 20; Насколько “плотной” должна быть точка, чтобы считаться core point (принадлежать к кластеру): чем значение меньше, тем больше шума попадает в кластеры
    metric='euclidean' # euclidean работает стабильнее после умап (+ там уже посчиталась косинусная близость)
)

clusters = clusterer.fit_predict(reduced_embeddings)

df['cluster'] = clusters

print(df['cluster'].value_counts())

texts = df[TEXT_COLUMN].astype(str)

print("docs:", len(texts))
print("avg length:", texts.str.split().apply(len).mean())
print("min length:", texts.str.split().apply(len).min())

from sklearn.feature_extraction.text import CountVectorizer

test_vec = CountVectorizer(ngram_range=(4, 5))
X = test_vec.fit_transform(df[TEXT_COLUMN].astype(str))

print("features:", len(test_vec.get_feature_names_out()))

import re

custom_stop_phrases = [
    "ограничения языковой модели",
    "при формировании ответов",
    "выборе вариантов",
    "мои ответы",
    "основана на",
    "при этом",
    "языковая модель",
    "в соответствии с",
    "моделью",
    "вариантов ответа опиралась",
    "при этом",
    "выборе вариантов",
    "вариантов ответа",
    "опиралась на",
    "ответ на вопрос",
    "ответов основана"
]

def remove_boilerplate(text):
    text = text.lower()
    for phrase in custom_stop_phrases:
        text = text.replace(phrase, " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text

df = df.copy()
df["clean_text"] = df["explanation"].fillna("").apply(remove_boilerplate)

"""## Проверка, насколько хорошо модель разделила на кластеры"""

# Embeddings-based cluster validation

# cosine similarity внутри кластеров

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def cluster_quality_score(embeddings, labels):

    clusters = [c for c in np.unique(labels) if c != -1]

    results = {}

    for c in clusters:
        idx = np.where(labels == c)[0]
        emb = embeddings[idx]

        centroid = emb.mean(axis=0).reshape(1, -1)
        sims = cosine_similarity(emb, centroid).flatten()

        results[c] = {
            "size": len(idx),
            "cohesion": sims.mean()
        }

    return results

scores = cluster_quality_score(embeddings, df["cluster"].values)
scores

import numpy as np
# Взвешенное среднее
sizes = [580, 1626, 422, 538, 159, 1193, 373]

cohesions = [
    0.9219003628591734,
    0.9003655982861545,
    0.9347480324476899,
    0.9132395986039534,
    0.9243064734845552,
    0.9090036192460403,
    0.9410901997376362
]

weighted_mean = np.average(cohesions, weights=sizes)
weighted_mean

"""cohesion = средняя близость к центру кластера
значение vs интерпретация

0.75–1.0	очень хороший кластер
0.6–0.75	нормальный
<0.6	шум / плохой кластер
"""

# Межкластерная separability


def inter_cluster_separation(embeddings, labels):

    clusters = [c for c in np.unique(labels) if c != -1]

    centroids = []

    for c in clusters:
        idx = np.where(labels == c)[0]
        centroids.append(embeddings[idx].mean(axis=0))

    centroids = np.array(centroids)

    sim_matrix = cosine_similarity(centroids)

    return sim_matrix, clusters

"""Хорошо:
тёмные блоки на диагонали
светлые между кластерами
Плохо:
всё светлое → кластеры одинаковые
диагональ не выражена → нет структуры
"""

import matplotlib.pyplot as plt

sim_matrix, clusters = inter_cluster_separation(embeddings, df["cluster"].values)

plt.imshow(sim_matrix)
plt.colorbar()
plt.title("Inter-cluster similarity")
plt.show()

"""→ Модель хорошо разделила на кластеры"""

# =========================
# TF-IDF 3-4 GRAMS
# =========================

vectorizer = TfidfVectorizer(
    ngram_range=(3, 4),
    min_df=3,
    max_df=0.8
)


X_tfidf = vectorizer.fit_transform(df["clean_text"].astype(str))

feature_names = np.array(vectorizer.get_feature_names_out())

"""## Примеры n-gramm"""

# Посмотреть нграммы для кластеров

def show_top_ngrams_per_cluster(
    df,
    tfidf_matrix,
    feature_names,
    cluster_column="cluster",
    top_k=5
):

    clusters = sorted([c for c in df[cluster_column].unique() if c != -1])

    for cluster_id in clusters:

        print(f"CLUSTER {cluster_id}")

        cluster_idx = df[df[cluster_column] == cluster_id].index

        # TF-IDF только для кластера
        cluster_tfidf = tfidf_matrix[cluster_idx]

        # средний TF-IDF по всем документам кластера
        mean_scores = np.asarray(cluster_tfidf.mean(axis=0)).ravel()

        # top n-grams
        top_idx = mean_scores.argsort()[::-1][:top_k]

        for i in top_idx:
            print(f"{feature_names[i]:<60} {mean_scores[i]:.4f}")

show_top_ngrams_per_cluster(
    df=df,
    tfidf_matrix=X_tfidf,
    feature_names=feature_names,
    cluster_column="cluster",
    top_k=10
)

"""## Примеры кластеров"""

def show_and_save_clusters(
    df,
    embeddings,
    filename="cluster_examples.txt",
    cluster_column="cluster",
    text_column="text",
    top_k=15
):

    clusters = sorted([c for c in df[cluster_column].unique() if c != -1])

    lines = []

    for cluster_id in clusters:

        header = "\n" + "=" * 100 + f"\nCLUSTER {cluster_id}\n" + "=" * 100

        print(header)
        lines.append(header)

        mask = df[cluster_column] == cluster_id
        idx = np.where(mask)[0]

        cluster_emb = embeddings[idx]
        cluster_texts = df.iloc[idx][text_column].astype(str).values

        centroid = cluster_emb.mean(axis=0).reshape(1, -1)
        sims = cosine_similarity(cluster_emb, centroid).flatten()

        top_idx = sims.argsort()[::-1][:top_k]

        for rank, i in enumerate(top_idx, start=1):

            line1 = f"\n[{rank}] score={sims[i]:.4f}"
            line2 = cluster_texts[i]
            sep = "-" * 80

            print(line1)
            print(line2)
            print(sep)

            lines.append(line1)
            lines.append(line2)
            lines.append(sep)

    # сохранение
    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nSaved to: {filename}")

show_and_save_clusters(
    df=df,
    embeddings=embeddings,
    filename="clusters.txt",
    cluster_column="cluster",
    text_column=TEXT_COLUMN,
    top_k=15
)

"""## Визуализация"""

# 2D визуализация кластеров

import matplotlib.pyplot as plt

reducer_2d = umap.UMAP(
    n_neighbors=50,
    n_components=2,
    metric="cosine",
    random_state=42
)

emb_2d = reducer_2d.fit_transform(embeddings)

plt.figure(figsize=(10, 7))

plt.scatter(
    emb_2d[:, 0],
    emb_2d[:, 1],
    c=df["cluster"],
    cmap="tab20",
    s=5
)

plt.title("UMAP projection of clusters")
plt.show()

# Semantic heatmap
# плотность кластеров, шум vs структура

plt.hexbin(
    emb_2d[:, 0],
    emb_2d[:, 1],
    gridsize=50,
    cmap="Blues"
)

plt.title("Density map of embeddings")
plt.show()

"""## Суммаризация

Сначала выделяем contrastive keywords
"""

def get_cluster_keywords(df, tfidf_matrix, feature_names, cluster_column="cluster", top_k=10):

    clusters = [c for c in np.unique(df[cluster_column]) if c != -1]

    results = {}

    for c in clusters:

        mask = df[cluster_column].values == c
        idx = np.where(mask)[0]

        cluster_mean = tfidf_matrix[idx].mean(axis=0)
        rest_mean = tfidf_matrix[~mask].mean(axis=0)

        scores = np.asarray(cluster_mean - rest_mean).ravel()

        top_idx = scores.argsort()[::-1][:top_k]

        results[c] = [feature_names[i] for i in top_idx]

    return results

import json

def summarize_cluster_json(df, cluster_id, keywords, text_column="explanation", n_samples=15):

    texts = (
        df[df["cluster"] == cluster_id][text_column]
        .dropna()
        .astype(str)
        .head(n_samples)
        .tolist()
    )

    prompt = f"""
Ты анализируешь кластер текстов.

Тебе даны:
1) примеры текстов
2) ключевые слова (TF-IDF)

Задача:
Верни СТРОГО JSON в следующем формате:

{{
  "cluster_id": {cluster_id},
  "title": "...",
  "description": "...",
  "key_features": ["...", "...", "..."],
  "interpretation_type": "..."
}}

КЛЮЧЕВЫЕ СЛОВА:
{", ".join(keywords)}

ТЕКСТЫ:
{chr(10).join([f"{i+1}. {t}" for i, t in enumerate(texts)])}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Ты аналитик текстовых кластеров. Всегда возвращай только валидный JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    content = response.choices[0].message.content

    return json.loads(content)

def run_hybrid_interpretation(df, tfidf_matrix, feature_names):

    keywords_dict = get_cluster_keywords(df, tfidf_matrix, feature_names)

    cluster_ids = sorted([c for c in df["cluster"].unique() if c != -1])

    results = []

    for c in cluster_ids:

        print(f"Processing cluster {c}...")

        keywords = keywords_dict[c]

        try:
            summary = summarize_cluster_json(
                df=df,
                cluster_id=c,
                keywords=keywords
            )
            results.append(summary)

        except Exception as e:
            print(f"Error in cluster {c}: {e}")

    return results

print(type(X_tfidf))

results = run_hybrid_interpretation(
    df=df,
    tfidf_matrix=X_tfidf,
    feature_names=feature_names
)

results_df = pd.DataFrame(results)
results_df

results_df.to_csv("cluster_interpretations.csv", index=False)
