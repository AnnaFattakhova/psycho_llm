# -*- coding: utf-8 -*-

import os
import json
import re
import numpy as np
from pathlib import Path

import pandas as pd

GITHUB_TOKEN = "..."

!git clone https://{GITHUB_TOKEN}@github.com/...

BASE_DIR = "/content/psycho_llm/results"
OUTPUT_DIR = "/content/psycho_llm"

# Очистка данных

def clean_explanation(explanation_obj):
    if explanation_obj is None:
        return ""

    # 1. извлекаем текст
    if isinstance(explanation_obj, dict):
        text = (
            explanation_obj.get("Объяснение")
            or explanation_obj.get("explanation")
            or ""
        )
    else:
        text = explanation_obj

    if not isinstance(text, str):
        text = str(text)

    text = text.strip()

    # 2. если внутри JSON
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            text = (
                parsed.get("Объяснение")
                or parsed.get("explanation")
                or text
            )
    except:
        pass

    # 3. убираем префикс
    text = re.sub(
        r'^\s*"?\s*(Объяснение|explanation)\s*"?\s*:\s*',
        '',
        text,
        flags=re.IGNORECASE
    )

    # 4. если есть явная склейка ("..." + "...")
    if '+' in text or re.search(r'"\s*"', text):

        parts = re.findall(r'"([^"]+)"', text)

        # берём только длинные куски (не отдельные слова)
        parts = [p.strip() for p in parts if len(p.strip()) > 20]

        if len(parts) >= 2:
            text = " ".join(parts)

    # 6. чистка мусора
    text = text.replace("+", " ")
    text = text.replace("{", " ").replace("}", " ")
    text = re.sub(r'[()\[\]]+', ' ', text)

    # 7. убираем слово explanation / Объяснение
    text = re.sub(r'\b(Объяснение|explanation)\b', '', text, flags=re.IGNORECASE)

    # 8. переносы строк
    text = text.replace("\n", " ")

    # 9. финальная нормализация
    text = text.strip().strip('"').strip()
    text = re.sub(r'\s+', ' ', text)

    return text

# Фильтруем по шкалам, которые нужно игнорировать

def should_skip_scale(test, scale):
    """
    Возвращает True, если шкалу нужно пропустить
    """

    if not isinstance(scale, str):
        return False

    scale_lower = scale.lower()
    test_lower = str(test).lower()

    # --- ASQ / attachment
  #  if test_lower in ["attachment", "asq"]:
 #       if scale_lower in ["доминирующий_тип", "dominant"]:
  #          return True

    # --- EPI / eysenck
    if test_lower in ["eysenck", "epi"]:
        if scale_lower in ["темперамент", "temperament"]:
            return True

    return False

def parse_scores(scores, test):
    rows = []

    if not isinstance(scores, dict) or len(scores) == 0:
        return [(test, np.nan, np.nan)]

    test_lower = str(test).lower()

    # одна шкала
    if "score" in scores and len(scores) <= 3:
        val = scores.get("score")
        return [(test, val, scores.get("interpretation", np.nan))]

    # несколько шкал
    for scale, value in scores.items():

        scale_str = str(scale).strip()
        scale_lower = scale_str.lower()

        # 1. error только по scale
        if scale_lower == "error":
            rows.append((scale_str, np.nan, np.nan))
            continue

        # 2. фильтры

        #if ("attachment" in test_lower or "asq" in test_lower):
        #    if scale_lower in ["доминирующий_тип", "dominant"]:
         #       continue

        if ("eysenck" in test_lower or "epi" in test_lower):
            if scale_lower in ["темперамент", "temperament"]:
                continue

        result = np.nan
        interpretation = np.nan

        # 3. 16PF (надёжно)
        if isinstance(value, dict):

            result = np.nan
            interpretation = np.nan

            for k, v in value.items():
                key = str(k).strip().lower()

                # стен / sten
                if "стен" in key or "sten" in key:
                    result = v

                # уровень / interpretation (на всякий случай)
                if "уровень" in key or "interpretation" in key:
                    interpretation = v

            # fallback для обычных тестов
            if np.isnan(result) and "score" in value:
                result = value.get("score", np.nan)
                interpretation = value.get("interpretation", np.nan)

        # числа
        elif isinstance(value, (int, float)):
            result = value

        # строки
        elif isinstance(value, str):
            interpretation = value

        if not scale_str:
            scale_str = test

        rows.append((scale_str, result, interpretation))

    return rows

all_rows = []

for model_name in os.listdir(BASE_DIR):
    model_path = os.path.join(BASE_DIR, model_name)

    if not os.path.isdir(model_path):
        continue

    files = list(Path(model_path).glob("*.json"))

    print(f"{model_name}: найдено файлов {len(files)}")

    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            model = model_name  # ← ВАЖНО: берём из папки
            temperature = str(data.get("temperature", "default"))
            run = data.get("run", np.nan)
            test = data.get("test", np.nan)

            explanation = clean_explanation(data.get("explanation"))
            if not explanation:
                explanation = np.nan

            scores = data.get("scores", {})
            parsed_scores = parse_scores(scores, test)

            for scale, result, interpretation in parsed_scores:

                if not scale:
                    scale = test

                if pd.isna(result):
                    result_out = np.nan
                else:
                    result_out = str(result)

                all_rows.append({
                    "model": model,
                    "temperature": temperature,
                    "run": run,
                    "test": test,
                    "scale": scale,
                    "results": result_out,
                    "interpretation": interpretation,
                    "explanation": explanation
                })

        except Exception as e:
            print(f"Ошибка в {file_path.name}: {e}")

# Dataframe + сохранение

df = pd.DataFrame(all_rows)

# сортировка для удобства
df["temperature"] = df["temperature"].astype(str)

# естественная сортировка шкал
def natural_key(text):
    return [
        int(part) if part.isdigit() else part
        for part in re.split(r"(\d+)", str(text))
    ]

scale_order = sorted(df["scale"].dropna().unique(), key=natural_key)

df["scale"] = pd.Categorical(
    df["scale"],
    categories=scale_order,
    ordered=True
)

# ГЛАВНАЯ сортировка (как ты хочешь)
df = df.sort_values(
    by=["model", "test", "temperature", "run", "scale"]
).reset_index(drop=True)

# сохранить
OUTPUT_PATH = "/content/psycho_dataset.csv"
df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

print("Готово:", OUTPUT_PATH)
print(df.shape)
df.head()

"""# RENAMING SCALE"""

import pandas as pd

# Укажите имя загруженного файла
#df = pd.read_csv("/content/dataset - all_models.csv")

df = pd.read_csv("/content/psycho_dataset.csv")

# Dataframe + сохранение

df = pd.DataFrame(all_rows)

# сортировка для удобства
df["temperature"] = df["temperature"].astype(str)

# естественная сортировка шкал
def natural_key(text):
    return [
        int(part) if part.isdigit() else part
        for part in re.split(r"(\d+)", str(text))
    ]

scale_order = sorted(df["scale"].dropna().unique(), key=natural_key)

df["scale"] = pd.Categorical(
    df["scale"],
    categories=scale_order,
    ordered=True
)

# ГЛАВНАЯ сортировка (как ты хочешь)
df = df.sort_values(
    by=["model", "test", "temperature", "run", "scale"]
).reset_index(drop=True)

# сохранить
OUTPUT_PATH = "/content/psycho_dataset.csv"
df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

print("Готово:", OUTPUT_PATH)
print(df.shape)
df.head()

df.shape

# Список уникальных значений столбца loc
unique_loc = df["scale"].unique()

print(unique_loc)

len(unique_loc)

# Словарь для переименования шкал
scale_mapping = {

     'Anxiety':'anxiety',
     'Depression': 'depression',
     'Hostility': 'hostility',
     'Interpersonal Sensitivity': 'interpersonal sensitivity',
     'Obsessive-Compulsive': 'obsessive-compulsive',
     'Paranoid Ideation': 'paranoid ideation',
     'Phobic Anxiety': 'phobic anxiety',
     'Psychoticism': 'psychoticism',
     'Somatization': 'somatization',

    'Избегающе-отвергающий': 'dismissive avoidant',
    'Надежный': 'secure attached',
    'Тревожно-избегающий': 'fearful avoidant',
    'Тревожный': 'anxious preoccupied',

    'Вербальная агрессия': 'verbal hostility',
    'Косвенная агрессия': 'indirect hostility',
    'Негативизм': 'negativism',
    'Обида': 'resentment',
    'Подозрительность': 'suspicion',
    'Раздражение': 'irritability',
    'Физическая агрессия': 'assault',
    'Чувство вины': 'guilt',
    'индекс_агрессивности': 'aggression index',
    'индекс_враждебности': 'hostility index',

    'Доброжелательность': 'agreeableness',
    'Добросовестность': 'conscientiousness',
    'Нейротизм': 'neuroticism',
    'нейротизм': 'neuroticism',
    'Открытость_к_опыту': 'openness',
    'Экстраверсия': 'extraversion',
    'экстраверсия': 'extraversion',
    'шкала_лжи': 'lie scale',

    'Враждебность': 'hostility',
    'Депрессия': 'depression',
    'Навязчивости': 'obsessive-compulsive',
    'Паранойяльность': 'paranoid ideation',
    'Психотизм': 'psychoticism',
    'Сенситивность': 'interpersonal sensitivity',
    'Соматизация': 'somatization',
    'Тревожность': 'anxiety',
    'Фобия': 'phobic anxiety',

    'макиавеллизм': 'machiavellianism',
    'нарциссизм': 'narcissism',
    'психопатия': 'psychopathy'
}

# Базовое переименование
df['scale'] = df['scale'].replace(scale_mapping)

# HADS
mask_hads_anxiety = (
    (df['test'].astype(str).str.strip().str.lower() == 'hads') &
    (df['scale'].astype(str).str.strip().str.lower() == 'anxiety')
)

df.loc[mask_hads_anxiety, 'scale'] = 'hads_anxiety'

mask_hads_dep = (
    (df['test'].astype(str).str.strip().str.lower() == 'hads') &
    (df['scale'].astype(str).str.strip().str.lower() == 'depression')
)

df.loc[mask_hads_dep, 'scale'] = 'hads_depression'


# SCL-90
mask_scl_anx = (
    (df['test'].astype(str).str.strip().str.lower() == 'scl-90') &
    (df['scale'].astype(str).str.strip().str.lower() == 'anxiety')
)

df.loc[mask_scl_anx, 'scale'] = 'scl-90_anxiety'

mask_scl_dep = (
    (df['test'].astype(str).str.strip().str.lower() == 'scl-90') &
    (df['scale'].astype(str).str.strip().str.lower() == 'depression')
)

df.loc[mask_scl_dep, 'scale'] = 'scl-90_depression'


# EPI
mask_epi_neuro = (
    (df['test'].astype(str).str.strip().str.lower() == 'epi') &
    (df['scale'].astype(str).str.strip().str.lower() == 'neuroticism')
)

df.loc[mask_epi_neuro, 'scale'] = 'epi_neuroticism'

mask_epi_extra = (
    (df['test'].astype(str).str.strip().str.lower() == 'epi') &
    (df['scale'].astype(str).str.strip().str.lower() == 'extraversion')
)

df.loc[mask_epi_extra, 'scale'] = 'epi_extraversion'


# BFI-2
mask_bfi_neuro = (
    (df['test'].astype(str).str.strip().str.lower() == 'bfi-2') &
    (df['scale'].astype(str).str.strip().str.lower() == 'neuroticism')
)

df.loc[mask_bfi_neuro, 'scale'] = 'bfi-2_neuroticism'

mask_bfi_extra = (
    (df['test'].astype(str).str.strip().str.lower() == 'bfi-2') &
    (df['scale'].astype(str).str.strip().str.lower() == 'extraversion')
)

df.loc[mask_bfi_extra, 'scale'] = 'bfi-2_extraversion'

# Проверка уникальных значений после замены
print(sorted(df['scale'].unique()))

len(df['scale'].unique())

# Для ASQ переносим тип привязанности из scale в interpretation

asq_scales = [
    'dismissive avoidant',
    'secure attached',
    'fearful avoidant',
    'anxious preoccupied'
]

# маска нужных строк
mask = (
    (df['test'].astype(str).str.lower() == 'asq') &
    (df['scale'].astype(str).str.strip().isin(asq_scales))
)

# переносим значения в interpretation
df.loc[mask, 'interpretation'] = df.loc[mask, 'scale']

# заменяем scale на ASQ
df.loc[mask, 'scale'] = 'ASQ'

# Проверка

print(
    df.loc[
        df['test'].astype(str).str.lower() == 'asq',
        ['test', 'scale', 'interpretation']
    ].head(30)
)

"""# RENAMING INTERPRETATION"""

# Список уникальных значений столбца loc
unique_loc = df["interpretation"].unique()

print(unique_loc)

len(unique_loc)

# Переименование interpretation


interp_clean = df['interpretation'].astype(str).str.strip().str.lower()

# low
mask = interp_clean == 'низкий уровень'
df.loc[mask, 'interpretation'] = 'low'

mask = interp_clean == 'низкий'
df.loc[mask, 'interpretation'] = 'low'

mask = interp_clean == 'низкая'
df.loc[mask, 'interpretation'] = 'low'


# moderate
mask = interp_clean == 'средний уровень'
df.loc[mask, 'interpretation'] = 'moderate'

mask = interp_clean == 'умеренная'
df.loc[mask, 'interpretation'] = 'moderate'

mask = interp_clean == 'умеренный'
df.loc[mask, 'interpretation'] = 'moderate'

mask = interp_clean == 'повышенный'
df.loc[mask, 'interpretation'] = 'moderate'


# high
mask = interp_clean == 'высокий уровень'
df.loc[mask, 'interpretation'] = 'high'

mask = interp_clean == 'высокая'
df.loc[mask, 'interpretation'] = 'high'

mask = interp_clean == 'высокий'
df.loc[mask, 'interpretation'] = 'high'


# normal
mask = interp_clean == 'норма'
df.loc[mask, 'interpretation'] = 'normal'

mask = interp_clean == 'нормальное состояние.'
df.loc[mask, 'interpretation'] = 'minimal depression'


# anxiety
mask = interp_clean == 'незначительный уровень тревоги.'
df.loc[mask, 'interpretation'] = 'low anxiety'

mask = interp_clean == 'умеренный уровень тревожности.'
df.loc[mask, 'interpretation'] = 'mild anxiety'

mask = interp_clean == 'средний уровень тревожности.'
df.loc[mask, 'interpretation'] = 'moderate anxiety'


# depression
mask = interp_clean == 'депрессивное расстройство тяжелой степени тяжести.'
df.loc[mask, 'interpretation'] = 'severe depression'


# borderline / abnormal
mask = interp_clean == 'субклинически выраженная тревога / депрессия'
df.loc[mask, 'interpretation'] = 'borderline abnormal'

mask = interp_clean == 'клинически выраженная тревога / депрессия'
df.loc[mask, 'interpretation'] = 'abnormal'


# personality-related
mask = interp_clean == 'нестабильный'
df.loc[mask, 'interpretation'] = 'high neuroticism'

mask = interp_clean == 'искажение'
df.loc[mask, 'interpretation'] = 'invalid result'

mask = interp_clean == 'выраженный экстраверт'
df.loc[mask, 'interpretation'] = 'extraversion'

mask = interp_clean == 'амбиверт'
df.loc[mask, 'interpretation'] = 'average zone'

print(
    df['interpretation']
    .value_counts(dropna=False)
    .head(50)
)

df.to_csv("all_models_translated.csv", index=False)
