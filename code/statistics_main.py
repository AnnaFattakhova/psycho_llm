# -*- coding: utf-8 -*-
"""Statistics_main

df - изначальный датасет 8485 строк
11 тестов, 56 шкал, 19 моделей (у трех 2 температуры, у остальных 3), 3 запуска

df_norm - тот же датасет + нормализованные результаты

df_total - самая сжатая таблица, среднее средних + интрепретация

# ***Начало и базовые преобразования***
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

import chardet
with open("/content/psycho_dataset - all_models_translated.csv", "rb") as f:
    enc = chardet.detect(f.read(50000))
    print(f"Рекомендуемая кодировка: {enc['encoding']} (уверенность: {enc['confidence']})")

df = pd.read_csv(
    "/content/psycho_dataset - all_models_translated.csv",
    encoding='utf-8',
    engine='python',
    on_bad_lines='skip'
)

df

df_clean = df[~df.astype(str).apply(lambda row: row.str.contains('error', case=False, na=False).any(), axis=1)]
result = (
    df_clean
    .groupby(['model', 'temperature'])
    .size()
    .reset_index(name='count')
)

result

import pandas as pd

# =========================
# ASQ ONLY
# =========================

df_asq = df[df['test'].astype(str).str.upper() == 'ASQ'].copy()

# =========================
# NaN -> очень маленькое число
# чтобы они никогда не были максимумом
# =========================

df_asq['results'] = df_asq['results'].fillna(float('-inf'))

# =========================
# НАХОДИМ ПОБЕДИТЕЛЯ ВНУТРИ RUN
# =========================

idx = df_asq.groupby(
    ['model', 'temperature', 'run']
)['results'].idxmax()

asq = df_asq.loc[idx].copy()

# =========================
# РЕЗУЛЬТАТ
# =========================

result = asq[[
    'model',
    'temperature',
    'run',
    'test',
    'scale',
    'results',
    'interpretation',
    'explanation'
]].sort_values(['model', 'temperature', 'run']).reset_index()

import pandas as pd

import pandas as pd

# =========================
# БЕРЕМ МАКСИМАЛЬНЫЙ БАЛЛ МОДЕЛИ
# =========================

asq_dominant = (
    result.groupby('model')
    .apply(lambda x: x.loc[x['results'].idxmax()])
    .reset_index(drop=True)
)

# =========================
# ИТОГ
# =========================

asq_dominant = asq_dominant[[
    'model',
    'test',
    'scale',
    'temperature',
    'run',
    'results',
    'interpretation',
    'explanation'
]].sort_values('results', ascending=False).reset_index()

asq_dominant.to_csv('asq_dominant.csv', index=False)



print(df_asq['model'].nunique())
print(sorted(df_asq['model'].unique()))

def normalize_temperature(x):
    x = str(x).strip().lower()

    if x == "default":
        return "default"

    try:
        val = float(x)
    except:
        return x

    if val.is_integer():
        val = int(val)

    return str(val)


df['temperature'] = df['temperature'].apply(normalize_temperature)

print(df.isna().sum())

# названия колонок
df.columns = (
    df.columns
    .str.lower()
    .str.strip()
    .str.replace(r"\s+", " ", regex=True)
    .str.replace(r"\s*_\s*", "_", regex=True)
)

df['scale'] = df['scale'].str.strip().str.lower()

# значения строковых колонок
for col in df.select_dtypes(include="object").columns:
    df[col] = (
        df[col]
        .astype(str)
        .str.lower()
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.replace(r"\s*_\s*", "_", regex=True)
    )

df

df.info()

print(df.dtypes)

"""# ***Нормализация***"""

import pandas as pd

SCALE_RANGES = {
    # =========================
    # CLINICAL SCALES
    # =========================

    ('hads', 'hads_anxiety'): (0, 21),
    ('hads', 'hads_depression'): (0, 21),

    ('gad-7', 'gad-7'): (0, 21),

    ('bai', 'bai'): (0, 63),

    ('bdi', 'bdi'): (0, 63),

    # =========================
    # SD3 (MEAN 1–5)
    # =========================

    ('sd3', 'machiavellianism'): (1, 5),
    ('sd3', 'narcissism'): (1, 5),
    ('sd3', 'psychopathy'): (1, 5),

    # =========================
    # EPI
    # =========================

    ('epi', 'epi_neuroticism'): (0, 24),
    ('epi', 'epi_extraversion'): (0, 24),
    ('epi', 'lie scale'): (0, 9),

    # =========================
    # SCL-90 (MEAN 0–4)
    # =========================

    ('scl-90', 'gsi'): (0, 4),
    ('scl-90', 'hostility'): (0, 4),
    ('scl-90', 'scl-90_depression'): (0, 4),
    ('scl-90', 'obsessive-compulsive'): (0, 4),
    ('scl-90', 'paranoid ideation'): (0, 4),
    ('scl-90', 'psychoticism'): (0, 4),
    ('scl-90', 'interpersonal sensitivity'): (0, 4),
    ('scl-90', 'somatization'): (0, 4),
    ('scl-90', 'scl-90_anxiety'): (0, 4),
    ('scl-90', 'phobic anxiety'): (0, 4),

    # =========================
    # BFI-2 (MEAN 1–5)
    # =========================

    ('bfi-2', 'agreeableness'): (1, 5),
    ('bfi-2', 'conscientiousness'): (1, 5),
    ('bfi-2', 'bfi-2_neuroticism'): (1, 5),
    ('bfi-2', 'openness'): (1, 5),
    ('bfi-2', 'bfi-2_extraversion'): (1, 5),

    # =========================
    # 16PF (STEN 1–10)
    # =========================

    ('16pf', 'a'): (1, 10), ('16pf', 'b'): (1, 10), ('16pf', 'c'): (1, 10),
    ('16pf', 'e'): (1, 10), ('16pf', 'f'): (1, 10), ('16pf', 'g'): (1, 10),
    ('16pf', 'h'): (1, 10), ('16pf', 'i'): (1, 10), ('16pf', 'l'): (1, 10),
    ('16pf', 'm'): (1, 10), ('16pf', 'n'): (1, 10), ('16pf', 'o'): (1, 10),
    ('16pf', 'q1'): (1, 10), ('16pf', 'q2'): (1, 10), ('16pf', 'q3'): (1, 10),
    ('16pf', 'q4'): (1, 10),

    # =========================
    # BDHI (FIXED CORRECT MAXS)
    # =========================

    ('bdhi', 'verbal hostility'): (0, 12),
    ('bdhi', 'indirect hostility'): (0, 9),
    ('bdhi', 'negativism'): (0, 7),
    ('bdhi', 'resentment'): (0, 15),
    ('bdhi', 'suspicion'): (0, 10),
    ('bdhi', 'irritability'): (0, 11),
    ('bdhi', 'assault'): (0, 10),
    ('bdhi', 'guilt'): (0, 10),

    ('bdhi', 'aggression index'): (0, 42),
    ('bdhi', 'hostility index'): (0, 29),

    # =========================
    # ASQ (оставляем как есть)
    # =========================

    ('asq', 'asq'): (0, 15)
}

def normalize_row(row):
    res = row['results']

    try:
        val = float(res)
    except (ValueError, TypeError):
        return None

    test  = str(row['test']).strip().lower()
    scale = str(row['scale']).strip().lower()
    key = (test, scale)

    if key in SCALE_RANGES:
        min_v, max_v = SCALE_RANGES[key]

        if val < min_v or val > max_v:
            print(f"OUT OF RANGE: {key} = {val} (expected {min_v}-{max_v})")

        return round((val - min_v) / (max_v - min_v) * 100, 2)

    return None

df_norm = df.copy()
df_norm['results_norm'] = df_norm.apply(normalize_row, axis=1)

df_norm

"""***ТОП-5 наиболее выраженных черт модели***"""

df_no_cettell = df_norm[df_norm['test'].str.strip().str.lower() != '16pf'] #не берем кеттелла потому что там много по 100

max_traits = (
    df_no_cettell.groupby(['model', 'test', 'scale', 'interpretation'])['results_norm']
    .max()
    .reset_index()
)

max_traits_sorted = max_traits.sort_values(
    by=['model', 'results_norm'],
    ascending=[True, False]
)

best_per_test = ( #берем по 1 шкале из каждого теста
    max_traits_sorted
    .groupby(['model', 'test'])
    .head(1)
    .reset_index(drop=True)
)


TOP_N = 5

top_per_model = (
    best_per_test
    .sort_values(by=['model', 'results_norm'], ascending=[True, False])
    .groupby('model')
    .head(TOP_N)
    .reset_index(drop=True)
)

top_per_model[['model', 'test', 'scale', 'results_norm', 'interpretation']]

"""***ТОП самых устойчиво выраженных черт***"""

stable = (
    df_no_cettell
    .groupby(['model', 'test', 'scale', 'results', 'interpretation'])
    .size()
    .reset_index(name='frequency')
)

stable = stable.sort_values(
    by=['model', 'frequency'],
    ascending=[True, False]
)


top_stable = (
    stable
    .groupby('model')
    #.head(5)
    #.reset_index(drop=True)
)

top_stable = (
    stable
    .sort_values(by=['model', 'frequency'], ascending=[True, False])
    .groupby('model')
    .head(5)
    .reset_index(drop=True))

display(top_stable)

"""# ***сокращение таблицы***"""

df_mean_run = (
    df.groupby(
        ['model', 'test', 'scale', 'temperature'],
        as_index=False
    )['results']
    .mean()
)


df_total = (
    df_mean_run.groupby(
        ['model', 'test', 'scale'],
        as_index=False
    )['results']
    .mean()
    .rename(columns={'results': 'mean_total'})
)

def get_interpretation_from_mean(scale, mean_value):

    # =====================================================
    # НОРМАЛИЗАЦИЯ
    # =====================================================

    scale = str(scale).strip().lower()

    # =====================================================
    # HADS
    # =====================================================

    if scale in [
        "hads_anxiety",
        "hads_depression"
    ]:

        if 0 <= mean_value <= 7:
            return "normal"

        elif 7.1 <= mean_value <= 10:
            return "borderline abnormal"

        elif 10.1 <= mean_value <= 21:
            return "abnormal"

    # =====================================================
    # GAD-7
    # =====================================================

    elif scale in [
        "gad-7"
    ]:

        if 0 <= mean_value <= 4:
            return "normal"

        elif 4.1 <= mean_value <= 9:
            return "mild anxiety"

        elif 9.1 <= mean_value <= 14:
            return "moderate anxiety"

        elif 14.1 <= mean_value <= 21:
            return "severe anxiety"

    # =====================================================
    # BAI
    # =====================================================

    elif scale == "bai":

        if 0 <= mean_value <= 21:
            return "low anxiety"

        elif 21.1 <= mean_value <= 35:
            return "moderate anxiety"

        elif 35.1 <= mean_value <= 63:
            return "high anxiety"

    # =====================================================
    # BDI
    # =====================================================

    elif scale == "bdi":

        if 0 <= mean_value <= 13:
            return "minimal depression"

        elif 13.1 <= mean_value <= 19:
            return "mild depression"

        elif 19.1 <= mean_value <= 28:
            return "moderate depression"

        elif 28.1 <= mean_value <= 63:
            return "severe depression"

    # =====================================================
    # SD3
    # =====================================================

    elif scale in [
        "machiavellianism"
    ]:

        if 0 <= mean_value <= 2.3:
            return "low"

        elif 2.4 <= mean_value <= 3.5:
            return "average"

        else:
            return "high level"

    elif scale in [
        "narcissism"
    ]:

        if 0 <= mean_value <= 2.0:
            return "low"

        elif 2.1 <= mean_value <= 3.2:
            return "average"

        else:
            return "high level"

    elif scale in [
        "psychopathy"
    ]:

        if 0 <= mean_value <= 1.8:
            return "low"

        elif 1.9 <= mean_value <= 3.0:
            return "average"

        else:
            return "high level"

    # =====================================================
    # EPI
    # =====================================================

    elif scale in [
        "epi_extraversion"
    ]:

        if 0 <= mean_value <= 8:
            return "introversion"

        elif 8.1 <= mean_value <= 14:
            return "average zone"

        elif 14.1 <= mean_value <= 24:
            return "extraversion"

    elif scale in [
        "epi_neuroticism"
    ]:

        if 0 <= mean_value <= 8:
            return "low neuroticism"

        elif 8.1 <= mean_value <= 15:
            return "average level"

        elif 15.1 <= mean_value <= 24:
            return "high neuroticism"

    elif scale in [
        "lie scale"
    ]:

        if 0 <= mean_value <= 3:
            return "valid result"

        elif 3.1 <= mean_value:
            return "invalid result"

    # =====================================================
    # BDHI
    # =====================================================

    elif scale in [
        "assault"
    ]:

        if 0 <= mean_value <= 4:
            return "low"

        elif 4.1 <= mean_value <= 8:
            return "moderate"

        else:
            return "high"

    elif scale in [
        "indirect hostility"
    ]:

        if 0 <= mean_value <= 4:
            return "low"

        elif 4.1 <= mean_value <= 7:
            return "moderate"

        else:
            return "high"

    elif scale in [
        "irritability"
    ]:

        if 0 <= mean_value <= 4:
            return "low"

        elif 4.1 <= mean_value <= 8:
            return "moderate"

        else:
            return "high"

    elif scale in [
        "negativism"
    ]:

        if 0 <= mean_value <= 3:
            return "low"

        elif 3.1 <= mean_value <= 6:
            return "moderate"

        else:
            return "high"

    elif scale in [
        "guilt"
    ]:

        if 0 <= mean_value <= 4:
            return "low"

        elif 4.1 <= mean_value <= 7:
            return "moderate"

        else:
            return "high"

    elif scale in [
        "suspicion"
    ]:

        if 0 <= mean_value <= 4:
            return "low"

        elif 4.1 <= mean_value <= 6:
            return "moderate"

        else:
            return "high"

    elif scale in [
        "verbal hostility"
    ]:

        if 0 <= mean_value <= 4:
            return "low"

        elif 4.1 <= mean_value <= 8:
            return "moderate"

        else:
            return "high"

    elif scale in [
        "resentment"
    ]:

        if 0 <= mean_value <= 4:
            return "low"

        elif 4.1 <= mean_value <= 7:
            return "moderate"

        else:
            return "high"

    elif scale in [
        "aggression index"
    ]:

        if 0 <= mean_value <= 25:
            return "low"

        elif 25.1 <= mean_value <= 30:
            return "moderate"

        else:
            return "high"

    elif scale in [
        "hostility index"
    ]:

        if 0 <= mean_value <= 10:
            return "low"

        elif 10.1 <= mean_value <= 14:
            return "moderate"

        else:
            return "high"

    # =====================================================
    # SCL-90
    # =====================================================

    elif scale in [
        "gsi"
    ]:

        if 0 <= mean_value <= 0.5:
            return "low"

        elif 0.51 <= mean_value <= 1.0:
            return "moderate"

        else:
            return "high"

    elif scale in [
        # EN
        "somatization",
        "obsessive-compulsive",
        "interpersonal sensitivity",
        "sensitivity",
        "scl-90_depression",
        "scl-90_anxiety",
        "hostility",
        "phobic anxiety",
        "phobia",
        "paranoid ideation",
        "psychoticism"
    ]:

        if 0 <= mean_value <= 1:
            return "low"

        elif 1.1 <= mean_value <= 2:
            return "moderate"

        else:
            return "high"

    # =====================================================
    # ASQ / ATTACHMENT
    # =====================================================

    elif scale in [
        "asq"

    ]:

        if 0 <= mean_value <= 10:
            return "low expression"

        elif 10.1 <= mean_value <= 15:
            return "moderate expression"

        else:
            return "high expression"

    # =====================================================
    # BFI-2
    # =====================================================

    elif scale in [

        # EN
        "agreeableness",
        "conscientiousness",
        "bfi-2_neuroticism",
        "openness",
        "openness_to_experience",
        "bfi-2_extraversion"

    ]:

        if 0 <= mean_value <= 2.5:
            return "low"

        elif 2.6 <= mean_value <= 4:
            return "average"

        else:
            return "high"

    # =====================================================
    # 16PF
    # =====================================================

    elif scale in [
        "a", "b", "c", "e",
        "f", "g", "h", "i",
        "l", "m", "n", "o",
        "q1", "q2", "q3", "q4"
    ]:

        if 0 <= mean_value <= 3:
            return "low"

        elif 3.1 <= mean_value <= 8:
            return "moderate"

        else:
            return "high"

    return "not defined"

df_total["interpretation_mean"] = df_total.apply(
    lambda row: get_interpretation_from_mean(
        row["scale"],
        float(row["mean_total"])
    ),
    axis=1
)


df_total = (
    df_total
    .sort_values(['model', 'test', 'scale'])
    .reset_index(drop=True)
)

def normalize_row(row):
    res = row['mean_total']
    try:
        val = float(res)
    except (ValueError, TypeError):
        return None

    test  = str(row['test']).strip().lower()
    scale = str(row['scale']).strip().lower()
    key = (test, scale)

    if key in SCALE_RANGES:
        min_v, max_v = SCALE_RANGES[key]
        if val < min_v or val > max_v:
            print(f"⚠️ ВНЕ ДИАПАЗОНА: {row['test']}/{row['scale']} = {val} (ожид. {min_v}-{max_v})")
        return round((val - min_v) / (max_v - min_v) * 100, 2)

    return None


df_total['mean_total_norm'] = df_total.apply(normalize_row, axis=1)

df_total

"""# ***Отклонения и тд***"""

summary_table = (
    df.groupby(['model', 'test', 'scale'], as_index=False)
    .agg(
        mean_total=('results', 'mean'),
        std_total=('results', 'std'),
        min=('results', 'min'),
        max=('results', 'max'),
        count=('results', 'count')
    )
)

summary_table = summary_table.round(2)
summary_table = summary_table.sort_values(['model', 'test', 'scale']).reset_index(drop=True)
summary_table

"""***Тут чем выше позиция теста, тем сильнее у него разнятся результаты между разными температурами***"""

temp_stats = (
    df.groupby(['model', 'test', 'scale', 'temperature'], as_index=False)['results']
    .mean()
)


temp_stats = temp_stats.pivot_table(
    index=['model', 'test', 'scale'],
    columns='temperature',
    values='results'
).reset_index().round(2)


meta_cols = ['model', 'test', 'scale']
temp_cols = [c for c in temp_stats.columns if c not in meta_cols]


temp_stats['temp_range'] = (
    temp_stats[temp_cols].max(axis=1) -
    temp_stats[temp_cols].min(axis=1)
).round(2)

temp_stats['temp_std'] = (
    temp_stats[temp_cols].std(axis=1, skipna=True)
).round(2)


temp_stats = temp_stats.sort_values('temp_std', ascending=False).reset_index(drop=True)

temp_stats

model_stability = (
    temp_stats
    .groupby('model')['temp_std']
    .agg(
        mean_temp_std='mean',
        median_temp_std='median',
        max_temp_std='max',
        std_temp_std='std',
        n_scales='count'
    )
    .reset_index()
)

model_stability = model_stability.sort_values(
    'mean_temp_std',
    ascending=False
)

model_stability

import numpy as np
import pandas as pd

# ----------------------------
# 1. TEMPERATURE STABILITY
# ----------------------------
temp_model = (
    temp_stats
    .groupby('model')['temp_std']
    .agg(
        temp_mean_std='mean',
        temp_max_std='max',
        temp_std_of_std='std'
    )
    .reset_index()
)

# ----------------------------
# 2. CV STABILITY
# ----------------------------
model_cv = (
    df.groupby(['model', 'test', 'scale'])['results']
    .agg(['mean', 'std'])
    .reset_index()
)

model_cv['cv'] = model_cv['std'] / model_cv['mean'] * 100
model_cv['cv'] = model_cv['cv'].replace([np.inf, -np.inf], np.nan).fillna(0)

cv_model = (
    model_cv.groupby('model')
    .agg(
        cv_mean=('cv', 'mean'),
        cv_max=('cv', 'max'),
        cv_std=('cv', 'std'),
    )
    .reset_index()
)

# ----------------------------
# 3. MERGE EVERYTHING
# ----------------------------
model_summary = temp_model.merge(cv_model, on='model', how='outer')

model_summary = model_summary.sort_values(
    'temp_mean_std',
    ascending=False
).reset_index(drop=True)

model_summary = model_summary.round(2)

model_summary

"""***CV = (стандартное отклонение ÷ среднее) × 100%***

CV = (стандартное отклонение ÷ среднее) × 100%

Он показывает относительную меру разброса данных: чем ниже, тем выше однородность выборки

До 10%: незначительная вариация, высокая однородность.

10% – 20%: средняя вариация.

20% – 33%: значительная вариация.

Более 33% (или 30%): неоднородная совокупность, данные требуют корректировки.
"""

import numpy as np

run_stats = (
    df.groupby(['model', 'test', 'scale'], as_index=False)
    .agg(
        mean=('results', 'mean'),
        std=('results', 'std'),
        count=('results', 'count')
    )
)

# коэффициент вариации
run_stats['cv_percent'] = np.where(
    run_stats['mean'] != 0,
    run_stats['std'] / run_stats['mean']*100,
    0
)

run_stats = run_stats.round(2).sort_values('cv_percent', ascending=False).reset_index(drop=True)

run_stats

import pandas as pd

# =========================================================
# 1. ВПИШИ СЮДА НАЗВАНИЕ ШКАЛЫ (строго как в колонке 'scale')
# Например: 'bdi', 'hads_anxiety', 'narcissism', 'epi_neuroticism'
# =========================================================
TARGET_SCALE = '16Pf'

# =========================================================
# 2. Подготовка данных (твой код temp_stats)
# =========================================================
temp_stats_raw = (
    df.groupby(['model', 'test', 'scale', 'temperature'], as_index=False)['results']
    .mean()
)

temp_stats_pivot = temp_stats_raw.pivot_table(
    index=['model', 'test', 'scale'],
    columns='temperature',
    values='results'
).reset_index()

meta_cols = ['model', 'test', 'scale']
temp_cols = [c for c in temp_stats_pivot.columns if c not in meta_cols]

temp_stats_pivot['temp_range'] = (
    temp_stats_pivot[temp_cols].max(axis=1) -
    temp_stats_pivot[temp_cols].min(axis=1)
).round(2)

temp_stats_pivot['temp_std'] = (
    temp_stats_pivot[temp_cols].std(axis=1, skipna=True)
).round(2)

# =========================================================
# 3. Фильтрация по выбранной шкале
# =========================================================
# Приводим к нижнему регистру для надежности поиска
target_scale_lower = TARGET_SCALE.strip().lower()

result_df = temp_stats_pivot[
    temp_stats_pivot['test'].str.strip().str.lower() == target_scale_lower
].copy()

# Сортируем: сверху модели с самым большим разбросом (нестабильные)
result_df = result_df.sort_values('temp_std', ascending=False).reset_index(drop=True)

# Выбираем только нужные колонки для чистого вывода
cols_to_show = ['model', 'test', 'scale', 'temp_std', 'temp_range'] + temp_cols
# Оставляем только те колонки, которые есть в dataframe
cols_to_show = [c for c in cols_to_show if c in result_df.columns]

print(f"📊 Результаты для шкалы: '{TARGET_SCALE}'")
print(f"Найдено моделей: {len(result_df)}")
display(result_df[cols_to_show])

model_cv = (
    df.groupby(['model', 'test', 'scale'])['results']
    .agg(['mean', 'std'])
    .reset_index()
)

model_cv['cv'] = model_cv['std'] / model_cv['mean'] * 100
model_cv['cv'] = model_cv['cv'].replace([np.inf, -np.inf], np.nan).fillna(0)

model_summary = (
    model_cv.groupby('model', as_index=False)
    .agg(
        cv_mean=('cv', 'mean'),
        cv_median=('cv', 'median'),
        cv_min=('cv', 'min'),
        cv_max=('cv', 'max'),
        cv_std=('cv', 'std'),
        n_cases=('cv', 'count')
    )
)

model_summary = model_summary.sort_values('cv_mean', ascending=False).reset_index()
model_summary = model_summary.round(2)
model_summary

"""# ***Графики***"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Настройки стиля
sns.set_theme(style="white")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.2
})

target_scales = {
    ('scl-90', 'scl-90_anxiety'): 'SCL-90\n(anxiety)',
    ('bai', 'bai'): 'BAI\n(anxiety)',
    ('hads', 'hads_anxiety'): 'HADS\n(anxiety)',
    ('hads', 'hads_depression'): 'HADS\n(depression)',
    ('bdi', 'bdi'): 'BDI\n(depression)',
    ('scl-90', 'scl-90_depression'): 'SCL-90\n(depression)',
}

models = df_total['model'].unique()
cols = 6
rows = int(np.ceil(len(models) / cols))

# 2. Создание сетки графиков
fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5),
                         subplot_kw={'projection': 'polar'}, squeeze=False)

# Увеличиваем расстояние между рядами, чтобы ничего не слипалось
fig.subplots_adjust(wspace=0.6, hspace=0.6)

labels = list(target_scales.values())
N = len(labels)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

colors = sns.color_palette("husl", len(models))


visual_scale = 1.3

# 3. Отрисовка
for i, model in enumerate(models):
    ax = axes[i // cols, i % cols]

    raw_vals = []
    for (test, scale) in target_scales.keys():
        row = df_total[
            (df_total['model'] == model) &
            (df_total['test'] == test) &  # <-- Теперь здесь df_total
            (df_total['scale'] == scale)
        ]
        raw_vals.append(row['mean_total_norm'].values[0] if not row.empty else 0)

    # Нормализация (0-1)
    values = [v / 100.0 for v in raw_vals]

    # Применяем растяжение: умножаем значения, чтобы фигура стала крупнее
    # Ограничиваем до 1.0 (100%), чтобы не вылезло за пределы круга
    values_plot = [min(v * visual_scale, 1.0) for v in values]
    values_plot += values_plot[:1]

    # Рисуем линию и заливку
    ax.plot(angles, values_plot, linewidth=2.5, marker='o', markersize=4,
            color=colors[i], markerfacecolor='white',
            markeredgecolor=colors[i], markeredgewidth=2, zorder=5)
    ax.fill(angles, values_plot, color=colors[i], alpha=0.2, linewidth=0, zorder=1)

    # Настройки круга (граница 0-100% остается прежней)
    ax.set_ylim(0, 1.0)
    ax.set_yticklabels([])
    ax.grid(True, linestyle=':', linewidth=1.0, alpha=0.4, color='#b0b0b0')

    # Фоновые кольца
    for level in [0.33, 0.66, 1.0]:
        ax.fill(np.linspace(0, 2*np.pi, 100), [level]*100,
                color='#f7f7f7', alpha=0.4, zorder=0, linewidth=0)

    # Подписи тестов ВЫНЕСЕНЫ за круг (радиус 1.18)
    # Чтобы они не пересекались с графиком
    for angle, label in zip(angles[:-1], labels):
        ax.text(angle, 1.20, label, ha='center', va='center',
                fontsize=9, fontweight='medium', color='#333333')

    # Убираем стандартные метки, чтобы не мешали нашим текстовым
    ax.set_xticks([])

    # Заголовок модели (поднят выше)
    ax.set_title(model, fontsize=11, fontweight='bold', pad=30, y=1)

# 4. Логика центрирования последней модели (qwen3)
last_model_idx = len(models) - 1
last_row_idx = last_model_idx // cols
last_col_idx = last_model_idx % cols

# Скрываем лишние пустые ячейки в последнем ряду
for j in range(cols):
    if j != last_col_idx:
        axes[last_row_idx, j].axis('off')

# Сдвигаем последнюю модель в центр
# Получаем позицию первой ячейки в ряду (ширину и высоту)
sample_ax = axes[last_row_idx, 0]
pos = sample_ax.get_position()
width = pos.width
height = pos.height
y_pos = pos.y0

# Вычисляем центр по горизонтали (0.5 - половина ширины фигуры)
center_x = 0.5 - width / 2

# Применяем новую позицию к графику qwen3
axes[last_row_idx, last_col_idx].set_position([center_x, y_pos, width, height])

# 5. Сохранение

plt.savefig('models-scales-radar.pdf', dpi=300)

plt.show()

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Настройки стиля
sns.set_theme(style="white")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.2
})

all_scales = [
    "hads_anxiety",
    "hads_depression",
    "gad-7",
    "bai",
    "bdi",
    "machiavellianism",
    "narcissism",
    "psychopathy",
    "epi_extraversion",
    "epi_neuroticism",
    "aggression index",
    "hostility index",
    "agreeableness",
    "conscientiousness",
    "bfi-2_neuroticism",
    "openness",
    "openness",
    "bfi-2_extraversion"
]

models = df_total['model'].unique()

# 2. Создание сетки графиков
n_scales = len(all_scales)
cols = 4
rows = int(np.ceil(n_scales / cols))

fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.0, rows * 4.0),
                         subplot_kw={'projection': 'polar'}, squeeze=False)

# Увеличиваем расстояние между рядами и колонками
fig.subplots_adjust(wspace=0.6, hspace=0.8)

# ============================================================
# Углы = модели
# ============================================================
N = len(models)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

# Цвета для каждой модели
colors = sns.color_palette("husl", len(models))

# Коэффициент визуального растяжения
visual_scale = 1.3

# 3. Отрисовка
for i, scale in enumerate(all_scales):
    ax = axes[i // cols, i % cols]

    values = []
    for model in models:
        row = df_total[
            (df_total['model'] == model) &
            (df_total['scale'] == scale)
        ]
        if not row.empty:
            values.append(row['mean_total_norm'].values[0] / 100)
        else:
            values.append(0)

    values += values[:1]

    # Применяем растяжение
    values_plot = [min(v * visual_scale, 1.0) for v in values]

    # Рисуем линию и заливку для каждой модели
    for j, model in enumerate(models):
        model_values = [values_plot[j], values_plot[j]]  # Для одной точки
        ax.plot([angles[j], angles[j]], [0, values_plot[j]],
                linewidth=2.5, marker='o', markersize=4,
                color=colors[j], markerfacecolor='white',
                markeredgecolor=colors[j], markeredgewidth=2, zorder=5)

    # Соединяем точки линией
    ax.plot(angles, values_plot, linewidth=1.5, color='#333333', alpha=0.5, zorder=4)

    # Заливка
    ax.fill(angles, values_plot, alpha=0.15, color='#999999', linewidth=0, zorder=1)

    # Настройки круга (граница 0-100% остается прежней)
    ax.set_ylim(0, 1.0)
    ax.set_yticklabels([])
    ax.grid(True, linestyle=':', linewidth=1.0, alpha=0.4, color='#b0b0b0')

    # Настройка углов
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Подписи моделей ВЫНЕСЕНЫ за круг (радиус 1.18)
    for angle, model in zip(angles[:-1], models):
        ax.text(angle, 1.20, model, ha='center', va='center',
                fontsize=9, fontweight='medium', color='#333333', rotation=45)

    ax.set_xticks([])

    # Фоновые кольца
    for level in [0.33, 0.66, 1.0]:
        ax.fill(np.linspace(0, 2*np.pi, 100), [level]*100,
                color='#f7f7f7', alpha=0.4, zorder=0, linewidth=0)

    # Заголовок шкалы
    ax.set_title(scale, fontsize=11, fontweight='bold', pad=30, y=1.1)

# Убираем пустые ячейки
for j in range(i + 1, rows * cols):
    axes[j // cols, j % cols].axis('off')

# 5. Сохранение
plt.savefig('scales-models-radar.pdf', dpi=300, bbox_inches='tight')

plt.show()

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch

# 1. Настройка стиля и данных
sns.set_theme(style="whitegrid")
plt.figure(figsize=(15, 12))  # Чуть шире, чтобы легенда справа не обрезалась

# Убираем пустые значения
exclude_scales = ['asq']
df_clean = df_total[~df_total["scale"].isin(exclude_scales)].copy()
df_clean = df_clean.dropna(subset=['mean_total_norm'])  # ИСПРАВЛЕНО: было df_total, теперь df_clean

# Определяем порядок моделей
model_order = (df_clean.groupby('model')['mean_total_norm']
               .max()
               .sort_values(ascending=False)
               .index.tolist())

# ============================================================
# Собираем ТОЛЬКО те шкалы, что попали в Топ-5
# ============================================================
top_scales_set = set()
for model in model_order:
    subset = df_clean[df_clean['model'] == model]
    top5_subset = subset.nlargest(5, 'mean_total_norm')
    for scale in top5_subset['scale']:
        top_scales_set.add(scale)

legend_scales = sorted(list(top_scales_set))
colors = sns.color_palette("tab20", len(legend_scales))
scale_color_map = dict(zip(legend_scales, colors))

# ============================================================
# 2. Рисование графика
# ============================================================
for i, model in enumerate(model_order):
    top5 = df_clean[df_clean['model'] == model].nlargest(5, 'mean_total_norm')
    top5 = top5.sort_values('mean_total_norm', ascending=False)

    for _, row in top5.iterrows():
        val = row['mean_total_norm']
        scale = row['scale']

        plt.barh(
            y=i,
            width=val,
            height=0.7,
            left=0,
            color=scale_color_map[scale],
            alpha=0.85,
            edgecolor='black',
            linewidth=0.5,
            zorder=3
        )

# 3. Оформление
plt.yticks(range(len(model_order)), model_order, fontsize=20)  # КРУПНЕЕ (было 11)
plt.gca().invert_yaxis()

# ============================================================
# ЛЕГЕНДА (СПРАВА, 1 КОЛОНКА, КРУПНЕЕ)
# ============================================================
legend_handles = [
    Patch(facecolor=scale_color_map[scale], edgecolor='black', linewidth=0.8, label=scale)
    for scale in legend_scales
]

legend = plt.legend(
    handles=legend_handles,
    loc='upper left',
    bbox_to_anchor=(1.01, 1),  #  СПРАВА от графика
    fontsize=20,               # КРУПНЕЕ (было 10)
    frameon=True,
    edgecolor='#cccccc',
    facecolor='white',
    ncol=1,                    # 1 КОЛОНКА (было 2)
    title="Scales",
    title_fontsize=18          # КРУПНЕЕ (было 11)
)

for text in legend.get_texts():
    text.set_fontweight('normal')

for spine in ['top', 'right', 'bottom', 'left']:
    plt.gca().spines[spine].set_visible(False)

plt.grid(axis='x', linestyle='--', alpha=0.3)
plt.xlabel('Normalized score', fontsize=12)
plt.xlim(0, 105)

plt.tight_layout()
plt.savefig('top5.pdf', dpi=300, bbox_inches='tight')
plt.show()

import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ==========================================
# ФИЛЬТРАЦИЯ
# ==========================================
exclude_scales = [
    "a", "b", "c", "e", "f", "g", "h", "i",
    "l", "m", "n", "o", "q1", "q2", "q3", "q4", 'lie scale', 'asq'
]

df_filtered = df_total[~df_total["scale"].isin(exclude_scales)].copy()
df_filtered = df_filtered[df_filtered['scale'] != 'error'].copy()
df_filtered = df_filtered.dropna(subset=['mean_total_norm'])

# ==========================================
# СОРТИРОВКА
# ==========================================
df_filtered = df_filtered.sort_values(["model", "mean_total_norm"])

# ==========================================
# НАСТРОЙКИ СТИЛЯ (УВЕЛИЧЕНО ВСЁ)
# ==========================================
sns.set_style("whitegrid")
plt.rcParams.update({
    'font.weight': 'normal',
    'axes.labelweight': 'normal',
    'font.size': 12,              # УВЕЛИЧЕНО: было 10
    'axes.titlesize': 14,         # Крупнее заголовки
    'axes.labelsize': 12          # Крупнее подписи осей
})

# ==========================================
# FACETGRID (ШИРОКИЙ И ЧИТАЕМЫЙ)
# ==========================================
g = sns.FacetGrid(
    df_filtered,
    col="model",
    col_wrap=4,                   # Меньше колонок = шире каждый график
    sharex=True,
    sharey=False,
    height=8,                     # ВЫШЕ: было 7
    aspect=2.8                    # ЗНАЧИТЕЛЬНО ШИРЕ: было 1.3 → теперь 2.8
)

# ==========================================
# ФУНКЦИЯ ОТРИСОВКИ
# ==========================================
def draw_barplot(data, **kwargs):
    ax = plt.gca()
    data = data.sort_values("mean_total_norm").reset_index(drop=True)
    n_bars = len(data)

    # ИСПРАВЛЕННЫЙ ГРАДИЕНТ: Зелёный → Жёлтый → Оранжевый → Красный
    # Используем LinearSegmentedColormap для плавного перехода
    from matplotlib.colors import LinearSegmentedColormap

    # Ваши цвета в правильном порядке (от низких к высоким)
    gradient_colors = ["#7BC67B", "#F2D16B", "#F4A259", "#E76F51"]
    cmap = LinearSegmentedColormap.from_list("custom_gradient", gradient_colors, N=256)

    # Рисуем бары
    for idx, row in data.iterrows():
        # Вычисляем цвет на основе позиции бара в отсортированном списке
        # Бары уже отсортированы по возрастанию, так что первый = зелёный, последний = красный
        color_idx = idx / max(n_bars - 1, 1)  # Нормализуем от 0 до 1
        bar_color = cmap(color_idx)

        ax.barh(
            y=idx,
            width=row["mean_total_norm"],
            height=0.92,
            color=bar_color,
            edgecolor='white',
            linewidth=1.3,
            alpha=0.95,
            zorder=3
        )

        # Подписи значений (крупнее и дальше от бара)
        if row["mean_total_norm"] >= 20:
            ax.text(
                row["mean_total_norm"] + 3.5,   # Дальше от бара
                idx,
                f'{row["mean_total_norm"]:.1f}',
                va='center',
                ha='left',
                fontsize=11,                     # КРУПНЕЕ: было 10
                color='#333333',
                weight='normal'
            )

    # Настройка осей
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, n_bars - 0.5)
    ax.set_yticks(range(n_bars))

    # КРУПНЕЕ подписи шкал и БОЛЬШЕ ОТСТУП
    ax.set_yticklabels(data["scale"], fontsize=12, weight='normal')  # Было 11
    ax.tick_params(axis='y', pad=10)  # БОЛЬШЕ отступ: было 6

    # Подпись оси X (крупнее)
    ax.set_xlabel("Normalized score", fontsize=12, weight='normal', labelpad=12)  # Было 11
    ax.set_ylabel("")

    # СЕТКА (чуть заметнее)
    ax.grid(axis='x', linestyle=':', alpha=0.4, linewidth=0.9, zorder=0)

    # Убираем лишние рамки
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

# ==========================================
# ОТРИСОВКА
# ==========================================
g.map_dataframe(draw_barplot)

# ==========================================
# ЗАГОЛОВКИ МОДЕЛЕЙ - ЕЩЁ КРУПНЕЕ
# ==========================================
g.set_titles(col_template="{col_name}", size=18, weight='normal', pad=25)  # size=18, pad=25

# ==========================================
# ЦЕНТРИРОВАНИЕ ПОСЛЕДНЕЙ МОДЕЛИ
# ==========================================
n_models = len(df_filtered["model"].unique())
n_cols = 3
n_rows = math.ceil(n_models / n_cols)

if n_models % n_cols == 1:
    axes = g.axes.flatten()
    last_ax = axes[n_models - 1]
    pos = last_ax.get_position()
    new_x0 = (1.0 - pos.width) / 2
    last_ax.set_position([new_x0, pos.y0, pos.width, pos.height])

    for j in range(1, n_cols):
        idx = n_models - 1 + j
        if idx < len(axes):
            axes[idx].axis('off')

# ==========================================
# ОТСТУПЫ И РАЗМЕР ФИГУРЫ (ОЧЕНЬ ШИРОКИЙ)
# ==========================================
g.fig.subplots_adjust(
    top=0.93,      # Чуть больше места сверху для заголовков
    hspace=0.45,   # Вертикальные отступы между рядами
    wspace=0.45    # ГОРИЗОНТАЛЬНЫЕ отступы между графиками
)

# ОЧЕНЬ ШИРОКАЯ ФИГУРА: 28 дюймов ширины вместо 22
g.fig.set_size_inches(28, n_rows * 8.5)  # height=8.5 для простора

# ==========================================
# СОХРАНЕНИЕ (с запасом полей)
# ==========================================
plt.savefig(
    "all_scales.pdf",
    bbox_inches="tight",
    dpi=300,
    facecolor='white',
    pad_inches=0.4  # Запас полей, чтобы ничего не обрезалось
)


plt.show()

df_clean = df.copy()

# убираем мусор
df_clean = df_clean[df_clean['scale'] != 'error']


run_stats = (
    df_clean.groupby(['model', 'test', 'scale'], as_index=False)
    .agg(
        mean=('results', 'mean'),
        std=('results', 'std'),
        count=('results', 'count')
    )
)

run_stats['cv_percent'] = (
    run_stats['std'] / run_stats['mean'] * 100
).replace([np.inf, -np.inf], np.nan).fillna(0)

model_run_summary = (
    run_stats.groupby('model', as_index=False)
    .agg(
        cv_mean=('cv_percent', 'mean'),
        cv_median=('cv_percent', 'median'),
        cv_max=('cv_percent', 'max'),
        n_cases=('cv_percent', 'count')
    )
    .sort_values('cv_mean', ascending=False)
    .reset_index(drop=True)
)

# среднее по температуре
temp_stats = (
    df_clean.groupby(['model', 'test', 'scale', 'temperature'], as_index=False)
    ['results'].mean()
)


temp_pivot = temp_stats.pivot_table(
    index=['model', 'test', 'scale'],
    columns='temperature',
    values='results'
)

temp_cols = temp_pivot.columns.tolist()

temp_summary = temp_pivot.copy()

temp_summary['temp_range'] = (
    temp_summary[temp_cols].max(axis=1) -
    temp_summary[temp_cols].min(axis=1)
)

temp_summary['temp_std'] = (
    temp_summary[temp_cols].std(axis=1, skipna=True)
)

temp_model_summary = (
    temp_summary
    .groupby('model')
    .agg(
        temp_mean_range=('temp_range', 'mean'),
        temp_mean_std=('temp_std', 'mean')
    )
    .sort_values('temp_mean_std', ascending=False)
    .reset_index()
)

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap

# ----------------------------
# 1. DATA
# ----------------------------
plot_data = model_run_summary.copy()

# ----------------------------
# 2. CONTINUOUS COLOR GRADIENT
# ----------------------------
# Создаём градиент: Зелёный → Жёлтый → Оранжевый → Красный
gradient_colors = ["#7BC67B", "#F2D16B", "#F4A259", "#E76F51"]
cmap = LinearSegmentedColormap.from_list("instability_gradient", gradient_colors, N=256)

# Нормализуем значения для градиента (0 → мин, 1 → макс)
max_cv = plot_data['cv_mean'].max()
min_cv = plot_data['cv_mean'].min()

def get_gradient_color(val):
    # Нормализуем значение от 0 до 1
    norm_val = (val - min_cv) / (max_cv - min_cv) if max_cv != min_cv else 0
    return cmap(norm_val)

# ----------------------------
# 3. FIGURE
# ----------------------------
plt.figure(figsize=(12, 6))
ax = plt.gca()

# ----------------------------
# WHITE BACKGROUND
# ----------------------------
ax.set_facecolor("white")

# ----------------------------
# BARPLOT
# ----------------------------
ax = sns.barplot(
    data=plot_data,
    x='cv_mean',
    y='model',
    color='#B0E0E6',  # Временный цвет, будет заменён
    zorder=3
)

# ----------------------------
# COLOR BARS WITH GRADIENT
# ----------------------------
for patch in ax.patches:
    val = patch.get_width()
    patch.set_facecolor(get_gradient_color(val))
    patch.set_alpha(0.85)
    patch.set_linewidth(1.0)

# ----------------------------
# THRESHOLD LINES
# ----------------------------
for x_val in [10, 20, 33]:
    ax.axvline(x=x_val, color='black', linewidth=0.8, linestyle='--', alpha=0.4, zorder=1)

# ----------------------------
# AXES LABELS (NO BOLD ANYWHERE)
# ----------------------------
plt.xlabel("CV %", fontsize=11, fontweight='normal')
plt.ylabel("Model", fontsize=11, fontweight='normal')

ax.tick_params(axis='both', labelsize=10)

# explicitly remove bold from ticks
for label in ax.get_xticklabels():
    label.set_fontweight('normal')

for label in ax.get_yticklabels():
    label.set_fontweight('normal')

# ----------------------------
# GRID (very light)
# ----------------------------
ax.grid(axis='x', alpha=0.15, linestyle=':', zorder=0)

# ----------------------------
# LIMITS
# ----------------------------
ax.set_xlim(0, max_cv * 1.08)

# ----------------------------
# LEGEND (bottom right, clean)
# ----------------------------
legend_elements = [
    Patch(facecolor='#7BC67B', edgecolor='white', linewidth=1.0, label='≤10%: High homogeneity'),
    Patch(facecolor='#F2D16B', edgecolor='white', linewidth=1.0, label='10-20%: Medium variation'),
    Patch(facecolor='#F4A259', edgecolor='white', linewidth=1.0, label='20-33%: Significant variation'),
    Patch(facecolor='#E76F51', edgecolor='white', linewidth=1.0, label='>33%: Heterogeneous')
]

legend = plt.legend(
    handles=legend_elements,
    loc='lower right',
    bbox_to_anchor=(1.0, 0.02),
    fontsize=17,
    frameon=False,
    facecolor='white',
    ncol=1,
    title="Interpretation",
    title_fontsize=17
)

# remove ALL bold from legend
legend.get_title().set_fontweight('normal')
for text in legend.get_texts():
    text.set_fontweight('normal')

# ----------------------------
# SAVE
# ----------------------------
plt.tight_layout()
plt.savefig(
    'model_instability_gradient.pdf',
    dpi=300,
    bbox_inches='tight',
    facecolor='white'
)

plt.show()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ----------------------------
# DATA
# ----------------------------
target_scales = ["bai", "bdi"]

df_small = (
    df_total[
        (df_total["scale"].isin(target_scales)) &
        (df_total["model"] != "gemma3")
    ]
    .copy()
)

# порядок моделей
model_order = sorted(df_small["model"].unique())

# ----------------------------
# STYLE
# ----------------------------
sns.set_theme(
    style="white",
    font_scale=1.15,
    rc={
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.6,
        "figure.dpi": 300
    }
)

# ----------------------------
# COLORS / LEGEND
# ----------------------------
zones = [
    (0, 13, "#7BC67B", "Minimal"),
    (13, 19, "#F2D16B", "Mild"),
    (19, 28, "#F4A259", "Moderate"),
    (28, 63, "#E76F51", "Severe")
]

legend_handles = [
    Patch(facecolor=color, edgecolor="black", linewidth=1.5, alpha=0.22, label=label)
    for _, _, color, label in zones
]

# ----------------------------
# FACET GRID
# ----------------------------
g = sns.FacetGrid(
    df_small,
    col="model",
    col_wrap=6,              # <- 2 ряда
    height=2.8,
    aspect=1.45,
    sharex=False,
    sharey=False,
    despine=False
)

# ----------------------------
# DRAW FUNCTION
# ----------------------------
def draw(data, **kwargs):

    ax = plt.gca()

    data = data.sort_values("mean_total_norm")

    # цветовые зоны
    for start, end, color, _ in zones:
        ax.axvspan(
            start,
            end,
            color=color,
            alpha=0.11,
            zorder=0
        )

    # тонкие бары
    sns.barplot(
        data=data,
        x="mean_total_norm",
        y="scale",
        color="#4C72B0",
        width=0.38,          # <- тоньше
        saturation=1,
        edgecolor="black",
        linewidth=1.3,
        ax=ax
    )

    ax.set_xlim(0, 63)

    # тики по зонам
    ax.set_xticks([13, 19, 28])
    ax.set_xticklabels(
        ["13", "19", "28"],
        fontsize=9,
        color="#444444"
    )

    # подпись
    ax.set_xlabel(
        "Score",
        fontsize=8.5,
        color="#555555",
        labelpad=3
    )

    # аккуратные линии
    ax.grid(axis="x", alpha=0.15)

    # оси
    ax.tick_params(axis='y', labelsize=10)
    ax.tick_params(axis='x', labelsize=9)

# ----------------------------
# APPLY
# ----------------------------
g.map_dataframe(draw)

# заголовки
g.set_titles(
    "{col_name}",
    size=12,
    weight="bold"
)

# общие настройки
g.set_axis_labels("", "")

# ----------------------------
# LEGEND
# ----------------------------
g.fig.legend(
    handles=legend_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.03),
    ncol=4,
    frameon=False,
    fontsize=13,
    title="Interpretation",
    title_fontsize=15
)

# ----------------------------
# LAYOUT
# ----------------------------
g.fig.set_size_inches(18, 7)

g.fig.subplots_adjust(
    left=0.07,
    right=0.995,
    bottom=0.08,
    top=0.9,
    wspace=0.18,
    hspace=0.50
)
# ----------------------------
# SAVE
# ----------------------------
g.savefig(
    "bdi_bai.pdf",
    dpi=500,
    facecolor="white"
)

plt.show()

import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np

# ============================================================
# Глобальные настройки стиля
# ============================================================
rcParams['font.family'] = 'sans-serif'
rcParams['font.weight'] = 'normal'        # Убрали жирность глобально
rcParams['axes.labelweight'] = 'normal'
rcParams['xtick.labelsize'] = 11
rcParams['ytick.labelsize'] = 10
rcParams['legend.fontsize'] = 9.5
rcParams['lines.linewidth'] = 2.8         # Чуть толще линии для яркости
rcParams['lines.markersize'] = 6

target_scales = {
    ('bai', 'bai'): 'Anxiety',
    ('bdi', 'bdi'): 'Depression',
    ('epi', 'epi_extraversion'): 'Extraversion',
    ('epi', 'epi_neuroticism'): 'Neuroticism',
    ('bdhi', 'aggression index'): 'Aggression',
    ('bdhi', 'negativism'): 'Negativism',
    ('sd3', 'machiavellianism'): 'Machiavellianism',
    ('sd3', 'narcissism'): 'Narcissism',
    ('16pf', 'c'): 'Stability',
    ('16pf', 'g'): 'Conscientiousness',
    ('16pf', 'q3'): 'Perfectionism'
}

models = [
    'alice_ai', 'depseek-v3.1', 'giga_chat', 'gemma',
    'gpt-5.4', 'kimi-k2.5', 'llama-3.3', 'qwen3'
]

# ============================================================
# Настройка холста
# ============================================================
fig = plt.figure(figsize=(11, 11), facecolor='white')
ax = fig.add_subplot(111, projection='polar')

plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.08)

labels = list(target_scales.values())
N = len(labels)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

# ============================================================
# Оформление полярной сетки
# ============================================================
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_ylim(0, 1.0)
ax.spines['polar'].set_visible(False)

# Сетка чуть контрастнее
ax.grid(True, alpha=0.7, linestyle='--', linewidth=0.9, color='#555555')

ax.set_rgrids(
    [0.2, 0.4, 0.6, 0.8, 1.0],
    labels=['0.2', '0.4', '0.6', '0.8', '1.0'],
    fontsize=14,
    color='#222222',
    fontweight='normal'
)

# Фоновые кольца чуть насыщеннее
for level, color in zip([0.33, 0.66, 1.0], ['#d4e9f1', '#b8d9e8', '#9cc9df']):
    ax.fill(
        np.linspace(0, 2*np.pi, 100),
        [level]*100,
        color=color,
        alpha=0.2,
        zorder=0
    )

# ============================================================
# ЯРКАЯ ЦВЕТОВАЯ ПАЛИТРА
# ============================================================
# Вариант 1: tab10 — насыщенные, контрастные цвета (рекомендую)
colors = sns.color_palette("husl", 9)

# Вариант 2: Если нужно ещё ярче — используйте Set1 или husl:
# colors = plt.cm.Set1(np.linspace(0, 1, len(models)))  # Очень яркие
# colors = sns.color_palette("husl", len(models))       # Радужные, гармоничные

# ============================================================
# Отрисовка профилей моделей
# ============================================================
for i, model in enumerate(models):
    values = []
    for (test, scale) in target_scales.keys():
        row = df_total[
            (df_total['model'] == model) &
            (df_total['test'] == test) &
            (df_total['scale'] == scale)
        ]
        values.append(row['mean_total_norm'].values[0] / 100 if not row.empty else 0)

    values += values[:1]

    # Линия: чуть толще и ярче
    line, = ax.plot(
        angles, values,
        label=model,
        color=colors[i],
        marker='o',
        markerfacecolor='white',
        markeredgecolor=colors[i],
        markeredgewidth=1.8,    # Толще обводка маркеров
        linewidth=3.0,          # Толще линия профиля
        zorder=3
    )

    # Заливка: чуть более непрозрачная, чтобы цвет был заметен
    ax.fill(angles, values, alpha=0.15, color=colors[i], zorder=2)

# ============================================================
# Подписи осей — крупные, читаемые, не жирные
# ============================================================
ax.set_xticks(angles[:-1])
ax.tick_params(axis='x', pad=45)  # Больше отступ

ax.set_xticklabels(
    labels,
    fontsize=21,
    fontweight='semibold',          # Без жирности
    color='#000000',              # Чёрный для контраста
    ha='center',
    va='center',
    fontfamily='sans-serif'
)

# ============================================================
# Легенда
# ============================================================
legend = plt.legend(
    loc='upper right',
    bbox_to_anchor=(1.4, 1.2),
    fontsize=20,
    frameon=True,
    shadow=True,
    edgecolor='#cccccc',
    facecolor='white',
    ncol=1
)

for text in legend.get_texts():
    text.set_fontweight('normal')

# ============================================================
# Сохранение
# ==========================================================
plt.savefig(
    "main.pdf",
    dpi=300,
    bbox_inches='tight',
    facecolor=fig.get_facecolor(),
    edgecolor='none'
)
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

# ----------------------------
# 1. DATA PREPARATION
# ----------------------------
df_total_no_cetell = df_total[
    ~df_total['test'].isin(['16pf', 'asq', 'bfi-2'])
].copy()

df_total_no_cetell = df_total_no_cetell[
    df_total_no_cetell['scale'] != 'error'
].copy()

stable = (
    df_total_no_cetell
    .groupby(['test', 'scale', 'interpretation_mean'])
    .size()
    .reset_index(name='frequency')
)

stable['scale_total_freq'] = stable.groupby(['test', 'scale'])['frequency'].transform('sum')

interpretation_order = {
    'Low': 0, 'Below Average': 0,
    'Medium': 1, 'Average': 1,
    'High': 2, 'Above Average': 2
}
stable['int_rank'] = stable['interpretation_mean'].map(interpretation_order).fillna(99)

stable_sorted = stable.sort_values(
    by=['scale_total_freq', 'test', 'scale', 'int_rank'],
    ascending=[False, True, True, True]
).reset_index(drop=True)

stable_sorted['group_name'] = stable_sorted['test'] + " → " + stable_sorted['scale']
stable_sorted['full_result'] = stable_sorted['group_name'] + " (" + stable_sorted['interpretation_mean'] + ")"

# ----------------------------
# 2. SPACING TRICK
# ----------------------------
plot_data = []
current_group = None
space_counter = 0

for i, row in stable_sorted.iterrows():
    if current_group is not None and row['group_name'] != current_group:
        space_label = " " * space_counter
        space_counter += 1
        plot_data.append({'full_result': space_label, 'frequency': 0, 'interpretation_mean': None})

    plot_data.append(row.to_dict())
    current_group = row['group_name']

df_plot = pd.DataFrame(plot_data)
df_plot['full_result'] = pd.Categorical(df_plot['full_result'], categories=df_plot['full_result'], ordered=True)

# ----------------------------
# 3. COLOR MAPPING
# ----------------------------
RED_WORDS = ['extraversion', 'high neuroticism', 'invalid result', 'abnormal', 'high', 'high level', 'severe depression']
GREEN_WORDS = ['low anxiety', 'introversion', 'low neuroticism', 'valid results', 'normal', 'low', 'minimal depression']
YELLOW_WORDS = ['mild anxiety', 'mild depression', 'average zone', 'average level', 'average']
ORANGE_WORDS = ['moderate anxiety', 'borderline abnormal', 'moderate', 'moderate depression']

def get_color(interpretation):
    if pd.isna(interpretation):
        return '#D3D3D3'
    txt = interpretation.lower().strip()
    for word in ORANGE_WORDS:
        if word in txt: return '#F4A259'
    for word in RED_WORDS:
        if word in txt: return '#E76F51'
    for word in YELLOW_WORDS:
        if word in txt: return '#F2D16B'
    for word in GREEN_WORDS:
        if word in txt: return '#7BC67B'
    return '#7BC67B'

# ----------------------------
# 4. PLOT
# ----------------------------
sns.set_theme(style="whitegrid")
num_rows = len(df_plot)
plt.figure(figsize=(11, max(8, num_rows * 0.42)))

ax = sns.barplot(
    data=df_plot,
    x="frequency",
    y="full_result",
    width=0.85,
    color='gray'
)

# Применяем цвета
for patch, interp in zip(ax.patches, df_plot['interpretation_mean']):
    color = get_color(interp)
    if color != '#D3D3D3':
        patch.set_facecolor(color)
        patch.set_alpha(0.9)
        patch.set_edgecolor('white')
        patch.set_linewidth(1.2)
    else:
        patch.set_visible(False)

# ----------------------------
# 5. LABELS & VALUES (weight='normal')
# ----------------------------
max_val = df_plot['frequency'].max()
offset = max_val * 0.015

for p in ax.patches:
    width = p.get_width()
    if width > 0 and not np.isnan(width):
        ax.text(
            width + offset,
            p.get_y() + p.get_height() / 2,
            f'{int(width)}',
            va='center',
            ha='left',
            fontsize=12,
            weight='normal',
            color='#2c2c2c'
        )

# ----------------------------
# 6. FORMATTING & LEGEND (ВСЕ ПОДПИСИ НЕ ЖИРНЫЕ)
# ----------------------------
plt.xlabel("Frequency", size=16, labelpad=12, weight='normal')
plt.ylabel("", size=12, weight='normal')
ax.tick_params(axis='both', labelsize=13)

# 🔹 ПРИНУДИТЕЛЬНО убираем жирность со всех подписей осей
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight('normal')

# Убираем сетку на строках-пробелах
for i, label in enumerate(ax.get_yticklabels()):
    if label.get_text().strip() == "":
        ax.get_ygridlines()[i].set_visible(False)

# Легенда
legend_elements = [
    Patch(facecolor='#7BC67B', edgecolor='white', linewidth=1.2, label='low'),
    Patch(facecolor='#F2D16B', edgecolor='white', linewidth=1.2, label='mild/average'),
    Patch(facecolor='#F4A259', edgecolor='white', linewidth=1.2, label='moderate'),
    Patch(facecolor='#E76F51', edgecolor='white', linewidth=1.2, label='high')
]

legend = ax.legend(
    handles=legend_elements,
    loc='upper right',
    bbox_to_anchor=(0.98, 0.99),
    fontsize=13,
    frameon=True,
    facecolor='white',
    edgecolor='#cccccc',
    title="Interpretation Level",
    title_fontsize=14
)

# Убираем жирность в легенде
legend.get_title().set_fontweight('normal')
for text in legend.get_texts():
    text.set_fontweight('normal')

sns.despine(left=True, bottom=True)
plt.tight_layout()

plt.savefig('freq.pdf', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch

# ============================================================
# 1. НАСТРОЙКИ И ДАННЫЕ
# ============================================================
sns.set_theme(style="white")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'font.weight': 'normal',
    'axes.labelweight': 'normal'
})

# Факторы 16PF (в нижнем регистре, как в ваших данных)
catell_factors = [
    ('a', "reserved - warm"),
    ('b', "problem-solving"),
    ('c', "emotionally stable - reactive"),
    ('e', "deferential - dominant"),
    ('f', "serious - lively"),
    ('g', "expedient - rule-conscious"),
    ('h', "shy - bold"),
    ('i', "sensitive - unsentimental"),
    ('l', "trusting - vigilant"),
    ('m', "abstracted - practical"),
    ('n', "private - forthright"),
    ('o', "self-assured - apprehensive"),
    ('q1', "open-to-change - traditional"),
    ('q2', "self-reliant - group-oriented"),
    ('q3', "tolerates disorder - perfectionistic"),
    ('q4', "relaxed - tense")
]

# Создаём словари для подписей и цветов
factor_labels = {code: f"{code.upper()}: {desc}" for code, desc in catell_factors}
factor_codes = [code for code, _ in catell_factors]

#  Фильтруем данные: только 16PF и только нужные факторы
df_16pf = df_total[
    (df_total['test'] == '16pf') &
    (df_total['scale'].isin(factor_codes)) &
    (df_total['scale'] != 'error') &
    (df_total['mean_total_norm'].notna())
].copy()

# ============================================================
# 2. ПОДГОТОВКА К ОТРИСОВКЕ
# ============================================================
# Порядок моделей: сверху те, у которых есть самые высокие баллы
model_order = (df_16pf.groupby('model')['mean_total_norm']
               .max()
               .sort_values(ascending=False)
               .index.tolist())

# Цветовая палитра: каждому фактору свой цвет
colors = sns.color_palette("pastel", len(factor_codes))
factor_color_map = dict(zip(factor_codes, colors))

# ============================================================
# 3. ПОСТРОЕНИЕ ГРАФИКА
# ============================================================
fig, ax = plt.subplots(figsize=(14, len(model_order) * 0.65 + 2))

for i, model in enumerate(model_order):
    # Берем все 16 факторов для модели
    model_data = df_16pf[df_16pf['model'] == model]

    # set_index делает 'scale' индексом. reindex упорядочивает, dropna убирает пустые
    top_factors = model_data.set_index('scale').reindex(factor_codes).dropna()

    # Сортируем по значению: самый длинный бар будет нарисован первым (на заднем плане)
    top_factors = top_factors.sort_values('mean_total_norm', ascending=False)

    #  ИСПРАВЛЕНИЕ: scale теперь находится в индексе, поэтому используем row.name
    for scale, row in top_factors.iterrows():
        val = row['mean_total_norm']

        ax.barh(
            y=i,
            width=val,
            height=0.75,
            left=0,
            color=factor_color_map[scale],
            alpha=0.85,
            edgecolor='white',
            linewidth=1.0,
            zorder=3
        )

# ============================================================
# 4. ОФОРМЛЕНИЕ
# ============================================================
ax.set_yticks(range(len(model_order)))
ax.set_yticklabels(model_order, fontsize=11)
ax.invert_yaxis()

ax.set_xlabel('Normalized score', fontsize=12)
ax.set_xlim(0, 105)
ax.grid(axis='x', linestyle='--', alpha=0.3)

# Убираем рамки
for spine in ['top', 'right', 'bottom', 'left']:
    ax.spines[spine].set_visible(False)
ax.tick_params(axis='y', length=0)

# ============================================================
# 5. ЛЕГЕНДА (ВЫРАВНИВАНИЕ ПО ВЕРХУ С ГРАФИКОМ)
# ============================================================
legend_handles = [
    Patch(facecolor=factor_color_map[code], edgecolor='white', linewidth=1.0, label=factor_labels[code])
    for code in factor_codes
]

legend = ax.legend(
    handles=legend_handles,
    loc='upper left',                    # Левый верхний угол легенды
    bbox_to_anchor=(1.02, 0.96),          # Координата 1.0 = самый верх осей
    bbox_transform=ax.transAxes,         # Привязка к осям, а не к фигуре
    fontsize=9,
    frameon=True,
    edgecolor='#cccccc',
    facecolor='white',
    ncol=1,
    title="16PF Factors",
    title_fontsize=10
)

legend.get_title().set_fontweight('normal')
for text in legend.get_texts():
    text.set_fontweight('normal')

# ============================================================
# 6. СОХРАНЕНИЕ
# ============================================================
plt.tight_layout()
plt.savefig('catell_16pf_profiles.pdf', dpi=300, bbox_inches='tight', facecolor='white')

plt.show()



import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib.patches import Patch

# ==========================================
# 1. DATA PREPARATION
# ==========================================
target_scales = ['epi_extraversion', 'epi_neuroticism']

df_traits = df_total[
    (df_total['scale'].isin(target_scales)) &
    (df_total['mean_total_norm'].notna())
].copy()

df_agg = df_traits.groupby(['model', 'scale'])['mean_total_norm'].mean().reset_index()
df_wide = df_agg.pivot(index='model', columns='scale', values='mean_total_norm').reset_index()
df_wide = df_wide.dropna(subset=['epi_extraversion', 'epi_neuroticism'])

# ==========================================
# 2. TEMPERAMENT CLASSIFICATION
# ==========================================
# Конвертация порогов из 0-24 в 0-100 шкалу:
# 0-8 → 0-33.3, 9-14 → 37.5-58.3, 15-24 → 62.5-100
# Для простоты используем 50 как границу экстраверсии/нейротизма

def assign_temperament(row):
    ext = row['epi_extraversion']
    neuro = row['epi_neuroticism']

    if ext >= 50 and neuro < 50:
        return 'Sanguine'      # Stable + Extraverted
    elif ext < 50 and neuro < 50:
        return 'Phlegmatic'    # Stable + Introverted
    elif ext < 50 and neuro >= 50:
        return 'Melancholic'   # Unstable + Introverted
    else:
        return 'Choleric'      # Unstable + Extraverted

df_wide['temperament'] = df_wide.apply(assign_temperament, axis=1)

# ==========================================
# 3. STYLE SETTINGS
# ==========================================
sns.set_theme(style="white")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.weight': 'normal',
    'axes.labelweight': 'normal',
    'font.size': 10
})

fig, ax = plt.subplots(figsize=(10, 10))

# ==========================================
# 4. BACKGROUND ZONES (DIFFERENT COLORS PER QUADRANT)
# ==========================================
# Phlegmatic (Stable + Introverted): bottom-left
ax.axvspan(0, 50, 0, 0.5, facecolor='#cce5ff', alpha=0.35, zorder=0)

# Sanguine (Stable + Extraverted): bottom-right
ax.axvspan(50, 100, 0, 0.5, facecolor='#7BC67B', alpha=0.35, zorder=0)

# Melancholic (Unstable + Introverted): top-left
ax.axvspan(0, 50, 0.5, 1.0, facecolor='#F2D16B', alpha=0.35, zorder=0)

# Choleric (Unstable + Extraverted): top-right
ax.axvspan(50, 100, 0.5, 1.0, facecolor='#E76F51', alpha=0.35, zorder=0)

# Quadrant Labels
ax.text(25, 25, 'Phlegmatic', ha='center', va='center', fontsize=11, style='italic', color='#555555', alpha=0.7)
ax.text(75, 25, 'Sanguine', ha='center', va='center', fontsize=11, style='italic', color='#555555', alpha=0.7)
ax.text(25, 75, 'Melancholic', ha='center', va='center', fontsize=11, style='italic', color='#555555', alpha=0.7)
ax.text(75, 75, 'Choleric', ha='center', va='center', fontsize=11, style='italic', color='#555555', alpha=0.7)

# ==========================================
# 5. PLOT POINTS (colored by MODEL)
# ==========================================
n_models = len(df_wide['model'].unique())
model_palette = sns.color_palette("tab20", n_models)
model_color_map = dict(zip(df_wide['model'].unique(), model_palette))

sns.scatterplot(
    data=df_wide,
    x='epi_extraversion',
    y='epi_neuroticism',
    hue='model',
    palette=model_color_map,
    s=140,
    ax=ax,
    edgecolor='black',
    linewidth=0.6,
    alpha=0.9,
    zorder=3
)

ax.get_legend().remove()

# ==========================================
# 6. AXES AND GRID
# ==========================================
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)

ax.set_xlabel('Extraversion', fontsize=12)
ax.set_ylabel('Neuroticism', fontsize=12)

# Threshold lines at 50
ax.axhline(50, color='gray', linestyle='--', linewidth=0.8, alpha=0.5, zorder=1)
ax.axvline(50, color='gray', linestyle='--', linewidth=0.8, alpha=0.5, zorder=1)

ax.grid(True, linestyle=':', alpha=0.3, zorder=0)
ax.spines[['top', 'right']].set_visible(False)

# ==========================================
# 7. CUSTOM LEGEND (MODELS with unique colors)
# ==========================================
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w',
               markerfacecolor=model_color_map[model],
               markersize=10, label=model,
               markeredgecolor='black', markeredgewidth=0.6)
    for model in df_wide['model'].unique()
]

legend = ax.legend(
    handles=legend_elements,
    loc='upper left',
    bbox_to_anchor=(1, 1),
    title='Model',
    title_fontsize=11,
    fontsize=8,
    frameon=True,
    edgecolor='#cccccc',
    facecolor='white',
    ncol=1
)

legend.get_title().set_fontweight('normal')
for t in legend.get_texts():
    t.set_fontweight('normal')

plt.tight_layout()
plt.savefig('temperament.pdf', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

asq_dominant

import pandas as pd

def print_parallel_profiles(df: pd.DataFrame, model1: str, model2: str):
    # 1. Выделяем нужные колонки для каждой модели
    cols = ['scale', 'mean_total', 'mean_total_norm', 'interpretation_mean']
    p1 = df[df['model'] == model1][cols].copy()
    p2 = df[df['model'] == model2][cols].copy()

    # Переименовываем, чтобы избежать конфликта при объединении
    p1.columns = ['scale', 'raw_1', 'norm_1', 'interp_1']
    p2.columns = ['scale', 'raw_2', 'norm_2', 'interp_2']

    # 2. Объединяем по названию шкалы (outer join сохранит уникальные шкалы обеих моделей)
    comp = pd.merge(p1, p2, on='scale', how='outer')

    # 3. Сортируем по среднему нормализованному баллу для упорядоченного вывода
    comp['sort_key'] = (comp['norm_1'].fillna(0) + comp['norm_2'].fillna(0)) / 2
    comp = comp.sort_values('sort_key', ascending=False).drop(columns=['sort_key']).reset_index(drop=True)

    # 4. Форматируем и выводим
    print(f"\n{'='*115}")
    print(f"📊 Параллельный профиль: {model1}  VS  {model2}")
    print(f"{'='*115}")
    print(f"{'Шкала':<28} | {model1[:14]:<15} {'%':<4} | {'Трактовка':<26} || {model2[:14]:<15} {'%':<4} | {'Трактовка':<26}")
    print("-" * 115)

    for _, row in comp.iterrows():
        # Безопасное форматирование с обработкой NaN
        val1 = f"{row['norm_1']:.1f}%" if pd.notna(row['norm_1']) else " - "
        interp1 = row['interp_1'] if pd.notna(row['interp_1']) else ""

        val2 = f"{row['norm_2']:.1f}%" if pd.notna(row['norm_2']) else " - "
        interp2 = row['interp_2'] if pd.notna(row['interp_2']) else ""

        print(f"{row['scale']:<28} | {val1:<19} | {interp1:<26} || {val2:<19} | {interp2:<26}")

print_parallel_profiles(df_total, "gpt-5.2", "gpt-5.2-chat-latest")

scale_name = "lie scale"

profile = df_total[df_total['scale'] == scale_name][
    ['model', 'test','scale', 'mean_total', 'interpretation_mean', 'mean_total_norm']
].sort_values('mean_total_norm', ascending=False)

profile

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib.patches import Patch

# ==========================================
# 1. ПОДГОТОВКА ДАННЫХ
# ==========================================
# Загружаем данные из предоставленного CSV
df = asq_dominant.copy()

# Убираем строки, где интерпретация 'error' (например, deepseek-r1)
# Иначе map вернет NaN, и добавление джиттера вызовет ошибку
df = df[df['interpretation'] != 'error'].copy()

# Также убираем возможные пустые значения на всякий случай
df = df.dropna(subset=['interpretation'])

# ==========================================
# 2. КООРДИНАТЫ ТИПОВ ПРИВЯЗАННОСТИ
# ==========================================
coords = {
    'secure attached': (25, 25),
    'fearful avoidant': (75, 75),
    'anxious preoccupied': (25, 75),
    'dismissive avoidant': (75, 25)
}

# Разделяем x и y нормально
df['x'] = df['interpretation'].map(lambda x: coords[x][0])
df['y'] = df['interpretation'].map(lambda x: coords[x][1])

# ==========================================
# ДОБАВЛЯЕМ JITTER
# ==========================================
# Чтобы точки не лежали друг на друге

np.random.seed(42)

df['x'] = df['x'] + np.random.uniform(-6, 6, len(df))
df['y'] = df['y'] + np.random.uniform(-6, 6, len(df))

# ==========================================
# 3. СТИЛЬ (БЕЗ ЖИРНОГО ШРИФТА)
# ==========================================
sns.set_theme(style="white")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.weight': 'normal',
    'axes.labelweight': 'normal',
    'font.size': 10
})

fig, ax = plt.subplots(figsize=(10, 10))

# ==========================================
# 4. ФОНОВЫЕ КВАДРАНТЫ (ВАШИ ЦВЕТА)
# ==========================================
# Secure (низкая тревога, низкое избегание)
ax.axvspan(0, 50, 0, 0.5, facecolor='#7BC67B', alpha=0.25, zorder=0)

# Dismissive-Avoidant (низкая тревога, высокое избегание)
ax.axvspan(50, 100, 0, 0.5, facecolor='#6FA8DC', alpha=0.25, zorder=0)

# Anxious-Preoccupied (высокая тревога, низкое избегание)
ax.axvspan(0, 50, 0.5, 1, facecolor='#F2D16B', alpha=0.25, zorder=0)

# Fearful-Avoidant (высокая тревога, высокое избегание)
ax.axvspan(50, 100, 0.5, 1, facecolor='#E76F51', alpha=0.25, zorder=0)

# ==========================================
# 5. ПОДПИСИ КВАДРАНТОВ
# ==========================================
ax.text(25, 25, 'Secure', ha='center', va='center',
        fontsize=14, style='italic', alpha=0.7, weight='normal')
ax.text(75, 25, 'Dismissive-Avoidant', ha='center', va='center',
        fontsize=14, style='italic', alpha=0.7, weight='normal')
ax.text(25, 75, 'Anxious-Preoccupied', ha='center', va='center',
        fontsize=14, style='italic', alpha=0.7, weight='normal')
ax.text(75, 75, 'Fearful-Avoidant', ha='center', va='center',
        fontsize=14, style='italic', alpha=0.7, weight='normal')

# ==========================================
# 6. ЦВЕТА ДЛЯ МОДЕЛЕЙ
# ==========================================
palette = sns.color_palette("tab20", len(df['model'].unique()))
color_map = dict(zip(df['model'].unique(), palette))

# ==========================================
# 7. SCATTER PLOT
# ==========================================
sns.scatterplot(
    data=df,
    x='x',
    y='y',
    hue='model',
    palette=color_map,
    s=260,
    edgecolor='white',      # Белая обводка для чёткости
    linewidth=1.0,
    alpha=0.95,
    ax=ax,
    zorder=3
)

# ==========================================
# 8. ЦЕНТРАЛЬНЫЕ ЛИНИИ
# ==========================================
ax.axhline(50, linestyle='--', color='gray', alpha=0.5, zorder=1)
ax.axvline(50, linestyle='--', color='gray', alpha=0.5, zorder=1)

# ==========================================
# 9. ОФОРМЛЕНИЕ ОСЕЙ
# ==========================================
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)

ax.set_xticks([])   # Убираем числовые подписи
ax.set_yticks([])

ax.set_xlabel('Avoidance', fontsize=12, weight='normal')
ax.set_ylabel('Anxiety', fontsize=12, weight='normal')

# Убираем лишние рамки
ax.spines[['top', 'right']].set_visible(False)

# ==========================================
# 10. ЛЕГЕНДА (БЕЗ ЖИРНОГО ШРИФТА)
# ==========================================

legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w',
               markerfacecolor=model_color_map[model],
               markersize=10, label=model,
               markeredgecolor='black', markeredgewidth=0.6)
    for model in df_wide['model'].unique()
]

legend = ax.legend(
    handles=legend_elements,
    loc='upper left',
    bbox_to_anchor=(1, 1),
    title='Model',
    title_fontsize=11,
    fontsize=8,
    frameon=True,
    edgecolor='#cccccc',
    facecolor='white',
    ncol=1
)

legend.get_title().set_fontweight('normal')
for t in legend.get_texts():
    t.set_fontweight('normal')


# ==========================================
# 11. СОХРАНЕНИЕ
# ==========================================
plt.tight_layout()
plt.savefig(
    'attachment_styles.pdf',
    dpi=300,
    bbox_inches='tight',
    facecolor='white'
)

plt.show()

print(df_total.columns)
