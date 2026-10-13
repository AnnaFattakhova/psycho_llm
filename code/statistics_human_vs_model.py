# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

"""**r** — коэффициент корреляции(от -1 до +1; Если r = 0.88 → когда X растёт, Y почти всегда растёт тоже.)

| r       | Сила          |
| ------- | ------------- |
| 0.1–0.3 | слабая        |
| 0.3–0.7 | умеренная     |
| 0.7–1   | очень сильная |

**p** (случайность или нет) < 0.05 → значимо

**ci_low и ci_high** - то 95% доверительный интервал (в каком диапазоне находится "настоящая" корреляция): Мы на 95% уверены, что реальная корреляция лежит между ci_low и ci_high. Если интервал: не пересекает 0;  весь положительный → связь точно есть.



"""

import chardet
with open("/content/hackaton_dataset.csv", "rb") as f:
    enc = chardet.detect(f.read(50000))
    print(f"Рекомендуемая кодировка: {enc['encoding']} (уверенность: {enc['confidence']})")

df = pd.read_csv(
    "/content/hackaton_dataset.csv",
    encoding='ascii',
    engine='python',
    on_bad_lines='skip'
)

df

# Считаем среднее по каждой модели, тесту и шкале
mean_scores = (
    df.groupby(["model", "test", "scale"])["model_score"]
      .mean()
      .reset_index()
)

mean_scores

print(df.isna().sum())

df.info()

print(df.dtypes)

df["human_score"] = pd.to_numeric(df["human_score"], errors="coerce")
df["model_score"] = pd.to_numeric(df["model_score"], errors="coerce")

"""# ***Корреляция для каждого человека отдельно***"""

results = []

for person in df["person_id"].unique():
    temp = df[df["person_id"] == person]

    if len(temp) >= 3:
        r, p = pearsonr(temp["human_score"], temp["model_score"])
        n = len(temp)

        # доверительный интервал через преобразование Фишера
        z = np.arctanh(r)
        se = 1 / np.sqrt(n - 3)
        z_crit = 1.96
        lo = z - z_crit * se
        hi = z + z_crit * se

        ci_low = np.tanh(lo)
        ci_high = np.tanh(hi)

        results.append([person, n, r, p, ci_low, ci_high])

results_df = pd.DataFrame(
    results,
    columns=["person_id", "n", "r", "p_value", "ci_low", "ci_high"]
)

def interpret_r(r):
    r = abs(r)
    if r < 0.3:
        return "weak"
    elif r < 0.7:
        return "moderate"
    elif r < 1:
        return "strong"

results_df["strength"] = results_df["r"].apply(interpret_r)

print(results_df)

"""модели очень хорошо сохраняют структуру психологического профиля человека, то есть если человек тревожный, то и модель будет более тревожная

# ***Корреляция по всей выборке***
"""

# Общая корреляция
r, p = pearsonr(df["human_score"], df["model_score"])
n = len(df)

# Доверительный интервал через преобразование Фишера
z = np.arctanh(r)
se = 1 / np.sqrt(n - 3)
z_crit = 1.96
lo = z - z_crit * se
hi = z + z_crit * se

ci_low = np.tanh(lo)
ci_high = np.tanh(hi)

# Функция интерпретации
def interpret_r(r):
    r = abs(r)
    if r < 0.3:
        return "weak"
    elif r < 0.7:
        return "moderate"
    elif r < 1:
        return "strong"


overall_df = pd.DataFrame({
    "n": [n],
    "r": [r],
    "p_value": [p],
    "ci_low": [ci_low],
    "ci_high": [ci_high],
    "strength": [interpret_r(r)]
})

print(overall_df)

"""# ***Корреляция по тестам***

Мы берём один тест и смотрим насколько совпадают оценки модели и человека. насколько хорошо модель предсказывает именно этот тест?
"""

results = []

for test in df["test"].unique():

    temp = df[df["test"] == test]

    n = len(temp)

    # если слишком мало данных — пропускаем
    if n < 3:
        continue

    r, p = pearsonr(temp["human_score"], temp["model_score"])

    # доверительный интервал считаем только если r не идеальная
    if abs(r) < 0.999:

        z = np.arctanh(r)
        se = 1 / np.sqrt(n - 3)

        lo = z - 1.96 * se
        hi = z + 1.96 * se

        ci_low = np.tanh(lo)
        ci_high = np.tanh(hi)

    else:

        ci_low = np.nan
        ci_high = np.nan

    results.append([
        test,
        n,
        r,
        p,
        ci_low,
        ci_high
    ])

test_corr = pd.DataFrame(
    results,
    columns=[
        "test",
        "n",
        "r",
        "p_value",
        "ci_low",
        "ci_high"
    ]
)

test_corr = test_corr.sort_values("r", ascending=False)

pd.set_option("display.max_rows", None)

print(test_corr)

"""# ***Влияние тестов Leave-one-test-out***

какие тесты сильнее всего влияют на общую корреляцию между моделью и человеком.
"""

base_r, _ = pearsonr(df["human_score"], df["model_score"])

influence = []

for test in df["test"].unique():

    temp = df[df["test"] != test]   # убираем один тест

    r, _ = pearsonr(temp["human_score"], temp["model_score"])

    influence.append([
        test,
        r,
        base_r - r
    ])

influence_df = pd.DataFrame(
    influence,
    columns=["test", "r_without_test", "change_in_r"]
)

influence_df = influence_df.sort_values(
    "change_in_r",
    ascending=False
)

pd.set_option('display.max_rows', None)

print(influence_df)

"""## ***Графики***"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

from scipy.stats import pearsonr
import pandas as pd

scale_results = []

for (test, scale), temp in df.groupby(["test", "scale"]):

    temp = temp.dropna(subset=["human_score", "model_score"])

    if len(temp) >= 3:
        r, p = pearsonr(
            temp["human_score"],
            temp["model_score"]
        )

        scale_results.append([
            test,
            scale,
            len(temp),
            r,
            p
        ])

scale_corr = pd.DataFrame(
    scale_results,
    columns=["test", "scale", "n", "r", "p_value"]
)

# По каждому тесту выбираем шкалу с максимальным |r|
target_scales = (
    scale_corr
    .sort_values("r", key=abs, ascending=False)
    .groupby("test")
    .first()
    .reset_index()
)

print(target_scales)

# =========================
# НАСТРОЙКИ
# =========================

sns.set_theme(style="white")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.2
})

# =========================
# ВЫБИРАЕМ ПО ОДНОЙ ШКАЛЕ НА КАЖДЫЙ ТЕСТ
# =========================

scale_results = []

for (test, scale), temp in df.groupby(["test", "scale"]):

    temp = temp.dropna(subset=["human_score", "model_score"])

    if len(temp) >= 3:
        r, p = pearsonr(
            temp["human_score"],
            temp["model_score"]
        )

        scale_results.append([
            test,
            scale,
            len(temp),
            r,
            p
        ])

scale_corr = pd.DataFrame(
    scale_results,
    columns=["test", "scale", "n", "r", "p_value"]
)

selected_scales = (
    scale_corr
    .sort_values("r", key=abs, ascending=False)
    .groupby("test")
    .first()
    .reset_index()
)

target_scales = {
    row["test"]: row["scale"]
    for _, row in selected_scales.iterrows()
}

# =========================
# ПОДПИСИ
# =========================

labels = {
    "scl-90": "SCL-90",
    "bai": "BAI",
    "hads": "HADS",
    "bdi": "BDI",
    "gad-7": "GAD-7",
    "epi": "EPI",
    "bfi-2": "BFI-2",
    "16pf": "16PF",
    "sd3": "SD3",
    "bdhi": "BDHI",
    "asq": "ASQ"
}

# =========================
# ЦВЕТА
# =========================

palette = sns.color_palette("husl", 9)

human_color = palette[6]
model_color = palette[7]

# =========================
# ГРАФИКИ
# =========================

people = sorted(df["person_id"].dropna().unique())

# 5 графиков в одном ряду
cols = 5
rows = int(np.ceil(len(people) / cols))

fig, axes = plt.subplots(
    rows,
    cols,
    figsize=(cols * 4.5, rows * 3.8)
)

axes = np.array(axes).reshape(-1)

for i, person in enumerate(people):

    ax = axes[i]

    temp_person = df[df["person_id"] == person]

    human_values = []
    model_values = []
    test_labels = []

    for test, scale in target_scales.items():

        row = temp_person[
            (temp_person["test"] == test) &
            (temp_person["scale"] == scale)
        ]

        if not row.empty:

            human_values.append(
                row["human_score"].iloc[0]
            )

            model_values.append(
                row["model_score"].iloc[0]
            )

            test_labels.append(
                labels.get(test, test.upper())
            )

    x = np.arange(len(test_labels))

    # Human
    ax.plot(
        x,
        human_values,
        color=human_color,
        marker="o",
        linewidth=2.2,
        markersize=5,
        label="Human"
    )

    # Model
    ax.plot(
        x,
        model_values,
        color=model_color,
        marker="o",
        linewidth=2.2,
        markersize=5,
        label="Model"
    )

    # =========================
    # ОФОРМЛЕНИЕ
    # =========================

    model_name = temp_person["model"].dropna().iloc[0]

    ax.set_title(
        f"Person {person} ({model_name})",
        fontsize=12,
        fontweight="bold",
       pad=10
)

    ax.set_xticks(x)
    ax.set_xticklabels(
        test_labels,
        rotation=35,
        ha="right",
        fontsize=9
    )

    # ШКАЛА ОТ 0 ДО 30
    ax.set_ylim(0, 50)

    # БОЛЬШЕ ЗАСЕЧЕК
    ax.set_yticks(np.arange(0, 50, 5))

    ax.tick_params(
        axis="y",
        labelsize=9
    )

    ax.grid(
        axis="y",
        linestyle=":",
        linewidth=0.8,
        alpha=0.5
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# Убираем пустые графики
for i in range(len(people), len(axes)):
    axes[i].axis("off")

# =========================
# ОБЩАЯ ЛЕГЕНДА
# =========================

handles, legend_labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    legend_labels,
    loc="upper center",
    ncol=2,
    bbox_to_anchor=(0.5, 1.015),
    frameon=False,
    fontsize=20
)


plt.subplots_adjust(
    top=0.91,
    bottom=0.12,
    hspace=0.55,
    wspace=0.30
)


plt.savefig(
    "human-model.pdf",
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
    pad_inches=0.3
)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# =========================
# ВЫБИРАЕМ ЛЮДЕЙ
# =========================

selected_people = [4, 10, 13, 15]

# =========================
# ВЫБИРАЕМ ПО ОДНОЙ ШКАЛЕ НА КАЖДЫЙ ТЕСТ
# =========================

scale_results = []

for (test, scale), temp in df.groupby(["test", "scale"]):

    temp = temp.dropna(subset=["human_score", "model_score"])

    if len(temp) >= 3:
        r, p = pearsonr(
            temp["human_score"],
            temp["model_score"]
        )

        scale_results.append([
            test,
            scale,
            len(temp),
            r,
            p
        ])

scale_corr = pd.DataFrame(
    scale_results,
    columns=["test", "scale", "n", "r", "p_value"]
)

selected_scales = (
    scale_corr
    .sort_values("r", key=abs, ascending=False)
    .groupby("test")
    .first()
    .reset_index()
)

target_scales = {
    row["test"]: row["scale"]
    for _, row in selected_scales.iterrows()
}

# =========================
# ПОДПИСИ
# =========================

labels = {
    "scl-90": "SCL-90",
    "bai": "BAI",
    "hads": "HADS",
    "bdi": "BDI",
    "gad-7": "GAD-7",
    "epi": "EPI",
    "bfi-2": "BFI-2",
    "16pf": "16PF",
    "sd3": "SD3",
    "bdhi": "BDHI",
    "asq": "ASQ"
}

# =========================
# ЦВЕТА
# =========================

palette = sns.color_palette("husl", 9)

human_color = palette[6]
model_color = palette[7]

# =========================
# ГРАФИКИ
# =========================

people = selected_people

cols = 4
rows = 1  # ← был 2, стал 1

fig, axes = plt.subplots(
    rows,
    cols,
    figsize=(cols * 4.5, rows * 3.8)
)

axes = np.array(axes).reshape(-1)

for i, person in enumerate(people):

    ax = axes[i]

    temp_person = df[df["person_id"] == person]

    human_values = []
    model_values = []
    test_labels = []

    for test, scale in target_scales.items():

        row = temp_person[
            (temp_person["test"] == test) &
            (temp_person["scale"] == scale)
        ]

        if not row.empty:

            human_values.append(
                row["human_score"].iloc[0]
            )

            model_values.append(
                row["model_score"].iloc[0]
            )

            test_labels.append(
                labels.get(test, test.upper())
            )

    x = np.arange(len(test_labels))

    # =========================
    # HUMAN
    # =========================

    ax.plot(
        x,
        human_values,
        color=human_color,
        marker="o",
        linewidth=2.2,
        markersize=5,
        label="Human"
    )

    # =========================
    # MODEL
    # =========================

    ax.plot(
        x,
        model_values,
        color=model_color,
        marker="o",
        linewidth=2.2,
        markersize=5,
        label="Model"
    )

    # =========================
    # ЗАГОЛОВОК
    # =========================

    model_name = temp_person["model"].dropna().iloc[0]

    ax.set_title(
        f"Person {person} ({model_name})",
        fontsize=12,
        fontweight="bold",
        pad=10
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        test_labels,
        rotation=35,
        ha="right",
        fontsize=9
    )

    # =========================
    # ШКАЛА 0–35
    # =========================

    ax.set_ylim(0, 35)

    ax.set_yticks(
        np.arange(0, 36, 5)
    )

    ax.tick_params(
        axis="y",
        labelsize=9
    )

    # =========================
    # GRID
    # =========================

    ax.grid(
        axis="y",
        linestyle=":",
        linewidth=0.8,
        alpha=0.5
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ============================================================
# НАСТРАИВАЕМ РАССТОЯНИЯ
# ============================================================

plt.subplots_adjust(
    top=0.85,
    bottom=0.22,
    wspace=0.30
)


# ============================================================
# ОБЩАЯ ЛЕГЕНДА
# ============================================================

handles, legend_labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    legend_labels,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.004),
    ncol=2,
    frameon=False,
    fontsize=12,
    columnspacing=4.0,
    handletextpad=0.8,
    prop={"weight": "bold", "size": 12}
)


# ============================================================
# СОХРАНЕНИЕ
# ============================================================

plt.savefig(
    "human-model-selected.pdf",
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
    pad_inches=0.3
)

plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Загружаем таблицу
df = pd.read_csv("api_human.csv")

# Переводим оценки в числа
df["api_mean_total"] = pd.to_numeric(
    df["api_mean_total"].astype(str).str.replace(",", ".", regex=False),
    errors="coerce"
)

df["human_score"] = pd.to_numeric(
    df["human_score"].astype(str).str.replace(",", ".", regex=False),
    errors="coerce"
)

# Разница между API и человеком
df["diff"] = df["api_mean_total"] - df["human_score"]

# Сводная таблица:
# строки = модели
# столбцы = шкалы
# значения = средняя разница API - Human
pivot = df.pivot_table(
    values="diff",
    index="model",
    columns="scale",
    aggfunc="mean"
)

# Размеры шрифтов
plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14
})

# Фигура
plt.figure(figsize=(20, 8))

# Палитра
cmap = sns.diverging_palette(
    240, 15,
    s=75,
    l=65,
    as_cmap=True
)

# Heatmap
ax = sns.heatmap(
    pivot,
    annot=True,
    cmap=cmap,
    center=0,
    annot_kws={"size": 10}
)

# Подписи осей
plt.xlabel("Scale", fontsize=18)
plt.ylabel("Model", fontsize=18)

# Размер подписей
ax.tick_params(axis="x", labelsize=14)
ax.tick_params(axis="y", labelsize=12)

# Colorbar
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=14)

plt.tight_layout()

# Сохранение
plt.savefig(
    "api_human_diff_heatmap.pdf",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# загружаем данные
df = pd.read_csv("api_interface.csv")

# переводим значения в числовой формат
df["api_mean"] = pd.to_numeric(df["api_mean"], errors="coerce")
df["interface_mean"] = pd.to_numeric(df["interface_mean"], errors="coerce")

# разница между оценкой API и оценкой через интерфейс
df["diff_model"] = df["api_mean"] - df["interface_mean"]

# сводная таблица
pivot = df.pivot_table(
    values="diff_model",
    index="model",
    columns="scale",
    aggfunc="mean"
)

# настройка размеров шрифтов
plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14
})

# фигура
plt.figure(figsize=(20, 8))

# палитра
cmap = sns.diverging_palette(
    240, 15,
    s=75,
    l=65,
    as_cmap=True
)

# heatmap
ax = sns.heatmap(
    pivot,
    annot=True,
    cmap=cmap,
    center=0,
    annot_kws={"size": 10}
)

# подписи осей
plt.xlabel("Scale", fontsize=18)
plt.ylabel("Model", fontsize=18)

# размеры подписей
ax.tick_params(axis="x", labelsize=14)
ax.tick_params(axis="y", labelsize=12)

# colorbar
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=14)

plt.tight_layout()

# сохранение
plt.savefig(
    "api_interface_diff_heatmap.pdf",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()

import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. LOAD DATA
# ==========================================

df_main = pd.read_csv("hackaton_dataset.csv")
df_interface = pd.read_csv("api_interface.csv")


# ==========================================
# 2. CONVERT TO NUMERIC
# ==========================================

df_main["model_score"] = pd.to_numeric(
    df_main["model_score"],
    errors="coerce"
)

df_main["human_score"] = pd.to_numeric(
    df_main["human_score"],
    errors="coerce"
)

df_interface["api_mean"] = pd.to_numeric(
    df_interface["api_mean"],
    errors="coerce"
)

df_interface["interface_mean"] = pd.to_numeric(
    df_interface["interface_mean"],
    errors="coerce"
)


# ==========================================
# 3. ABSOLUTE ERRORS
# ==========================================

df_main["abs_error_human"] = (
    df_main["model_score"]
    - df_main["human_score"]
).abs()

df_interface["abs_error_interface"] = (
    df_interface["api_mean"]
    - df_interface["interface_mean"]
).abs()


# ==========================================
# 4. MAE FOR EACH MODEL
# ==========================================

mae_by_model = (
    df_main
    .groupby("model")["abs_error_human"]
    .mean()
)

print("MAE by model:")
print(mae_by_model)


# ==========================================
# 5. AGGREGATION
# ==========================================

human_error = (
    df_main
    .groupby(
        ["model", "test"],
        as_index=False
    )["abs_error_human"]
    .mean()
    .rename(columns={
        "abs_error_human": "error"
    })
)

human_error["comparison"] = "vs human_score"


interface_error = (
    df_interface
    .groupby(
        ["model", "test"],
        as_index=False
    )["abs_error_interface"]
    .mean()
    .rename(columns={
        "abs_error_interface": "error"
    })
)

interface_error["comparison"] = "vs interface"


plot_data = pd.concat(
    [
        human_error,
        interface_error
    ],
    ignore_index=True
)


# ==========================================
# 6. STYLE SETTINGS
# ==========================================

sns.set_style("whitegrid")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.weight": "normal",
    "axes.labelweight": "normal",
    "font.size": 16,
    "axes.titlesize": 22,
    "axes.labelsize": 18,
    "xtick.labelsize": 15,
    "ytick.labelsize": 17,
    "legend.fontsize": 20,
})


# ==========================================
# 7. COLOR PALETTE
# ==========================================

cmap = sns.diverging_palette(
    240, 15,
    s=75,
    l=65,
    as_cmap=True
)

negative_color = cmap(0.15)
positive_color = cmap(0.85)

comparison_palette = {
    "vs human_score": negative_color,
    "vs interface": positive_color
}


# ==========================================
# 8. FACETGRID
# ==========================================

g = sns.FacetGrid(
    plot_data,
    col="model",
    col_wrap=2,
    sharex=True,
    sharey=False,
    height=8.5,
    aspect=2.5
)


# ==========================================
# 9. DRAW FUNCTION
# ==========================================

def draw_barplot(data, **kwargs):

    ax = plt.gca()

    # sorting
    order = (
        data.groupby("test")["error"]
        .mean()
        .sort_values()
        .index
    )

    data = data.copy()

    data["test"] = pd.Categorical(
        data["test"],
        categories=order,
        ordered=True
    )

    data = data.sort_values("test")

    sns.barplot(
        data=data,
        x="error",
        y="test",
        hue="comparison",
        palette=comparison_palette,
        orient="h",
        dodge=True,
        ax=ax,
        edgecolor="white",
        linewidth=1.5,
        alpha=0.92
    )

    # ======================================
    # X AXIS LABEL
    # ======================================

    ax.set_xlabel(
        "MAE (Mean Absolute Error)",
        fontsize=18,
        fontweight="normal",
        labelpad=10
    )

    ax.set_ylabel("")

    # ======================================
    # Y AXIS — SCALE NAMES
    # ======================================

    ax.tick_params(
        axis="y",
        pad=10,
        labelsize=17
    )

    ax.tick_params(
        axis="x",
        labelsize=15,
        direction="out",
        length=6,
        width=1.2
    )

    # ======================================
    # X SCALE
    # ======================================

    ax.set_xticks([
        0, 5, 10, 15, 20, 25, 30
    ])

    ax.set_xlim(0, 30)

    for label in ax.get_xticklabels():
        label.set_fontweight("normal")

    for label in ax.get_yticklabels():
        label.set_fontweight("normal")

    # ======================================
    # GRID
    # ======================================

    ax.grid(
        axis="x",
        linestyle=":",
        alpha=0.5,
        linewidth=1.0,
        zorder=0
    )

    ax.grid(
        axis="y",
        visible=False
    )

    # ======================================
    # SPINES
    # ======================================

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)


# ==========================================
# 10. DRAW
# ==========================================

g.map_dataframe(draw_barplot)


# ==========================================
# 11. MODEL TITLES
# ==========================================

for ax, model_name in zip(
    g.axes.flatten(),
    plot_data["model"].unique()
):

    ax.set_title(
        str(model_name),
        fontsize=24,
        fontweight="bold",
        pad=12
    )


# ==========================================
# 12. REMOVE DUPLICATE LEGENDS
# ==========================================

for ax in g.axes.flatten():

    legend = ax.get_legend()

    if legend is not None:
        legend.remove()


# ==========================================
# 13. SHOW X LABELS ON EVERY GRAPH
# ==========================================

for ax in g.axes.flatten():

    ax.xaxis.set_tick_params(
        labelbottom=True
    )

    ax.xaxis.label.set_visible(True)


# ==========================================
# 14. GLOBAL LEGEND — НАВЕРХУ
# ==========================================

g.fig.legend(
    handles=[
        plt.Rectangle(
            (0, 0),
            1,
            1,
            color=negative_color
        ),
        plt.Rectangle(
            (0, 0),
            1,
            1,
            color=positive_color
        )
    ],
    labels=[
        "Model vs human",
        "API vs interface"
    ],
    loc="upper center",
    bbox_to_anchor=(0.5, 1.005),
    ncol=2,
    frameon=False,
    prop={
        "weight": "bold",
        "size": 25
    },
    columnspacing=2.5,
    handlelength=2.5,
    handleheight=1.5
)


# ==========================================
# 15. GRID SIZE
# ==========================================

n_models = len(
    plot_data["model"].unique()
)

n_cols = 2

n_rows = math.ceil(
    n_models / n_cols
)


# ==========================================
# 16. CENTER LAST GRAPH
# ==========================================

if n_models % n_cols == 1:

    axes = g.axes.flatten()

    last_ax = axes[n_models - 1]

    pos = last_ax.get_position()

    new_x0 = (
        1.0 - pos.width
    ) / 2

    last_ax.set_position([
        new_x0,
        pos.y0,
        pos.width,
        pos.height
    ])


# ==========================================
# 17. SPACING
# ==========================================

g.fig.subplots_adjust(
    top=0.92,
    bottom=0.055,
    hspace=0.45,
    wspace=0.28
)


# ==========================================
# 18. FIGURE SIZE
# ==========================================

g.fig.set_size_inches(
    28,
    n_rows * 8.5
)


# ==========================================
# 19. SAVE
# ==========================================

plt.savefig(
    "model_vs_human_vs_interface_test.pdf",
    bbox_inches="tight",
    dpi=300,
    facecolor="white",
    pad_inches=0.25
)

plt.show()

import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# 1. LOAD DATA
# ==========================================

df_main = pd.read_csv("hackaton_dataset.csv")
df_interface = pd.read_csv("api_interface.csv")


# ==========================================
# 2. CONVERT TO NUMERIC
# ==========================================

df_main["model_score"] = pd.to_numeric(
    df_main["model_score"],
    errors="coerce"
)

df_main["human_score"] = pd.to_numeric(
    df_main["human_score"],
    errors="coerce"
)

df_interface["api_mean"] = pd.to_numeric(
    df_interface["api_mean"],
    errors="coerce"
)

df_interface["interface_mean"] = pd.to_numeric(
    df_interface["interface_mean"],
    errors="coerce"
)


# ==========================================
# 3. ABSOLUTE ERRORS
# ==========================================

# Model vs human
df_main["abs_error_human"] = (
    df_main["model_score"]
    - df_main["human_score"]
).abs()

# API vs interface
df_interface["abs_error_interface"] = (
    df_interface["api_mean"]
    - df_interface["interface_mean"]
).abs()


# ==========================================
# 4. AGGREGATION BY MODEL + SCALE
# ==========================================

human_error = (
    df_main
    .groupby(
        ["model", "scale"],
        as_index=False
    )["abs_error_human"]
    .mean()
    .rename(columns={
        "abs_error_human": "error"
    })
)

human_error["comparison"] = "vs human_score"


interface_error = (
    df_interface
    .groupby(
        ["model", "scale"],
        as_index=False
    )["abs_error_interface"]
    .mean()
    .rename(columns={
        "abs_error_interface": "error"
    })
)

interface_error["comparison"] = "vs interface"


# Combine
plot_data = pd.concat(
    [
        human_error,
        interface_error
    ],
    ignore_index=True
)


# ==========================================
# 5. STYLE
# ==========================================

sns.set_style("whitegrid")

plt.rcParams.update({
    "font.weight": "normal",
    "axes.labelweight": "normal",
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 14
})


# ==========================================
# 6. COLORS
# ==========================================

cmap = sns.diverging_palette(
    240,
    15,
    s=75,
    l=65,
    as_cmap=True
)

negative_color = cmap(0.15)
positive_color = cmap(0.85)

comparison_palette = {
    "vs human_score": negative_color,
    "vs interface": positive_color
}


# ==========================================
# 7. FACETGRID
# ==========================================

g = sns.FacetGrid(
    plot_data,
    col="model",
    col_wrap=2,
    sharex=True,
    sharey=False,
    height=10,
    aspect=3.2
)


# ==========================================
# 8. DRAW FUNCTION
# ==========================================

def draw_barplot(data, **kwargs):

    ax = plt.gca()

    # --------------------------------------
    # Sorting scales by mean error
    # --------------------------------------

    order = (
        data
        .groupby("scale")["error"]
        .mean()
        .sort_values()
        .index
    )

    data = data.copy()

    data["scale"] = pd.Categorical(
        data["scale"],
        categories=order,
        ordered=True
    )

    data = data.sort_values("scale")


    # --------------------------------------
    # BARPLOT
    # --------------------------------------

    sns.barplot(
        data=data,
        x="error",
        y="scale",
        hue="comparison",
        palette=comparison_palette,
        orient="h",
        dodge=True,
        ax=ax,
        edgecolor="white",
        linewidth=1.3,
        alpha=0.92
    )


    # --------------------------------------
    # X AXIS
    # --------------------------------------

    ax.set_xlabel(
        "MAE",
        fontsize=12,
        weight="normal",
        labelpad=12
    )

    ax.set_ylabel("")

    ax.tick_params(
        axis="y",
        pad=14
    )


    # --------------------------------------
    # FONT
    # --------------------------------------

    for label in ax.get_xticklabels():
        label.set_fontsize(13)
        label.set_fontweight("normal")

    for label in ax.get_yticklabels():
        label.set_fontsize(13)
        label.set_fontweight("normal")


    # --------------------------------------
    # X AXIS RANGE
    # --------------------------------------

    ax.set_xticks([
        0, 5, 10, 15, 20, 25, 30
    ])

    ax.set_xlim(0, 30)


    # --------------------------------------
    # GRID
    # --------------------------------------

    ax.grid(
        axis="x",
        linestyle=":",
        alpha=0.4,
        linewidth=0.9,
        zorder=0
    )

    ax.grid(
        axis="y",
        visible=False
    )


    # --------------------------------------
    # SPINES
    # --------------------------------------

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)


# ==========================================
# 9. DRAW
# ==========================================

g.map_dataframe(draw_barplot)


# ==========================================
# 10. TITLES
# ==========================================

for ax, model_name in zip(
    g.axes.flatten(),
    plot_data["model"].unique()
):

    ax.set_title(
        str(model_name),
        fontsize=20,
        fontweight="bold",
        pad=30
    )


# ==========================================
# 11. REMOVE LEGENDS INSIDE PLOTS
# ==========================================

for ax in g.axes.flatten():

    leg = ax.get_legend()

    if leg is not None:
        leg.remove()


# ==========================================
# 12. SHOW X LABELS ON EVERY GRAPH
# ==========================================

for ax in g.axes.flatten():

    ax.xaxis.set_tick_params(
        labelbottom=True
    )

    ax.xaxis.label.set_visible(True)


# ==========================================
# 14. GLOBAL LEGEND — НАВЕРХУ
# ==========================================

g.fig.legend(
    handles=[
        plt.Rectangle(
            (0, 0),
            1,
            1,
            color=negative_color
        ),
        plt.Rectangle(
            (0, 0),
            1,
            1,
            color=positive_color
        )
    ],
    labels=[
        "Model vs human",
        "API vs interface"
    ],
    loc="upper center",
    bbox_to_anchor=(0.5, 1.005),
    ncol=2,
    frameon=False,
    prop={
        "weight": "bold",
        "size": 25
    },
    columnspacing=2.5,
    handlelength=2.5,
    handleheight=1.5
)


# ==========================================
# 14. GRID SIZE
# ==========================================

n_models = len(
    plot_data["model"].unique()
)

n_cols = 2

n_rows = math.ceil(
    n_models / n_cols
)


# ==========================================
# 15. CENTER LAST GRAPH
# ==========================================

if n_models % n_cols == 1:

    axes = g.axes.flatten()

    last_ax = axes[n_models - 1]

    pos = last_ax.get_position()

    new_x0 = (
        1.0 - pos.width
    ) / 2

    last_ax.set_position([
        new_x0,
        pos.y0,
        pos.width,
        pos.height
    ])

    # скрываем пустой subplot, если он есть
    for j in range(1, n_cols):

        idx = n_models - 1 + j

        if idx < len(axes):
            axes[idx].axis("off")


# ==========================================
# 16. SPACING + SIZE
# ==========================================

g.fig.subplots_adjust(
    top=0.93,
    hspace=0.20,
    wspace=0.15,
    bottom=0.12
)

g.fig.set_size_inches(
    30,
    n_rows * 11
)


# ==========================================
# 17. SAVE
# ==========================================

plt.savefig(
    "model_vs_human_vs_interface_scale.pdf",
    bbox_inches="tight",
    dpi=300,
    facecolor="white",
    pad_inches=0.5
)

plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. LOAD DATA
# ==========================================

df = pd.read_csv("hackaton_dataset.csv")


# ==========================================
# 2. NUMERIC COLUMNS
# ==========================================

df["model_score"] = pd.to_numeric(
    df["model_score"],
    errors="coerce"
)

df["human_score"] = pd.to_numeric(
    df["human_score"],
    errors="coerce"
)


# ==========================================
# 3. SYSTEMATIC BIAS
# ==========================================

df["diff"] = (
    df["model_score"]
    - df["human_score"]
)


# ==========================================
# 4. MEAN BIAS BY SCALE
# ==========================================

scale_diff = (
    df
    .groupby("scale")["diff"]
    .mean()
    .sort_values()
)


# ==========================================
# 5. COLORS
# ==========================================

cmap = sns.diverging_palette(
    240,
    15,
    s=75,
    l=65,
    as_cmap=True
)

negative_color = cmap(0.15)
positive_color = cmap(0.85)

colors = [
    positive_color if x > 0
    else negative_color
    for x in scale_diff.values
]


# ==========================================
# 6. FIGURE
# ==========================================

plt.figure(
    figsize=(12, 10)
)

ax = plt.gca()

ax.set_facecolor("white")


# ==========================================
# 7. BARPLOT
# ==========================================

plt.barh(
    scale_diff.index,
    scale_diff.values,
    color=colors,
    alpha=0.9
)


# ==========================================
# 8. ZERO LINE
# ==========================================

plt.axvline(
    0,
    color="black",
    linestyle="--",
    linewidth=0.8,
    alpha=0.5
)


# ==========================================
# 9. LABELS
# ==========================================

plt.xlabel(
    "Systematic bias",
    fontsize=11
)

plt.ylabel(
    "Scale",
    fontsize=11
)


# ==========================================
# 10. TICKS
# ==========================================

ax.tick_params(
    axis="both",
    labelsize=10
)


# ==========================================
# 11. GRID
# ==========================================

ax.grid(
    axis="x",
    alpha=0.35,
    linestyle=":",
    linewidth=1.2
)

ax.grid(
    axis="y",
    visible=False
)


# ==========================================
# 12. SPINES
# ==========================================

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)


# ==========================================
# 13. SAVE
# ==========================================

plt.tight_layout()

plt.savefig(
    "scale_diff_bias_human.pdf",
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
    pad_inches=0.3
)

plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# LOAD DATA
# ----------------------------
df = pd.read_csv("api_interface.csv")

# Если файл уже загружен в Colab под другим именем,
# укажи его точное название здесь.

# ----------------------------
# CONVERT SCORES TO NUMERIC
# ----------------------------
df["api_mean"] = pd.to_numeric(df["api_mean"], errors="coerce")
df["interface_mean"] = pd.to_numeric(
    df["interface_mean"], errors="coerce"
)

# Оставляем только строки, где доступны обе оценки
df_valid = df.dropna(
    subset=["api_mean", "interface_mean", "scale"]
).copy()

print("Всего строк:", len(df))
print("Строк с обеими оценками:", len(df_valid))
print("Исключено строк:", len(df) - len(df_valid))

# ----------------------------
# CALCULATE DIFFERENCE
# API - INTERFACE
# ----------------------------
df_valid["diff_api_interface"] = (
    df_valid["api_mean"] - df_valid["interface_mean"]
)

# ----------------------------
# MEAN DIFFERENCE BY SCALE
# ----------------------------
scale_diff = (
    df_valid.groupby("scale")["diff_api_interface"]
    .mean()
    .sort_values()
)

# ----------------------------
# DIVERGING PALETTE
# ----------------------------
cmap = sns.diverging_palette(
    240, 15,
    s=75,
    l=65,
    as_cmap=True
)

negative_color = cmap(0.15)  # Синий
positive_color = cmap(0.85)  # Розовый

colors = [
    positive_color if x > 0 else negative_color
    for x in scale_diff.values
]

# ----------------------------
# FIGURE
# ----------------------------
fig, ax = plt.subplots(figsize=(12, 10))
ax.set_facecolor("white")

ax.barh(
    scale_diff.index,
    scale_diff.values,
    color=colors,
    alpha=0.9
)

# Нулевая линия
ax.axvline(
    0,
    color="black",
    linestyle="--",
    linewidth=0.8,
    alpha=0.6
)

# Подписи
ax.set_xlabel(
    "Mean difference (API − Interface)",
    fontsize=11
)
ax.set_ylabel("Scale", fontsize=11)

ax.tick_params(axis="both", labelsize=10)

# Сетка
ax.grid(
    axis="x",
    alpha=0.35,
    linestyle=":",
    linewidth=1.2
)
ax.grid(axis="y", visible=False)
ax.set_axisbelow(True)

# ----------------------------
# SAVE
# ----------------------------
plt.tight_layout()

plt.savefig(
    "scale_diff_bias_model.pdf",
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
    pad_inches=0.3
)

plt.savefig(
    "scale_diff_api_interface.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
    pad_inches=0.3
)

plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# ----------------------------
# DATA
# ----------------------------

df["abs_error_human"] = (
    df["model_score"] - df["human_score"]
).abs()

test_error = (
    df.groupby("scale")["abs_error_human"]
    .mean()
    .sort_values()
)


# ----------------------------
# CONTINUOUS COLOR GRADIENT
# ----------------------------

gradient_colors = [
    "#7BC67B",
    "#F2D16B",
    "#F4A259",
    "#E76F51"
]

cmap = LinearSegmentedColormap.from_list(
    "error_gradient",
    gradient_colors,
    N=256
)

max_val = test_error.max()
min_val = test_error.min()


def get_gradient_color(val):
    norm_val = (
        (val - min_val) / (max_val - min_val)
        if max_val != min_val else 0
    )
    return cmap(norm_val)


# ----------------------------
# FIGURE
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
    x=test_error.values,
    y=test_error.index,
    color="#B0E0E6",
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
# AXES LABELS
# ----------------------------

plt.xlabel(
    "MAE",
    fontsize=11,
    fontweight="normal"
)

plt.ylabel(
    "Scale",
    fontsize=11,
    fontweight="normal"
)


# ----------------------------
# TICKS
# ----------------------------

ax.tick_params(
    axis="both",
    labelsize=10
)

for label in ax.get_xticklabels():
    label.set_fontweight("normal")

for label in ax.get_yticklabels():
    label.set_fontweight("normal")


# ----------------------------
# GRID
# ----------------------------

ax.grid(
    axis="x",
    alpha=0.15,
    linestyle=":",
    zorder=0
)


# ----------------------------
# LIMITS
# ----------------------------

ax.set_xlim(
    0,
    max_val * 1.08
)


# ----------------------------
# SAVE
# ----------------------------

plt.tight_layout()

plt.savefig(
    "scale_error_gradient_human.pdf",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()

# ----------------------------
# DATA
# ----------------------------

df["abs_error_model"] = (
    df["model_score"] - df["human_score"]
).abs()

test_error = (
    df.groupby("scale")["abs_error_model"]
    .mean()
    .sort_values()
)


# ----------------------------
# CONTINUOUS COLOR GRADIENT
# ----------------------------

gradient_colors = [
    "#7BC67B",
    "#F2D16B",
    "#F4A259",
    "#E76F51"
]

cmap = LinearSegmentedColormap.from_list(
    "error_gradient",
    gradient_colors,
    N=256
)

max_val = test_error.max()
min_val = test_error.min()


def get_gradient_color(val):
    norm_val = (
        (val - min_val) / (max_val - min_val)
        if max_val != min_val else 0
    )
    return cmap(norm_val)


# ----------------------------
# FIGURE
# ----------------------------

plt.figure(figsize=(12, 6))
ax = plt.gca()

ax.set_facecolor("white")


# ----------------------------
# BARPLOT
# ----------------------------

ax = sns.barplot(
    x=test_error.values,
    y=test_error.index,
    color="#B0E0E6",
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
# AXES LABELS
# ----------------------------

plt.xlabel(
    "MAE",
    fontsize=11,
    fontweight="normal"
)

plt.ylabel(
    "Scale",
    fontsize=11,
    fontweight="normal"
)


# ----------------------------
# TICKS
# ----------------------------

ax.tick_params(
    axis="both",
    labelsize=10
)

for label in ax.get_xticklabels():
    label.set_fontweight("normal")

for label in ax.get_yticklabels():
    label.set_fontweight("normal")


# ----------------------------
# GRID
# ----------------------------

ax.grid(
    axis="x",
    alpha=0.15,
    linestyle=":",
    zorder=0
)


# ----------------------------
# LIMITS
# ----------------------------

ax.set_xlim(
    0,
    max_val * 1.08
)


# ----------------------------
# SAVE
# ----------------------------

plt.tight_layout()

plt.savefig(
    "scale_error_gradient_model.pdf",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ==========================================
# 1. ТОЛЬКО EPI
# ==========================================

df_epi = df[
    (df["test"] == "EPI") &
    (df["scale"].isin(["extraversion", "neuroticism"]))
].copy()


# ==========================================
# 2. Среднее по человеку + модели + шкале
# ==========================================

df_agg = (
    df_epi
    .groupby(
        ["person_id", "model", "scale"],
        as_index=False
    )
    .agg({
        "human_score": "mean",
        "model_score": "mean"
    })
)


# ==========================================
# 3. HUMAN
# ==========================================

human_wide = (
    df_agg[
        ["person_id", "model", "scale", "human_score"]
    ]
    .pivot_table(
        index=["person_id", "model"],
        columns="scale",
        values="human_score"
    )
    .reset_index()
)

human_wide["source"] = "Human"


# ==========================================
# 4. MODEL
# ==========================================

model_wide = (
    df_agg[
        ["person_id", "model", "scale", "model_score"]
    ]
    .pivot_table(
        index=["person_id", "model"],
        columns="scale",
        values="model_score"
    )
    .reset_index()
)

model_wide["source"] = "Model"


# ==========================================
# 5. Объединяем
# ==========================================

plot_df = pd.concat(
    [human_wide, model_wide],
    ignore_index=True
)

plot_df = plot_df.dropna(
    subset=["extraversion", "neuroticism"]
)


# ==========================================
# 6. Цвет = человек
# ==========================================

people = plot_df["person_id"].unique()

palette = sns.color_palette(
    "tab10",
    n_colors=len(people)
)

person_colors = dict(
    zip(people, palette)
)


# ==========================================
# 7. График
# ==========================================

sns.set_theme(style="white")

fig, ax = plt.subplots(
    figsize=(10, 10)
)


# ==========================================
# 8. Фон квадрантов
# ==========================================

MIDPOINT = 12

ax.axvspan(
    0, MIDPOINT,
    ymin=0, ymax=0.5,
    facecolor="#cce5ff",
    alpha=0.35
)

ax.axvspan(
    MIDPOINT, 24,
    ymin=0, ymax=0.5,
    facecolor="#7BC67B",
    alpha=0.35
)

ax.axvspan(
    0, MIDPOINT,
    ymin=0.5, ymax=1,
    facecolor="#F2D16B",
    alpha=0.35
)

ax.axvspan(
    MIDPOINT, 24,
    ymin=0.5, ymax=1,
    facecolor="#E76F51",
    alpha=0.35
)


# ==========================================
# 9. Названия темпераментов
# ==========================================

ax.text(
    6, 1,
    "Phlegmatic",
    ha="center",
    va="bottom",
    fontsize=11,
    style="italic",
    color="#555555",
    alpha=0.7
)

ax.text(
    18, 1,
    "Sanguine",
    ha="center",
    va="bottom",
    fontsize=11,
    style="italic",
    color="#555555",
    alpha=0.7
)

ax.text(
    6, 13,
    "Melancholic",
    ha="center",
    va="bottom",
    fontsize=11,
    style="italic",
    color="#555555",
    alpha=0.7
)

ax.text(
    18, 13,
    "Choleric",
    ha="center",
    va="bottom",
    fontsize=11,
    style="italic",
    color="#555555",
    alpha=0.7
)


# ==========================================
# 10. Рисуем пары
# ==========================================

for person_id in people:

    person_data = plot_df[
        plot_df["person_id"] == person_id
    ]

    color = person_colors[person_id]

    # Human
    human = person_data[
        person_data["source"] == "Human"
    ]

    ax.scatter(
        human["extraversion"],
        human["neuroticism"],
        color=color,
        marker="o",
        s=160,
        edgecolor="black",
        linewidth=0.8,
        label=f"{person_id} — Human"
    )

    # Model
    model = person_data[
        person_data["source"] == "Model"
    ]

    ax.scatter(
        model["extraversion"],
        model["neuroticism"],
        color=color,
        marker="X",
        s=160,
        edgecolor="black",
        linewidth=0.8,
        label=f"{person_id} — Model"
    )


# ==========================================
# 11. Оси
# ==========================================

ax.set_xlim(0, 24)
ax.set_ylim(0, 24)

ax.set_xlabel(
    "Extraversion",
    fontsize=12
)

ax.set_ylabel(
    "Neuroticism",
    fontsize=12
)


# ==========================================
# 12. Линии квадрантов
# ==========================================

ax.axhline(
    MIDPOINT,
    color="gray",
    linestyle="--",
    linewidth=0.8,
    alpha=0.5
)

ax.axvline(
    MIDPOINT,
    color="gray",
    linestyle="--",
    linewidth=0.8,
    alpha=0.5
)


# ==========================================
# 13. Оформление
# ==========================================

ax.tick_params(
    axis="both",
    labelsize=10
)

ax.grid(
    True,
    linestyle=":",
    alpha=0.3
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)


# ==========================================
# 14. Легенда
# ==========================================

ax.legend(
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    frameon=True,
    fontsize=9
)


# ==========================================
# 15. Сохранение
# ==========================================

plt.tight_layout()

plt.savefig(
    "temperament_comparison.pdf",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()

from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# ABSOLUTE ERROR
# ----------------------------
df["abs_error_human"] = (
    df["model_score"] - df["human_score"]
).abs()


# ----------------------------
# DATA
# ----------------------------
test_error = (
    df.groupby("test")["abs_error_human"]
    .mean()
    .sort_values()
)


# ----------------------------
# CONTINUOUS COLOR GRADIENT
# ----------------------------
gradient_colors = [
    "#7BC67B",
    "#F2D16B",
    "#F4A259",
    "#E76F51"
]

cmap = LinearSegmentedColormap.from_list(
    "error_gradient",
    gradient_colors,
    N=256
)

max_val = test_error.max()
min_val = test_error.min()


def get_gradient_color(val):
    norm_val = (
        (val - min_val) / (max_val - min_val)
        if max_val != min_val else 0
    )
    return cmap(norm_val)


# ----------------------------
# FIGURE
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
    x=test_error.values,
    y=test_error.index,
    color="#B0E0E6",
    zorder=3
)


# ----------------------------
# COLOR BARS WITH GRADIENT
# ----------------------------
for patch in ax.patches:
    val = patch.get_width()

    patch.set_facecolor(
        get_gradient_color(val)
    )

    patch.set_alpha(0.85)
    patch.set_linewidth(1.0)


# ----------------------------
# AXES LABELS
# ----------------------------
plt.xlabel(
    "MAE",
    fontsize=11,
    fontweight="normal"
)

plt.ylabel(
    "Test",
    fontsize=11,
    fontweight="normal"
)


# ----------------------------
# TICKS
# ----------------------------
ax.tick_params(
    axis="both",
    labelsize=10
)

for label in ax.get_xticklabels():
    label.set_fontweight("normal")

for label in ax.get_yticklabels():
    label.set_fontweight("normal")


# ----------------------------
# GRID
# ----------------------------
ax.grid(
    axis="x",
    alpha=0.15,
    linestyle=":",
    zorder=0
)


# ----------------------------
# LIMITS
# ----------------------------
ax.set_xlim(
    0,
    max_val * 1.08
)


# ----------------------------
# SAVE
# ----------------------------
plt.tight_layout()

plt.savefig(
    "test_error_gradient_human.pdf",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()

# ----------------------------
# DATA
# ----------------------------
test_error = (
    df.groupby("test")["abs_error_model"]
    .mean()
    .sort_values()
)

# ----------------------------
# CONTINUOUS COLOR GRADIENT
# ----------------------------
gradient_colors = ["#7BC67B", "#F2D16B", "#F4A259", "#E76F51"]

cmap = LinearSegmentedColormap.from_list(
    "error_gradient",
    gradient_colors,
    N=256
)

max_val = test_error.max()
min_val = test_error.min()

def get_gradient_color(val):
    norm_val = (
        (val - min_val) / (max_val - min_val)
        if max_val != min_val else 0
    )
    return cmap(norm_val)

# ----------------------------
# FIGURE
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
    x=test_error.values,
    y=test_error.index,
    color='#B0E0E6',
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
# AXES LABELS
# ----------------------------
plt.xlabel(
    "MAE",
    fontsize=11,
    fontweight='normal'
)

plt.ylabel(
    "Test",
    fontsize=11,
    fontweight='normal'
)

# ----------------------------
# TICKS
# ----------------------------
ax.tick_params(axis='both', labelsize=10)

for label in ax.get_xticklabels():
    label.set_fontweight('normal')

for label in ax.get_yticklabels():
    label.set_fontweight('normal')

# ----------------------------
# GRID
# ----------------------------
ax.grid(
    axis='x',
    alpha=0.15,
    linestyle=':',
    zorder=0
)

# ----------------------------
# LIMITS
# ----------------------------
ax.set_xlim(0, max_val * 1.08)

# ----------------------------
# SAVE
# ----------------------------
plt.tight_layout()

plt.savefig(
    'test_error_gradient_model.pdf',
    dpi=300,
    bbox_inches='tight',
    facecolor='white'
)

plt.show()

"""# ***Продолжение***

SYSTEMATIC BIAS

То есть: модель в среднем завышает или занижает оценки?
bias > 0
модель завышает показатели.

bias < 0
модель занижает.

bias ≈ 0
систематического смещения нет.
"""

df["diff"] = df["model_score"] - df["human_score"]

overall_bias = df["diff"].mean()

print("Overall bias:", overall_bias)

scale_bias = (
    df.groupby("scale")["diff"]
    .mean()
    .sort_values(ascending=False)
)

print(scale_bias)

"""Насколько модели ошибаются В СРЕДНЕМ."""

df["abs_error"] = abs(
    df["model_score"] - df["human_score"]
)

mae = df["abs_error"].mean()

print("MAE:", mae)

mae_by_scale = (
    df.groupby("scale")["abs_error"]
    .mean()
    .sort_values(ascending=False)
)

print(mae_by_scale)

import pandas as pd

# Загружаем исходный файл на 600 строк
df = pd.read_csv("hackaton_dataset.csv")

# Проверяем количество строк
print("Всего строк:", len(df))

# Приводим оценки к числовому формату
df["model_score"] = pd.to_numeric(df["model_score"], errors="coerce")
df["human_score"] = pd.to_numeric(df["human_score"], errors="coerce")

# Убираем строки, где нет одной из двух оценок
df_check = df.dropna(subset=["model_score", "human_score"]).copy()

# Разница: модель - человек
df_check["difference"] = (
    df_check["model_score"] - df_check["human_score"]
)

# Категории
df_check["comparison"] = "model_lower"

df_check.loc[
    df_check["difference"] > 0,
    "comparison"
] = "model_higher"

df_check.loc[
    df_check["difference"] == 0,
    "comparison"
] = "exact_match"

# Считаем количество
counts = df_check["comparison"].value_counts()

# Считаем проценты
percentages = (
    df_check["comparison"]
    .value_counts(normalize=True)
    * 100
)

print("\nКоличество:")
print(counts)

print("\nПроценты:")
print(percentages.round(2))

print("\nИтого наблюдений для сравнения:", len(df_check))

"""# ***Likert***"""

from scipy.stats import spearmanr

# Загружаем данные
df_Likert = pd.read_csv("/content/Likert.csv")

# Оставляем только строки, где есть оба показателя
df_Likert_clean = df_Likert[
    df_Likert["interface_score"].notna()
    & df_Likert["Likert_human_score"].notna()
].copy()

# Убеждаемся, что значения числовые
df_Likert_clean["interface_score"] = pd.to_numeric(
    df_Likert["interface_score"], errors="coerce"
)
df_Likert_clean["Likert_human_score"] = pd.to_numeric(
    df_Likert["Likert_human_score"], errors="coerce"
)


df_Likert_clean

"""Spearman сейчас проверяет:
В целом по всем доступным шкалам конкретного человека: когда interface_score выше, становится ли выше и Likert_human_score?

Когда X увеличивается, Y в целом увеличивается или уменьшается?

Он считает одно число — ρ (rho) от −1 до +1.

Например:

ρ = +0.8 → сильная положительная связь
ρ = 0 → связи практически нет
ρ = −0.8 → сильная отрицательная связь
"""

# Результаты
results = []

for person in sorted(df_Likert_clean["Person"].unique()):

    person_data = df_Likert_clean[df_Likert_clean["Person"] == person].copy()

    x = person_data["interface_score"]
    y = person_data["Likert_human_score"]

    # Для корреляции нужно минимум 3 наблюдения
    if len(person_data) >= 3 and x.nunique() > 1 and y.nunique() > 1:

        rho, p_value = spearmanr(x, y)

        results.append({
            "Person": person,
            "Model": person_data["Model_name"].iloc[0],
            "N": len(person_data),
            "Spearman_rho": rho,
            "p_value": p_value
        })

    else:

        results.append({
            "Person": person,
            "Model": person_data["Model_name"].iloc[0],
            "N": len(person_data),
            "Spearman_rho": None,
            "p_value": None
        })

# Создаём таблицу результатов
results_df = pd.DataFrame(results)

# Сортируем по человеку
results_df = results_df.sort_values("Person")

print("=" * 80)
print("SPEARMAN CORRELATION ДЛЯ КАЖДОГО ЧЕЛОВЕКА")
print("=" * 80)

display(results_df)

df_Likert_clean

df

"""**ordinal logistic regression**

Она использует дополнительную информацию о том, что человеческая оценка: 1 < 2 < 3 < 4 < 5

То есть это не просто произвольные числа, а порядковые категории.

Модель спрашивает: Когда interface_score увеличивается, увеличивается ли вероятность того, что человек попадёт в более высокую категорию Likert?
"""

from statsmodels.miscmodels.ordinal_model import OrderedModel
import statsmodels.api as sm


# Стандартизация interface_score внутри каждой психологической шкалы (z-стандартизация внутри каждой шкалы)
df_Likert_clean["interface_z"] = df_Likert_clean.groupby("scale")["interface_score"].transform(
    lambda x: (x - x.mean()) / x.std()
)

# Убираем строки, где стандартизация невозможна
df_Likert_clean = df_Likert_clean.dropna(subset=["interface_z"])

# Likert как порядковая переменная 1–5
df_Likert_clean["Likert_human_score"] = df_Likert_clean["Likert_human_score"].astype(int)

# Dummy-переменные для Person
person_dummies = pd.get_dummies(
    df_Likert_clean["Person"],
    prefix="Person",
    drop_first=True,
    dtype=float
)

X = pd.concat(
    [df_Likert_clean[["interface_z"]], person_dummies],
    axis=1
)

y = df_Likert_clean["Likert_human_score"]

# Ordinal logistic regression
model = OrderedModel(
    y,
    X,
    distr="logit"
)

result = model.fit(
    method="bfgs",
    disp=False
)

print(result.summary())

# Главный результат
print("\n" + "=" * 60)
print("ГЛАВНЫЙ РЕЗУЛЬТАТ")
print("=" * 60)

print(f"Коэффициент interface_score: {result.params['interface_z']:.4f}")
print(f"p-value: {result.pvalues['interface_z']:.4f}")

"""Есть ли в целом по всем 134 наблюдениям связь между модельным показателем и человеческой оценкой соответствующей характеристики?"""

from scipy.stats import spearmanr

# Оставляем только строки, где есть оба значения
df_Likert_clean = df_Likert_clean[["interface_score", "Likert_human_score"]].dropna()

# Корреляция Спирмена
rho, p_value = spearmanr(
    df_Likert_clean["interface_score"],
    df_Likert_clean["Likert_human_score"]
)

print("=" * 60)
print("ОБЩАЯ КОРРЕЛЯЦИЯ ПО ВСЕМ 134 НАБЛЮДЕНИЯМ")
print("=" * 60)

print(f"N = {len(df_Likert_clean)}")
print(f"Spearman rho = {rho:.3f}")
print(f"p-value = {p_value:.4f}")

if p_value < 0.05:
    print("Результат статистически значим (p < 0.05).")
else:
    print("Результат статистически НЕ значим (p >= 0.05).")
