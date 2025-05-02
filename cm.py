from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 2) Функция для отрисовки и сохранения теплокарты
#    если true_classes=None, то скрываем метки по оси Y

def plot_cm_df(cm_df: pd.DataFrame,
               true_classes: list,
               pred_classes: list,
               title: str,
               img_path: Path,
               show: bool = False,
               print_cm: bool = False):
    true_classes_len = 1 if true_classes is None else len(true_classes)
    pred_classes_len = 1 if pred_classes is None else len(pred_classes)

    fig, ax = plt.subplots(figsize=(1 + pred_classes_len, 1.5 + true_classes_len))
    im = ax.imshow(cm_df.values, cmap="Blues", aspect="auto")
    ax.set_xticks(np.arange(len(pred_classes)))
    ax.set_xticklabels(pred_classes, rotation=45, ha="right")
    if true_classes is not None:
        ax.set_yticks(np.arange(len(true_classes)))
        ax.set_yticklabels(true_classes)
        ax.set_ylabel("Истинный класс")
    else:
        ax.set_yticks([])
        ax.set_ylabel("")
    ax.set_xlabel("Предсказанный класс")
    ax.set_title(title)
    # цифры в ячейках
    for i in range(cm_df.shape[0]):
        for j in range(cm_df.shape[1]):
            ax.text(j, i, f"{cm_df.iat[i, j]:.2f}", ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    img_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(img_path)
    if print_cm:
        print(title)
        print(cm_df)
    if show:
        plt.show()
    plt.close()


def create_cms(csv_fp,
               out_dir,
               exp_class_dict,
               true_classes,
               pred_classes):
    df = pd.read_csv(csv_fp)

    # приводим к категориальному типу, чтобы были все классы
    if true_classes is not None:
        y_true = pd.Categorical(df["PGr"].astype(int),      categories=true_classes)
    else:
        # объединяем все истинные классы в один
        y_true = pd.Categorical([0] * len(df), categories=[0])
    y_pred = pd.Categorical(df["Pred_PGr"].astype(str), categories=pred_classes)

    # 4) Считаем общую матрицу
    cm = pd.crosstab(y_true, y_pred, dropna=False)

    # 5) Делаем три нормировки
    cm_all = cm / cm.values.sum()                       # глобально
    cm_col = cm.div(cm.sum(axis=0), axis=1).fillna(0)    # по столбцам (precision)
    cm_row = cm.div(cm.sum(axis=1), axis=0).fillna(0)    # по строкам (recall)

    # 6) Сохраняем общие CM
    # plot_cm_df(cm,     f"Матрица ошибок (raw)",                             out_dir / "cm_raw.svg")
    # plot_cm_df(cm_all, f"Матрица ошибок (глобальная нормировка)",           out_dir / "cm_global.svg")
    # plot_cm_df(cm_col, f"Матрица ошибок (нормировка по столбцам, precision)", out_dir / "cm_precision.svg")
    plot_cm_df(cm_row,
               true_classes,
               pred_classes,
               f"Матрица ошибок (нормировка по строкам, recall)",
               out_dir / "cm_recall.svg",
               print_cm=True)

    # 7) Группируем по Exp_dir и сохраняем CM для каждого эксперимента
    for exp_name, df_exp in df.groupby("Exp_dir"):
        if exp_class_dict is not None:
            exp_name = f'{exp_class_dict[exp_name]}'
        safe_name = str(exp_name).replace("/", "_").replace(" ", "_")
        # приводим к категориальному типу снова для подтаблицы
        if true_classes is not None:
            y_true_e = pd.Categorical(df_exp["PGr"].astype(int),      categories=true_classes)
        else:
            y_true_e = pd.Categorical([0] * len(df_exp), categories=[0])
        y_pred_e = pd.Categorical(df_exp["Pred_PGr"].astype(str), categories=pred_classes)
        cm_e = pd.crosstab(y_true_e, y_pred_e, dropna=False)
        cm_all_e = cm_e / cm_e.values.sum()
        cm_col_e = cm_e.div(cm_e.sum(axis=0), axis=1).fillna(0)
        cm_row_e = cm_e.div(cm_e.sum(axis=1), axis=0).fillna(0)

        # plot_cm_df(cm_e,     true_classes, pred_classes, f"[{exp_name}] Матрица ошибок (raw)",                             out_dir / f"{safe_name}_cm_raw.svg")
        # plot_cm_df(cm_all_e, true_classes, pred_classes, f"[{exp_name}] Матрица ошибок (глобальная нормировка)",           out_dir / f"{safe_name}_cm_global.svg")
        # plot_cm_df(cm_col_e, true_classes, pred_classes, f"[{exp_name}] Матрица ошибок (нормировка по столбцам, precision)", out_dir / f"{safe_name}_cm_precision.svg")
        plot_cm_df(cm_row_e,
                   true_classes,
                   pred_classes,
                   f"[{exp_name}] Матрица ошибок (нормировка по строкам, recall)",
                   out_dir / f"{safe_name}_cm_recall.svg",
                   print_cm=True)


def create_cms_prob(csv_fp: Path,
                    out_dir: Path,
                    exp_class_dict: dict = None,
                    true_classes: list = None,
                    pred_classes: list = None,
                    thresh_low: float = 0.4,
                    thresh_margin: float = 0.1,
                    thresh_tri: float = 0.05):
    """
    Аналогично create_cms, но сначала классифицирует строки по вероятностям PGr1_prob, PGr2_prob, PGr3_prob
    на основании заданных порогов, записывая результат в столбец 'Pred_PGr'.
    """
    # Читаем исходные данные
    df = pd.read_csv(csv_fp)

    # Генерируем метки Pred_PGr на основании порогов
    pred_labels = []
    for _, row in df.iterrows():
        p_vec = np.array([row['PGr1_prob'], row['PGr2_prob'], row['PGr3_prob']])
        max_v = p_vec.max()
        max_i = p_vec.argmax()
        span = p_vec.max() - p_vec.min()
        sorted_desc = np.argsort(p_vec)[::-1]
        margin = p_vec[sorted_desc[0]] - p_vec[sorted_desc[1]]

        # Determine prediction label based on thresholds 
        if max_v < thresh_low or span < thresh_tri:
            pred_label = 'Unknown'
        elif margin < thresh_margin:
            # ambiguous top-2 classes
            top2 = sorted_desc[:2]
            combo = sorted([int(top2[0]) + 1, int(top2[1]) + 1])
            pred_label = f"{combo[0]}/{combo[1]}"
            if pred_label == "1/3":
                pred_label = "3/1"
        else:
            pred_label = f"{max_i + 1}"
        pred_labels.append(pred_label)

    # Добавляем новый столбец с предсказаниями
    df['Pred_PGr'] = pred_labels

    # Убедимся, что директория для выходных файлов существует
    out_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_dir / Path(str(csv_fp.stem) + '_upd_classes.csv'))
    # Общая матрица (recall-нормировка по строкам)
    if true_classes is not None:
        y_true = pd.Categorical(df['PGr'].astype(int), categories=true_classes)
    else:
        y_true = pd.Categorical([0] * len(df), categories=[0])
    y_pred = pd.Categorical(df['Pred_PGr'].astype(str), categories=pred_classes)
    cm = pd.crosstab(y_true, y_pred, dropna=False)

    # 5) Делаем три нормировки
    cm_all = cm / cm.values.sum()                       # глобально
    cm_col = cm.div(cm.sum(axis=0), axis=1).fillna(0)    # по столбцам (precision)
    cm_row = cm.div(cm.sum(axis=1), axis=0).fillna(0)    # по строкам (recall)

    # 6) Сохраняем общие CM
    # plot_cm_df(cm,     f"Матрица ошибок (raw)",                             out_dir / "cm_raw.svg")
    # plot_cm_df(cm_all, f"Матрица ошибок (глобальная нормировка)",           out_dir / "cm_global.svg")
    # plot_cm_df(cm_col, f"Матрица ошибок (нормировка по столбцам, precision)", out_dir / "cm_precision.svg")
    plot_cm_df(cm_row,
               true_classes,
               pred_classes,
               f"Матрица ошибок (нормировка по строкам, recall)",
               out_dir / "cm_recall.svg",
               print_cm=True)

    # Матрицы по каждому эксперименту (Exp_dir)
    for exp_name, df_exp in df.groupby('Exp_dir'):
        key = exp_class_dict.get(exp_name, exp_name) if exp_class_dict else exp_name
        safe_name = str(key).replace('/', '_').replace(' ', '_')

        if true_classes is not None:
            y_true_e = pd.Categorical(df_exp['PGr'].astype(int), categories=true_classes)
        else:
            y_true_e = pd.Categorical([0] * len(df_exp), categories=[0])
        y_pred_e = pd.Categorical(df_exp['Pred_PGr'].astype(str), categories=pred_classes)
        cm_e = pd.crosstab(y_true_e, y_pred_e, dropna=False)
        cm_all_e = cm_e / cm_e.values.sum()
        cm_col_e = cm_e.div(cm_e.sum(axis=0), axis=1).fillna(0)
        cm_row_e = cm_e.div(cm_e.sum(axis=1), axis=0).fillna(0)

        # plot_cm_df(cm_e,     true_classes, pred_classes, f"[{exp_name}] Матрица ошибок (raw)",                             out_dir / f"{safe_name}_cm_raw.svg")
        # plot_cm_df(cm_all_e, true_classes, pred_classes, f"[{exp_name}] Матрица ошибок (глобальная нормировка)",           out_dir / f"{safe_name}_cm_global.svg")
        # plot_cm_df(cm_col_e, true_classes, pred_classes, f"[{exp_name}] Матрица ошибок (нормировка по столбцам, precision)", out_dir / f"{safe_name}_cm_precision.svg")
        plot_cm_df(cm_row_e,
                   true_classes,
                   pred_classes,
                   f"[{exp_name}] Матрица ошибок (нормировка по строкам, recall)",
                   out_dir / f"[{safe_name}]_cm_recall.svg",
                   print_cm=True)
