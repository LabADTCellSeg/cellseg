from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1) Жёстко задаём списки классов


# 2) Функция для отрисовки и сохранения теплокарты
def plot_cm_df(cm_df: pd.DataFrame, true_classes: list, pred_classes: list, title: str, img_path: Path, show: bool = False, print_cm: bool = False):
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(cm_df.values, cmap="Blues", aspect="auto")
    ax.set_xticks(np.arange(len(pred_classes)))
    ax.set_xticklabels(pred_classes, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(true_classes)))
    ax.set_yticklabels(true_classes)
    ax.set_xlabel("Предсказанный класс")
    ax.set_ylabel("Истинный класс")
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


def create_cms(csv_fp, out_dir, exp_class_dict, true_classes, pred_classes):
    df = pd.read_csv(csv_fp)

    # приводим к категориальному типу, чтобы были все классы
    y_true = pd.Categorical(df["PGr"].astype(int),      categories=true_classes)
    y_pred = pd.Categorical(df["Pred_PGr"].astype(str), categories=pred_classes)

    # 4) Считаем общую матрицу 3×7
    cm = pd.crosstab(y_true, y_pred, dropna=False)

    # 5) Делаем три нормировки
    cm_all = cm / cm.values.sum()                       # глобально
    cm_col = cm.div(cm.sum(axis=0), axis=1).fillna(0)    # по столбцам (precision)
    cm_row = cm.div(cm.sum(axis=1), axis=0).fillna(0)    # по строкам (recall)

    # 6) Сохраняем общие CM
    # plot_cm_df(cm,     f"Матрица ошибок (raw)",                             out_dir / "[all]_cm_raw.svg")
    # plot_cm_df(cm_all, f"Матрица ошибок (глобальная нормировка)",           out_dir / "[all]_cm_global.svg")
    # plot_cm_df(cm_col, f"Матрица ошибок (нормировка по столбцам, precision)", out_dir / "[all]_cm_precision.svg")
    plot_cm_df(cm_row, true_classes, pred_classes, f"Матрица ошибок (нормировка по строкам, recall)",     out_dir / "[all]_cm_recall.svg", print_cm=True)

    # 7) Группируем по Exp_dir и сохраняем CM для каждого эксперимента
    for exp_name, df_exp in df.groupby("Exp_dir"):
        if exp_class_dict is not None:
            exp_name = f'{exp_class_dict[exp_name]}'
        safe_name = str(exp_name).replace("/", "_").replace(" ", "_")
        # приводим к категориальному типу снова для подтаблицы
        y_true_e = pd.Categorical(df_exp["PGr"].astype(int),      categories=true_classes)
        y_pred_e = pd.Categorical(df_exp["Pred_PGr"].astype(str), categories=pred_classes)
        cm_e = pd.crosstab(y_true_e, y_pred_e, dropna=False)
        cm_all_e = cm_e / cm_e.values.sum()
        cm_col_e = cm_e.div(cm_e.sum(axis=0), axis=1).fillna(0)
        cm_row_e = cm_e.div(cm_e.sum(axis=1), axis=0).fillna(0)

        # plot_cm_df(cm_e,     true_classes, pred_classes, f"[{exp_name}] Матрица ошибок (raw)",                             out_dir / f"[{exp_name}]_cm_raw.svg")
        # plot_cm_df(cm_all_e, true_classes, pred_classes, f"[{exp_name}] Матрица ошибок (глобальная нормировка)",           out_dir / f"[{exp_name}]_cm_global.svg")
        # plot_cm_df(cm_col_e, true_classes, pred_classes, f"[{exp_name}] Матрица ошибок (нормировка по столбцам, precision)", out_dir / f"[{exp_name}]_cm_precision.svg")
        plot_cm_df(cm_row_e, true_classes, pred_classes, f"[{exp_name}] Матрица ошибок (нормировка по строкам, recall)",     out_dir / f"[{exp_name}]_cm_recall.svg", print_cm=True)


if __name__ == '__main__':
    # 3) Обходим все suffix и обрабатываем каждый CSV
    # suffix_list = ['None', '128', '256', '512', '1024']
    suffix_list = ['None', '128', '256', '512', '1024']

    # base_out = Path("out/WJ-MSC-P57")
    # exp_class_dict = {'2024-05-01-wj-MSC-P57p3': 'p3',
    #                     '2024-05-03-wj-MSC-P57p5': 'p5',
    #                     '2024-05-01-wj-MSC-P57p7': 'p7',
    #                     '2024-05-04-wj-MSC-P57p9-sl2': 'p9',
    #                     '2024-05-02-wj-MSC-P57p11': 'p11',
    #                     '2024-05-03-wj-MSC-P57p13': 'p13',
    #                     '2024-05-02-wj-MSC-P57p15sl2': 'p15'}
    # true_classes = [1, 2, 3]  # 3 истинных

    # base_out = Path("out/Microscopy_Ivan")
    # exp_class_dict = {'25 wjMSC R1 (П57) p6 ctrl': 'p6 ctrl',
    #                   '28 wjMSC R1 (П57) p6 H2O2 4h': 'p6 H2O2 4h'}
    # true_classes = [1, 2, 3]  # 3 истинных

    base_out = Path("out/MSC_conf")
    exp_class_dict = None
    true_classes = [1, 2, 3]  # 3 истинных
    pred_classes = ["1", "1/2", "2", "2/3", "3", "3/1", "Unknown"]  # 7 предсказанных

    # base_out = Path("out/MSC_light")
    # exp_class_dict = None
    # true_classes = [1, 2, 3]  # 3 истинных

    stat_dir_mask = "MC_DeepLabV3Plus_timm-efficientnet-b0_20250116_130611_test{suffix}"

    for suffix in suffix_list:
        model_results_dir = base_out / stat_dir_mask.format(suffix=suffix)
        csv_fp = model_results_dir / "stats_results/csv_stat/result_all.csv"
        if not csv_fp.exists():
            continue

        print(f"Processing: {model_results_dir}")
        create_cms(csv_fp, base_out / "cm" / suffix, exp_class_dict, true_classes, pred_classes)

        print()

    print('DONE!')
