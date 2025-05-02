from pathlib import Path
from cm import create_cms, create_cms_prob


if __name__ == '__main__':
    # путь к вашему файлу
    csv_fp = Path("datasets/290425_2025_Cell_staining_NSU_4/result_all_290425_2025_Cell_staining_NSU_4.csv")
    # куда сохранять папку с матрицами
    out_dir = Path("datasets/290425_2025_Cell_staining_NSU_4/cm")
    # у вас нет особых переименований экспериментов
    exp_class_dict = None
    # в файле только два истинных класса
    true_classes = None
    # и три возможных предсказания
    pred_classes = ["1", "1/2", "2", "2/3", "3", "3/1", "Unknown"]  # 7 предсказанных

    # запустим функцию
    create_cms_prob(csv_fp, out_dir, exp_class_dict, true_classes, pred_classes)
