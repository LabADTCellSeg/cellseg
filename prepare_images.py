from PIL import Image
from pathlib import Path
# import shutil
from tqdm import tqdm

root_dir = Path('datasets/Cells_2.0_for_Ivan/masked_MSC')
dir01 = root_dir / 'pics 2024-20240807T031703Z-001' / 'pics 2024'
dir02 = root_dir / 'pics 2024-20240807T031703Z-002' / 'pics 2024'

# Пути до input и output директорий
input_dir = dir01 / 'LF1'
resize_coef = 4
output_dir = dir01 / f'LF1_{resize_coef}'

# Список расширений файлов, которые мы ищем
ext_list = ['.jpg', '.jpeg', '.png']

# Целевой размер для resize
img_path = list(input_dir.rglob('*m.png'))[0]
with Image.open(img_path) as img:
    new_width = int(img.size[0] / resize_coef) // 32 * 32
    new_height = int(img.size[1] / resize_coef) // 32 * 32
    target_size = (new_width, new_height)

# Создаем output_dir, если она не существует
output_dir.mkdir(parents=True, exist_ok=True)

# Проходим по всем файлам в input_dir и вложенных папках
for img_path in tqdm(input_dir.rglob('*')):
    if img_path.suffix.lower() in ext_list:
        # Путь к выходной директории, сохраняем иерархию
        relative_path = img_path.relative_to(input_dir)
        output_img_path = output_dir / relative_path

        # Создаем директории для output_img_path, если их нет
        output_img_path.parent.mkdir(parents=True, exist_ok=True)

        # Открываем изображение и делаем resize
        with Image.open(img_path) as img:
            if img_path.suffix.lower() == '.png':
                # Используем NEAREST для сохранения точных значений пикселей
                resized_img = img.resize(target_size, Image.NEAREST)
            else:
                # Для остальных изображений используем стандартный метод (например, BICUBIC для плавного ресайза)
                resized_img = img.resize(target_size, Image.BICUBIC)
            # Сохраняем изображение в output_dir
            resized_img.save(output_img_path)

print("Изображения успешно обработаны и сохранены.")