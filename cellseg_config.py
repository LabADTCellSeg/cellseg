import re
import json
from pathlib import Path


def load_jsonc(path: Path) -> dict:
    """Считывает JSONC (с комментариями и trailing commas) и возвращает dict."""
    text = path.read_text(encoding="utf-8")
    # 1) Удалить однострочные комментарии //
    text = re.sub(r"//.*", "", text)
    # 2) Удалить многострочные /* ... */
    text = re.sub(r"/\*[\s\S]*?\*/", "", text)
    # 3) Удалить trailing commas перед закрывающей скобкой
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return json.loads(text)


def get_dataset_config(dataset_cfg_name):
    dataset_cfg_path = Path("dataset_configs") / f"{dataset_cfg_name}.jsonc"
    dataset_cfg = load_jsonc(dataset_cfg_path)
    return dataset_cfg


def get_config(dataset_cfg_name, model_name):
    dataset_cfg = get_dataset_config(dataset_cfg_name)
    model_results_dir_suffix = dataset_cfg["square_a"]
    if dataset_cfg["square_a"] is None:
        model_results_dir_suffix = 'None'

    model_cfg = dict(
        model_dir=f'models/WJ-MSC-P57/{model_name}',
        # model_results_dir=f'out/{dataset_cfg_name}/{model_name}'
        model_results_dir=f'out/{dataset_cfg_name}/{model_name}__sq_{model_results_dir_suffix}'
    )
    return dataset_cfg, model_cfg

