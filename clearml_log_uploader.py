import re
from collections import defaultdict
from pathlib import PosixPath, Path
from pprint import pprint
from tqdm import tqdm

from clearml import Task
from torch.utils.tensorboard import SummaryWriter


def parse_log_file(log_file_path):
    lines = list()
    with open(log_file_path, 'r', newline='\n') as file:
        for line in tqdm(file, desc='read_lines'):
            lines.append(line[:-1].split('\r')[-1])
    # with open(log_file_path, 'r') as file:
    #     lines = file.readlines()

    # Initialize variables
    params = {}
    metrics_train = defaultdict(list)
    metrics_valid = defaultdict(list)
    times_train = []
    times_valid = []

    # Regex patterns for extracting information
    metric_pattern = re.compile(r"(\w+) - ([0-9.e+-]+)")
    time_pattern = re.compile(r"\[[^<]*<")
 
    # Temporary storage for parameter lines
    param_lines = []
    inside_params = False

    # Parsing the file
    for line in tqdm(lines, desc='process lines'):
        # Collect lines for parameters
        if line.startswith("{'"):
            inside_params = True
            param_lines.append(line.strip())
        elif inside_params:
            param_lines.append(line.strip())
            if line.strip().endswith("}"):
                inside_params = False
                # Safely evaluate the collected parameter lines
                param_str = " ".join(param_lines)
                try:
                    params.update(eval(param_str))
                except SyntaxError as e:
                    print(f"Error parsing parameters: {e}")
                param_lines.clear()

        # Extract training metrics
        if line.startswith("train:"):
            metrics = metric_pattern.findall(line)
            for name, value in metrics:
                metrics_train[name].append(float(value))
            
            # # Extract learning rate
            # lr_match = re.search(r"LR - ([0-9.e+-]+)", line)
            # if lr_match:
            #     learning_rates.append(float(lr_match.group(1)))
            
            # Extract training time
            time_match = time_pattern.search(line)
            if time_match:
                t = time_match.group()[1:-1]
                times_train.append(t)

        # Extract validation metrics
        elif line.startswith("valid:"):
            metrics = metric_pattern.findall(line)
            for name, value in metrics:
                metrics_valid[name].append(float(value))
            
            # Extract validation time
            time_match = time_pattern.search(line)
            if time_match:
                t = time_match.group()[1:-1]
                times_valid.append(t)

    return params, dict(metrics_train), times_train, dict(metrics_valid), times_valid


# Usage example
log_file_path = Path("out_CASCADE/cellseg.45339.log")
run_clear_ml = True
project_name = "CellSeg4"
log_dir = f'{str(log_file_path.stem)}_from_logs'

params, metrics_train, times_train, metrics_valid, times_valid = parse_log_file(log_file_path)

params['dataset_dir'] = str(params['dataset_dir'])
learning_rates = metrics_train.pop('LR')

# # Output for testing
# print("Parameters:")
# pprint(params)
# print("Training Metrics:")
# pprint(metrics_train)
# print("Learning Rates:")
# pprint(learning_rates)
# print("Training Times:")
# pprint(times_train)
# print("Validation Metrics:")
# pprint(metrics_valid)
# print("Validation Times:")
# pprint(times_valid)

num_epochs = len(metrics_train.get(list(metrics_train.keys())[0]))

epochs_list = list()
for epoch_idx in range(num_epochs):
    epoch_dict = dict()
    for k in metrics_train.keys():
        epoch_dict[f'{k}/train'] = metrics_train[k][epoch_idx]
    for k in metrics_valid.keys():
        epoch_dict[f'{k}/val'] = metrics_valid[k][epoch_idx]
    epoch_dict['LR'] = learning_rates[epoch_idx]
    epochs_list.append(epoch_dict)

epoch_dict = dict()
for k in metrics_valid.keys():
    epoch_dict[f'{k}/test'] = metrics_valid[k][num_epochs]
epochs_list[0].update(epoch_dict)

if run_clear_ml:
    task = Task.init(project_name=project_name,
                    task_name=log_dir,
                    output_uri=False)
    task.connect(params)
else:
    task = None
writer = SummaryWriter(log_dir=log_dir)

for epoch_idx, epoch in enumerate(epochs_list):
    writer.add_scalar('LR', epoch.get('LR'), epoch_idx)
    for k, v in epoch.items():
        writer.add_scalar(k, v, epoch_idx)

writer.close()
if run_clear_ml:
    task.close()