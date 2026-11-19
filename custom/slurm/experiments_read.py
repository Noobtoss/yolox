import json
import re
import sys

def parse_array_ids(array_str):
    array_str = array_str.split('#')[0].strip()
    array_str = re.sub(r'%\d+', '', array_str).strip()
    ids = []
    for part in array_str.split(','):
        if '-' in part:
            start, end = part.split('-')
            ids.extend(range(int(start), int(end) + 1))
        else:
            ids.append(int(part))
    return ids

def experiments_read(text, task_ids):
    names = list(json.loads(text))

    experiments = []
    for task_id in task_ids:
        if not 1 <= task_id <= len(names):
            experiments.append(f"Warning: task_id {task_id} not found in experiments, skipping.")
        else:
            experiments.append(names[task_id - 1])
    return experiments

if __name__ == "__main__":
    array_str = sys.argv[1] if len(sys.argv) > 1 else "1-3"
    experiments_file = sys.argv[2] if len(sys.argv) > 2 else "experiments.json"
    task_ids  = parse_array_ids(array_str)

    with open(experiments_file, "r") as f:
        text = f.read()

    run_names = experiments_read(text, task_ids)
    for name in run_names:
        print(name)
