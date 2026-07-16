import re

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

def parse_experiments(text, task_ids):
    lines = [l.strip() for l in text.splitlines() if l.strip() and not l.strip().startswith('#')]
    experiments = []
    for task_id in task_ids:
        if task_id - 1 >= len(lines):
            experiments.append(f"Warning: task_id {task_id} exceeds number of lines ({len(lines)}), skipping.")
            continue
        line = lines[task_id - 1]
        match = re.search(r'exp_name\s+(\S+)', line)
        if match:
            experiments.append(match.group(1))
    return experiments

if __name__ == "__main__":
    array_str = "1,8-19%4"
    task_ids  = parse_array_ids(array_str)

    with open("slurm_params.txt", "r") as f:
        text = f.read()

    exp_names = parse_experiments(text, task_ids)
    for name in exp_names:
        print(name)
