import os
import re
import numpy as np


def _read_openfoam_field(file_path):
    lines = []
    with open(file_path, 'r') as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith('//'):
                continue
            lines.append(line)

    if not lines:
        raise ValueError(f"Empty OpenFOAM file: {file_path}")

    start = 0
    while start < len(lines) and 'internalField' not in lines[start]:
        start += 1
    if start >= len(lines):
        raise ValueError(f"internalField not found in {file_path}")

    line = lines[start]
    if 'uniform' in line:
        values = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', line)
        if len(values) == 0:
            raise ValueError(f"No uniform value parsed from {file_path}")
        if len(values) == 3:
            return np.array([float(v) for v in values], dtype=np.float32)
        return np.array([float(values[0])], dtype=np.float32)

    data_lines = []
    started = False
    for j in range(start, len(lines)):
        if '(' in lines[j] and 'List' not in lines[j] and not started:
            started = True
            if lines[j].strip() != '(':
                data_lines.append(lines[j])
            continue
        if started:
            if lines[j] == ')':
                break
            data_lines.append(lines[j])

    values = []
    vectors = []
    for entry in data_lines:
        if entry.startswith('(') and entry.endswith(')'):
            coords = [float(v) for v in entry[1:-1].split()]
            vectors.append(coords)
        else:
            try:
                values.append(float(entry))
            except ValueError:
                values.extend([float(v) for v in re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', entry)])

    if vectors:
        return np.array(vectors, dtype=np.float32)
    return np.array(values, dtype=np.float32)


def read_openfoam_case(case_dir):
    time_dirs = [d for d in os.listdir(case_dir) if os.path.isdir(os.path.join(case_dir, d)) and _is_time_dir(d)]
    time_dirs.sort(key=float)
    data = {}

    for time_dir in time_dirs:
        time_path = os.path.join(case_dir, time_dir)
        try:
            u_data = _read_openfoam_field(os.path.join(time_path, 'U'))
            p_data = _read_openfoam_field(os.path.join(time_path, 'p'))
            alpha_path = _find_alpha_field(time_path)
            alpha_data = _read_openfoam_field(alpha_path)
            data[float(time_dir)] = {'U': u_data, 'p': p_data, 'alpha': alpha_data}
        except Exception as e:
            print(f"Error reading time {time_dir}: {e}")
            continue

    return data


def _is_time_dir(name):
    try:
        float(name)
        return True
    except ValueError:
        return False


def _find_alpha_field(time_path):
    for name in ['alpha.vapour', 'alpha.water', 'alpha']:
        path = os.path.join(time_path, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"No alpha field found in {time_path}")
