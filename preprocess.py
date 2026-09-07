import os
import re
import numpy as np


def _is_time_dir(name):
    """Return True if the directory name looks like a numeric OpenFOAM time folder."""
    try:
        float(name)
        return True
    except ValueError:
        return False


def _parse_case_params(case_dir):
    """Extract sigma, Re, and AOA from a case folder name.

    Supported patterns:
        CN_<sigma>                       → Re defaults to 7.8e5, AOA = 8°
        CN_<sigma>_Re<value>             → Re from name
        CN_<sigma>_AOA<aoa>_Re<value>    → Re and AOA from name
        CN_<sigma>_Re<value>_AOA<aoa>    → order-insensitive

    Examples of valid names:
        CN_0.28_Re5e5
        CN_0.4_Re7.8e5
        CN_1.2_Re1.2e6
        CN_0.8_AOA8_Re7.8e5
    """
    basename = os.path.basename(case_dir.rstrip(os.sep))

    sigma_match = re.match(r'CN_([0-9]+(?:\.[0-9]+)?)', basename)
    if not sigma_match:
        raise ValueError(f"Cannot parse sigma from folder name '{basename}'")
    sigma = float(sigma_match.group(1))

    re_match = re.search(r'[Rr]e([0-9]+(?:\.[0-9]+)?[eE][+-]?[0-9]+)', basename)
    Re = float(re_match.group(1)) if re_match else 7.8e5

    aoa_match = re.search(r'AOA[=_]?([0-9]+(?:\.[0-9]+)?)', basename, re.IGNORECASE)
    aoa = float(aoa_match.group(1)) if aoa_match else 8.0

    return {'sigma': sigma, 'Re': Re, 'aoa': aoa}
    
def _parse_force_file(file_path):
    """Parse forceCoeffs.dat. Auto-detects column order from header. Returns t, cl, cd."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    header = None
    for line in lines:
        s = line.strip()
        if s.startswith('#') and 'Time' in s and ('Cl' in s or 'Cd' in s):
            header = s.lstrip('#').strip()
            break
    if header is None:
        raise ValueError(f"No header line found in {file_path}")
    tokens = header.split()
    lower = [t.lower() for t in tokens]
    cl_col = cd_col = None
    for name in ('cl', 'cl(t)', 'cl(total)', 'lift'):
        if name in lower:
            cl_col = lower.index(name); break
    for name in ('cd', 'cd(t)', 'cd(total)', 'drag'):
        if name in lower:
            cd_col = lower.index(name); break
    if cl_col is None or cd_col is None:
        raise ValueError(f"Could not find CL/CD columns in {file_path}")
    data = []
    for line in lines:
        s = line.strip()
        if not s or s.startswith('#'):
            continue
        try:
            row = [float(p) for p in s.split()]
            if len(row) > max(cl_col, cd_col):
                data.append(row)
        except ValueError:
            continue
    arr = np.array(data)
    return arr[:, 0], arr[:, cl_col], arr[:, cd_col]


def _find_force_file(case_dir):
    """Search common locations for forceCoeffs.dat."""
    candidates = [
        os.path.join(case_dir, 'postProcessing', 'forceCoeffs.dat'),
        os.path.join(case_dir, 'postProcessing', 'forces', 'forceCoeffs.dat'),
    ]
    pp = os.path.join(case_dir, 'postProcessing')
    if os.path.isdir(pp):
        for sub in os.listdir(pp):
            sp = os.path.join(pp, sub)
            if os.path.isdir(sp):
                candidates.append(os.path.join(sp, 'forceCoeffs.dat'))
                for sub2 in os.listdir(sp):
                    candidates.append(os.path.join(sp, sub2, 'forceCoeffs.dat'))
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def _re_to_string(Re):
    """Convert a Reynolds number to a compact filename-safe string."""
    exp = int(np.floor(np.log10(Re)))
    mantissa = Re / (10 ** exp)
    if abs(mantissa - round(mantissa)) < 1e-6:
        return f"{int(round(mantissa))}e{exp}"
    return f"{mantissa:.1f}e{exp}"


def _compute_U_from_Re(Re, c=0.07, nu=8.97e-7):
    """Freestream velocity from Reynolds number for water at ~25°C."""
    return Re * nu / c


def _read_openfoam_field(file_path):
    """Read internalField from an OpenFOAM field file (unchanged from your current version)."""
    with open(file_path, 'r') as f:
        content = f.read()

    content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
    content = re.sub(r'//[^\n]*', '', content)

    m = re.search(r'\binternalField\b\s+(\S+)', content)
    if m is None:
        raise ValueError(f"No internalField directive found in {file_path}")

    keyword = m.group(1)
    rest = content[m.end():]

    if re.fullmatch(r'uniform', keyword):
        vec = re.search(r'\(\s*([-+\d.eE\s]+)\)', rest)
        if vec is not None and vec.start() < 50:
            values = [float(v) for v in vec.group(1).split()]
            return np.array(values, dtype=np.float32)
        sca = re.search(r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', rest)
        if sca is not None:
            return np.array([float(sca.group(1))], dtype=np.float32)
        raise ValueError(f"Could not parse uniform value in {file_path}")

    if re.fullmatch(r'nonuniform', keyword):
        count_m = re.search(r'(\d+)\s*\(', rest)
        if count_m is None:
            raise ValueError(f"Could not find data block in {file_path}")
        open_paren = count_m.end() - 1

        depth = 0
        close_paren = -1
        for i in range(open_paren, len(rest)):
            c = rest[i]
            if c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
                if depth == 0:
                    close_paren = i
                    break
        if close_paren < 0:
            raise ValueError(f"Unbalanced parentheses in {file_path}")

        block = rest[open_paren + 1:close_paren]

        vec_entries = re.findall(r'\(([^()]+)\)', block)
        if vec_entries:
            vectors = [[float(v) for v in e.split()] for e in vec_entries]
            return np.array(vectors, dtype=np.float32)

        scalar_values = [
            float(v) for v in
            re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', block)
        ]
        if not scalar_values:
            raise ValueError(f"Empty nonuniform block in {file_path}")
        return np.array(scalar_values, dtype=np.float32)

    raise ValueError(f"Unrecognized internalField keyword '{keyword}' in {file_path}")


def _find_alpha_field(time_path):
    for name in ['alpha.vapour', 'alpha.water', 'alpha']:
        path = os.path.join(time_path, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"No alpha field found in {time_path}")


def preprocess_case(case_dir, output_dir):
    params = _parse_case_params(case_dir)
    sigma = params['sigma']
    Re = params['Re']
    aoa = params['aoa']
    U_inf = _compute_U_from_Re(Re)

    processed_dir = os.path.join(output_dir, 'processed')
    os.makedirs(processed_dir, exist_ok=True)

    print(f"\nProcessing case: {case_dir}")
    print(f"  sigma={sigma}, Re={Re:.3e}, AOA={aoa}°, U_inf={U_inf:.3f} m/s")

    MIN_TIME = 0.5
    time_dirs = [d for d in os.listdir(case_dir)
                 if os.path.isdir(os.path.join(case_dir, d))
                 and _is_time_dir(d)
                 and float(d) >= MIN_TIME]
    time_dirs.sort(key=float)

    if not time_dirs:
        print(f"  Warning: no time folders >= {MIN_TIME} found")
        return

    print(f"  using {len(time_dirs)} time folders, first = {time_dirs[0]}, last = {time_dirs[-1]}")

    fields_list = []
    times = []
    for time_dir in time_dirs:
        time_path = os.path.join(case_dir, time_dir)
        try:
            u = _read_openfoam_field(os.path.join(time_path, 'U'))
            p = _read_openfoam_field(os.path.join(time_path, 'p'))
            alpha_path = _find_alpha_field(time_path)
            alpha_v = _read_openfoam_field(alpha_path)

            if u.ndim == 1 and u.size % 3 == 0:
                u = u.reshape(-1, 3)
            if u.ndim == 1:
                u = u.reshape(-1, 1)
            if u.ndim == 2 and u.shape[1] >= 2:
                uv = u[:, :2]
            else:
                uv = np.concatenate([u, np.zeros((u.shape[0], 1), dtype=u.dtype)], axis=1)

            p = p.reshape(-1, 1)
            alpha_v = alpha_v.reshape(-1, 1)

            field = np.concatenate([uv, p, alpha_v], axis=1)
            fields_list.append(field)
            times.append(float(time_dir))
        except FileNotFoundError as exc:
            print(f"  skipped {time_dir}: missing field file ({exc})")
        except Exception as exc:
            print(f"  skipped {time_dir}: error reading fields ({exc})")

    if not fields_list:
        print(f"No valid time steps were processed for case {case_dir}")
        return

    fields = np.stack(fields_list, axis=0)
    times = np.array(times, dtype=float)
    
    # Load force coefficients from OpenFOAM runtime, aligned to snapshot times
    cl_target = np.zeros_like(times, dtype=np.float32)
    cd_target = np.zeros_like(times, dtype=np.float32)
    has_forces = False

    force_file = _find_force_file(case_dir)
    if force_file is not None:
        try:
            t_force, cl_force, cd_force = _parse_force_file(force_file)
            cl_target = np.interp(times, t_force, cl_force).astype(np.float32)
            cd_target = np.interp(times, t_force, cd_force).astype(np.float32)
            has_forces = True
            print(f"  loaded forceCoeffs.dat: {force_file}")
            print(f"  CL range over loaded interval: [{cl_target.min():.3f}, {cl_target.max():.3f}]")
        except Exception as exc:
            print(f"  WARNING: failed to parse forceCoeffs.dat: {exc}")
    else:
        print(f"  WARNING: no forceCoeffs.dat found for {case_dir}")
    
    re_str = _re_to_string(Re)
    output_file = os.path.join(processed_dir, f"CN_{sigma}_Re{re_str}.npz")
    
    np.savez_compressed(output_file,
                        fields=fields, times=times,
                        sigma=sigma, Re=Re, U_inf=U_inf, aoa=aoa,
                        cl_target=cl_target, cd_target=cd_target,
                        has_forces=has_forces)
    print(f"  Saved processed case to {output_file}")


def preprocess_all_cases(data_dir):
    case_dirs = [os.path.join(data_dir, d) for d in sorted(os.listdir(data_dir))
                 if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('CN_')]
    if not case_dirs:
        raise FileNotFoundError(f"No CN_* folders found in {data_dir}")
    for case_dir in case_dirs:
        preprocess_case(case_dir, data_dir)


if __name__ == '__main__':
    import yaml
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    preprocess_all_cases(config['data_dir'])
