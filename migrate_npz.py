"""
One-time migration: convert legacy 'CN_<sigma>.npz' files into the new
'CN_<sigma>_Re<value>.npz' format with U_inf and aoa metadata.

Run this once after updating preprocess.py. It does NOT touch raw OpenFOAM data —
it only renames and re-saves the already-processed .npz files.

Usage:
    python -m src.migrate_npz
"""
import os
import shutil
import numpy as np


def _re_to_string(Re):
    """Match the filename convention used in preprocess.py."""
    exp = int(np.floor(np.log10(Re)))
    mantissa = Re / (10 ** exp)
    if abs(mantissa - round(mantissa)) < 1e-6:
        return f"{int(round(mantissa))}e{exp}"
    return f"{mantissa:.1f}e{exp}"


def _compute_U_from_Re(Re, c=0.07, nu=8.97e-7):
    """Freestream velocity from Reynolds (water, ~25°C, c=0.07 m)."""
    return Re * nu / c


def migrate(processed_dir='data/processed',
            default_Re=7.8e5,
            default_aoa=8.0,
            backup=True):

    if not os.path.exists(processed_dir):
        raise FileNotFoundError(f"Processed directory not found: {processed_dir}")

    files = sorted(f for f in os.listdir(processed_dir)
                   if f.startswith('CN_') and f.endswith('.npz'))
    if not files:
        print(f"No CN_*.npz files in {processed_dir}")
        return

    if backup:
        backup_dir = processed_dir + '_legacy_backup'
        os.makedirs(backup_dir, exist_ok=True)
        print(f"Backup directory: {backup_dir}")

    converted, skipped = 0, 0
    for fname in files:
        old_path = os.path.join(processed_dir, fname)

        # Skip files already in new format
        if '_Re' in fname:
            print(f"  skip   (already new format): {fname}")
            skipped += 1
            continue

        npz = np.load(old_path)
        fields = npz['fields']
        times = npz['times']
        sigma = float(npz['sigma'])
        Re = float(npz['Re']) if 'Re' in npz else default_Re
        U_inf = _compute_U_from_Re(Re)
        aoa = default_aoa  # the legacy 'alpha' field was a placeholder, not real AOA

        re_str = _re_to_string(Re)
        new_fname = f"CN_{sigma}_Re{re_str}.npz"
        new_path = os.path.join(processed_dir, new_fname)

        if backup:
            shutil.copy2(old_path, os.path.join(backup_dir, fname))

        np.savez_compressed(new_path,
                            fields=fields, times=times,
                            sigma=sigma, Re=Re, U_inf=U_inf, aoa=aoa)
        os.remove(old_path)
        converted += 1
        print(f"  convert: {fname} -> {new_fname}")
        print(f"           sigma={sigma}, Re={Re:.3e}, U_inf={U_inf:.3f}, aoa={aoa}")

    print(f"\nDone. Converted {converted}, skipped {skipped}.")
    if backup:
        print(f"Backups of original files are in {backup_dir}.")


if __name__ == '__main__':
    migrate()
