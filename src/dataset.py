import numpy as np
import torch
from torch.utils.data import Dataset


REGIME_NAMES = ['Inception', 'Sheet', 'Cloud', 'Supercavitation']
NUM_REGIMES = 4


def assign_regime_from_sigma(sigma):
    """
    Assign cavitation regime from cavitation number sigma.

    Based on Clark-Y hydrofoil cavity characteristics:
        sigma > 0.92  ->  0 = Inception
        sigma > 0.55  ->  1 = Sheet
        sigma > 0.4  ->  2 = Cloud
        sigma <=  0.4  ->  3 = Supercavitation
    """
    if sigma > 0.92:
        return 0   # Inception
    elif sigma > 0.55:
        return 1   # Sheet
    elif sigma > 0.4:
        return 2   # Cloud
    else:
        return 3   # Supercavitation


class PRANOCavDataset(Dataset):
    """Memory-efficient dataset with regime labels."""

    def __init__(self, data_files, window_size, stats=None):
        self.window_size = window_size

        print(f"[dataset] loading {len(data_files)} files...")
        cases = []
        raw_cl = []
        raw_cd = []
        raw_has_forces = []
        raw_sigmas = []

        for file_path in data_files:
            npz = np.load(file_path)
            fields = npz['fields'].astype(np.float32)
            sigma  = float(npz['sigma'])  if 'sigma' in npz else 0.0
            aoa    = float(npz['aoa'])    if 'aoa'   in npz else 8.0
            Re     = float(npz['Re'])     if 'Re'    in npz else 7.8e5
            cond   = np.array([sigma, aoa, Re], dtype=np.float32)
            cases.append((fields, cond))
            raw_sigmas.append(sigma)

            regime = assign_regime_from_sigma(sigma)
            print(f"[dataset]   loaded {file_path}: "
                  f"fields={fields.shape}, sigma={sigma}, "
                  f"regime={REGIME_NAMES[regime]}")

            cl_t  = npz['cl_target'].astype(np.float32) if 'cl_target'  in npz.files \
                    else np.zeros(fields.shape[0], dtype=np.float32)
            cd_t  = npz['cd_target'].astype(np.float32) if 'cd_target'  in npz.files \
                    else np.zeros(fields.shape[0], dtype=np.float32)
            has_f = bool(npz['has_forces'])              if 'has_forces' in npz.files \
                    else False
            raw_cl.append(cl_t)
            raw_cd.append(cd_t)
            raw_has_forces.append(has_f)

        if stats is None:
            stacked    = np.concatenate([f.reshape(-1, f.shape[-1]) for f, _ in cases], axis=0)
            field_mean = stacked.mean(axis=0).astype(np.float32)
            field_std  = stacked.std(axis=0).astype(np.float32)
            field_std  = np.where(field_std < 1e-6, 1.0, field_std)
            cond_stack = np.stack([c for _, c in cases], axis=0)
            cond_mean  = cond_stack.mean(axis=0).astype(np.float32)
            cond_std   = cond_stack.std(axis=0).astype(np.float32)
            cond_std   = np.where(cond_std < 1e-6, 1.0, cond_std)
            self.stats = {
                'field_mean': field_mean, 'field_std': field_std,
                'cond_mean':  cond_mean,  'cond_std':  cond_std,
            }
        else:
            self.stats = {k: np.asarray(v, dtype=np.float32) for k, v in stats.items()}

        fm, fs = self.stats['field_mean'], self.stats['field_std']
        cm, cs = self.stats['cond_mean'],  self.stats['cond_std']

        self.case_fields         = []
        self.case_conds          = []
        self.cl_targets          = []
        self.cd_targets          = []
        self.has_forces_per_case = []
        self.regime_per_case     = []
        self.index               = []

        for ci, (fields, cond) in enumerate(cases):
            fields_n = ((fields - fm) / fs).astype(np.float32)
            cond_n   = ((cond   - cm) / cs).astype(np.float32)

            SUBSAMPLE_STRIDE = 10
            fields_n = fields_n[:, ::SUBSAMPLE_STRIDE, :]
            if ci == 0:
                print(f"[dataset] subsampling cells with stride={SUBSAMPLE_STRIDE}: "
                      f"N {fields.shape[1]} -> {fields_n.shape[1]}")

            self.case_fields.append(fields_n)
            self.case_conds.append(cond_n)
            self.cl_targets.append(raw_cl[ci])
            self.cd_targets.append(raw_cd[ci])
            self.has_forces_per_case.append(raw_has_forces[ci])

            regime = assign_regime_from_sigma(raw_sigmas[ci])
            self.regime_per_case.append(regime)

            T = fields_n.shape[0]
            for i in range(window_size, T):
                self.index.append((ci, i))

        regime_counts = [0] * NUM_REGIMES
        for r in self.regime_per_case:
            regime_counts[r] += 1
        print(f"[dataset] regime distribution across {len(cases)} cases:")
        for r_idx in range(NUM_REGIMES):
            print(f"  {REGIME_NAMES[r_idx]:18s}: {regime_counts[r_idx]} cases")

        print(f"[dataset] total windows = {len(self.index)}")

        n_check = min(200, len(self.index))
        idxs = np.linspace(0, len(self.index) - 1, n_check).astype(int)
        diffs_last = np.empty(n_check)
        diffs_zero = np.empty(n_check)
        for j, k in enumerate(idxs):
            ci, i = self.index[k]
            t    = self.case_fields[ci][i]
            last = self.case_fields[ci][i - 1]
            diffs_last[j] = np.mean((t - last) ** 2)
            diffs_zero[j] = np.mean(t ** 2)
        m_last = diffs_last.mean()
        m_zero = diffs_zero.mean()
        print(f"[dataset] mean MSE(target, last_frame) = {m_last:.4f}")
        print(f"[dataset] mean MSE(target, 0)          = {m_zero:.4f}")
        print(f"[dataset] implied corr r               = {1 - m_last/(2*m_zero):.4f}")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        ci, i    = self.index[idx]
        fields_n = self.case_fields[ci]
        cond_n   = self.case_conds[ci]

        history = fields_n[i - self.window_size:i].copy()
        target  = fields_n[i].copy()

        cl_val        = self.cl_targets[ci][i]
        cd_val        = self.cd_targets[ci][i]
        scalar_target = np.array([cl_val, cd_val], dtype=np.float32)
        scalar_mask   = np.float32(1.0 if self.has_forces_per_case[ci] else 0.0)

        regime_label = self.regime_per_case[ci]

        return (
            torch.from_numpy(history),
            torch.from_numpy(target),
            torch.from_numpy(cond_n),
            torch.from_numpy(scalar_target),
            torch.tensor(scalar_mask),
            torch.tensor(regime_label, dtype=torch.long),
        )
