# Experiment manifests

These files list the configurations behind the archived results. Each non-comment
line defines one run; the original run values and order are preserved.

| Manifest | Runs | Fields |
|---|---:|---|
| `grid.txt` | 62 | `env,seed,nqs,minqs,dropout,maxsteps` |
| `tps.txt` | 24 | `env,seed,nqs,minqs,dropout,tps,maxsteps` |
| `onpolicy.txt` | 9 | `env,seed,nqs,minqs,dropout,maxsteps` |

`nqs` is ensemble size N; `minqs` is target subset size M; `dropout` is the critic
dropout rate; `tps` is the target-policy noise standard deviation. All listed runs
use 1,000,000 environment steps.

The grid and TPS manifests map to `data/june-2026/results/`. The on-policy
manifest maps to `data/onpolicy/results/` and requires `--onpolicy_probe=True`.
See [training instructions](../docs/training.md) for the common fixed settings
and commands. The analysis validator checks these manifests against the archive.
