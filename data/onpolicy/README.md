# Supplementary on-policy sharpness probe

Nine pen runs at seed 0 measure the ensemble-mean critic at both fixed offline
state-action pairs and the actor's actions on recently visited states. The grid
contains N ∈ {2, 4, 6, 10} with and without dropout, plus the N=2, M=1 no-dropout
condition. Each run trains for 1M steps; configurations are listed in
[experiments/onpolicy.txt](../../experiments/onpolicy.txt).

```bash
python analysis/validate_data.py --onpolicy
python analysis/onpolicy_analysis.py
```

Among the eight M=2 configurations, final normalized on-policy sharpness has a
Spearman correlation of −0.83 with score (permutation p=0.016), compared with −0.43
for the offline probe (p=0.30). These are exploratory, single-seed associations.
The analysis script also prints raw roughness, wider-window checks, and
leave-one-configuration-out ranges.

The result does not identify a causal mechanism. Roughness and Q magnitude evolve
together, and the M=1 condition exhibits large value growth. Excluding M=1 does
not resolve that confound. There is only one seed per configuration, and the
design changes both the states and actions at which sharpness is measured.
Leave-one-configuration-out checks cannot establish seed stability.

These runs are retained as supplementary evidence and are excluded from the
paper's headline claims. A causal follow-up would need controls for Q magnitude,
separate interventions on probe states and actions, and multiple seeds.
