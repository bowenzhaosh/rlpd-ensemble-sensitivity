# On-policy actor-side sharpness probe — VERDICT (2026-06-15)

**Experiment** (job 79790, gpu-linuxlab RTX4000, 9/9 complete, 0 fail): pen
N∈{2,4,6,10}×{nodrop,drop0.01} + M=1 nodrop, seed 0, 1M steps, each logging
sharpness of the ensemble-mean Q at BOTH the fixed offline (s,a_dataset) probe
and the actor's own actions (s, π(s)) on on-policy states. Addresses the paper's
Limitation (i) / the peer reviewers' top-impact ask.

## Result (reproducible)
Config-level Spearman(sharpness, final score), permutation p, leave-one-config-out:
- **on-policy** excl-M1 (n=8): ρ=**−0.83** (p=.016, LOCO [−0.96,−0.75]); all (n=9) ρ=−0.67 (p=.057)
- **offline**  excl-M1 (n=8): ρ=−0.43 (p=.30, LOCO [−0.89,−0.14]); all (n=9) ρ=+0.00 (p=1.0)
- raw on-policy roughness vs score: ρ=−0.88 (p=.006); wider window (>600k) ρ=−0.86
- interaction (single-seed, descriptive): dropout dlog-sharp @N2/@N10 — offline (−1.45,−0.86), on-policy (−0.69,**+0.17**) → on-policy mirrors the score interaction better.

So on-policy sharpness tracks score MUCH better than the offline probe — the
predicted direction, statistically robust to LOCO/window/raw.

## Adversarial audit (results-auditor, opus): UNTRUSTWORTHY as a mechanism claim
1. **[BLOCKING] Q-overestimation confound.** Within-run trajectories: M1's on-policy
   roughness is NORMAL early (0.0049, |Q|=28 @50k) then EXPLODES with the Q-runaway
   (14200, |Q|=63000 @1M) — divergence over *time*, not failure-states; N10nd (good)
   roughness stays low/flat with |Q| bounded; N2nd (bad) |Q| creeps 16→52 and roughness
   0.013→0.068. Roughness, |Q|, and score are ALL downstream of how much N lets Qbar
   overestimate → COMMON CAUSE, not "smooth critic → good policy". Holds even excl-M1
   (healthy configs still show the |Q|↔score gradient). Rules out the simple success-state
   circularity but replaces it with a Q-magnitude/time confound the experiment can't break.
2. **[BLOCKING] Single seed.** n=1/config, GPU-nondeterministic; 4 configs in a score
   dead-heat with interleaved roughness ranks that could reorder under another seed.
   LOCO certifies no single-CONFIG leverage, NOT seed stability.
3. **[MAJOR] Noise paradox.** On-policy per-probe CV is 5–25× the offline; it "wins"
   because the 160× between-config spread is dominated by the same Q-divergence confound.
4. **[MINOR] Offline baseline is fair but near-degenerate** (~0.001 floor, non-monotonic),
   so almost any state/action-varying probe beats it — low bar; need a random-action control.

## Defensible (weakened) statement
"On-policy sharpness (states+actions) is ASSOCIATED with final score at single seed,
more than the offline-dataset probe (−0.88 vs −0.43, n=8); CONSISTENT WITH but does NOT
isolate the action-region mechanism — it is confounded with Qbar overestimation magnitude
(which N controls), and states-vs-actions was not decomposed." NOT "the actor's action
region is sharper", NOT causal "predicts", NOT general (single env/seed).

## DECISION: STOP escalating; do NOT integrate as a claim; keep the paper's honest framing.
This is the 3rd positive this session killed/qualified by adversarial audit (temporal
precedence, mediator interaction, M=1 power-law were the others). Consistent message: the
sharpness↔score mechanism is genuinely entangled with overestimation and weak — the
paper's current honest framing (sharpness = config-level descriptor, mechanism unproven)
is CORRECT and must not be overclaimed. Adding a confounded single-seed result would repeat
the overclaiming the honesty pass removed.

## Proposed (NOT run) follow-up — only worth it for an ARCHIVAL version, not the workshop
The one experiment that would actually isolate the mechanism:
(a) states×actions decomposition at matched |Q|: roughness at (s_dataset, π(s_dataset))
    vs (s_onpolicy, a_dataset) — separates "action choice makes Qbar rough" (intended
    mechanism) from "states/Q-magnitude" (confound); needs new runs (no checkpoints saved);
(b) ≥3 seeds/config; (c) a random-off-distribution-action control probe.
Estimate: 2–3 more RTX4000 fleets (~1–2 days). Recommend AGAINST for the workshop —
marginal value, high rabbit-hole risk; the honest framing already suffices.
