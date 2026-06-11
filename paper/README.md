# paper/ — "Beyond Pessimism" NeurIPS-2026 workshop draft

Build: `cd paper && latexmk -pdf -outdir=build main.tex`
(or `bash analysis/run_all.sh` to regenerate figures+tables first).

**Nothing numeric is typed by hand.** Every number/figure/table comes from
`analysis/make_{figures,tables}.py` over the tidy CSVs; missing fleet results
render as red `[pending]` + a provisional banner. The draft therefore cannot
overstate what the data contains.

## When the fleet finishes (~2026-06-15): the 30-minute path to a full draft
1. `bash analysis/run_all.sh` — sync, rebuild everything, recompile.
2. Confirm the banner flipped (status.tex → `\provisionalfalse`) and no red
   `[pending]` remains (`grep -c pendingnum build/main.log` should drop).
3. Resolve EVERY `[FINALIZE]` comment (`grep -n FINALIZE main.tex`) and
   verify EVERY `\prov{...}`-wrapped results sentence against the regenerated
   tables (`grep -n 'prov{' main.tex`) — those daggers are the hand-written
   claims the auto-pipeline cannot retract. Key items: TPS outcome (state the
   observed direction, incl. refutation — pre-registered rule in
   analysis/README), M=1+dropout rescue interpretation (§4.3), σ-robustness
   requirement (App. C), diversity paragraph quantitative line, repo
   URL + anonymization.
4. Run `/results-audit` then `/claim-check paper/main.tex` (citation pass).
5. Swap in the target workshop's actual `neurips_2026.sty` + fit to its page
   limit (current: ~6.3pp main text; cut order if a 4-5pp limit applies:
   App. pointers stay, fold §5 Related into footnotes-style brevity, shrink
   Fig 1/2 to 0.85\linewidth, move §4.4 detail to appendix).

## Claim discipline
- Headline claims = June fleet only (multi-seed, probe-instrumented harness).
- April-era numbers appear ONLY in App. D replication table.
- TPS arms: distribution stats only — never seed-paired (pre-registered).
- The prediction for TPS was fixed before the runs: dose-dependent gain at
  N=2, ~none at N=10. If the data lands otherwise, §4.3 reports it as a
  (partial) refutation — do not soften, do not drop the section.

## Venue notes (Aug–Sept 2026 deadlines)
Candidates (check CFPs when announced): NeurIPS workshops in offline/online
RL, RL theory-practice gap, science-of-deep-learning. Non-archival preferred
(keeps an ICLR-2028 main-track option open). Levine-orbit relevance: extends
RLPD; future-work hook = FQL (Seohong Park) — send him the camera-ready.
