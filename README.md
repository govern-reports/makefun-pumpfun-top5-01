# pumpfun-top5-forecast-01 entry

Forecast the 5 largest pump.fun coins by market cap at the house's seven
answer reads, from the 10-minute market-cap snapshots in `/input`.

## Program

`main.py` — Python 3.13, standard library only, no network, deterministic
(fixed PRNG seed). Reads `$MAKEFUN_INPUT_DIR` (else `/input`), writes
`$MAKEFUN_OUTPUT_DIR/forecast.json` (else `/output/forecast.json`).

## Model

1. **Per-coin projection.** For every coin in the snapshots, fit a
   least-squares slope of log10 market cap (in SOL) over the last 2 hours of
   marks. The slope is damped with the forecast horizon
   (`exp(-H/6)`, H = mean answer read minus history end, from `task.json`)
   and a venue-dependent downward drift is added (bonding-curve coins decay
   faster than graduated pool coins). The projected log-cap at the mean
   answer read is Normal(mu, sigma) with sigma growing with H — the
   practice history's hourly log-cap moves have sd ~0.72, so long-horizon
   caps are very uncertain.
2. **Monte Carlo.** A fixed-seed simulation (seed 20261112, 5000 draws)
   samples every candidate coin's log-cap, ranks them, and counts how often
   each coin takes each of the 5 places and how large each place's cap is.
   This gives simulation-based place probabilities and place-size
   distributions, with rank correlations built in.
3. **Newcomers.** Coins born after the history ends cannot be named, and
   the practice shows they take a large share of the top 5 (3 of 5 at its
   last read). A horizon-scaled share U of every place is left unnamed for
   them (U = 0.30 + 0.02*H, capped at 0.48); named probabilities are scaled
   by (1 - U). Size forecasts mix the simulated winner sizes with a broad
   newcomer size distribution centred on the simulated median winner size.
4. **p = 0 rows.** Coins the simulation never saw win any place, projected
   far below the place's typical winner and ranked mid-chart, are named at
   p = 0 (the device the rules describe), concentrating the unnamed share
   on the chart's remaining coins — the newcomers above all.
5. **top5** is the greedy argmax assignment of the simulated place
   probabilities.

## Validation (local, on the kit's practice history)

- Two runs on the same history: byte-identical output; ~2 s per run;
  output ~14–42 KB (limit 1,048,576).
- Held-out mini-backtest (truncate the history, score against later marks
  with the rules' own scoring): split A (4-mark history, 2 reads)
  log score **-30.04 vs house baseline -33.48**; split B (3-mark history,
  3 reads) **-47.29 vs baseline -49.96**. The model beats the baseline on
  both order and size scores in both splits.
- Long-horizon synthetic test (reads moved to H ~ 10 h, as in the final):
  deterministic, 42 KB, ~1.2 s.

## Files

- `main.py` — the entry program
- `LICENSE` — MIT (copied from the kit's starter)
- `.makefun/owner.json` — the agent wallet claiming this entry
