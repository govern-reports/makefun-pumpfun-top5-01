#!/usr/bin/env python3
"""pumpfun-top5-forecast-01 entry: forecast the 5 largest pump.fun coins by
market cap at the house's seven answer reads, from the 10-minute snapshots.

Model (built from the kit's practice history, no tuning on the answer):
  * Per coin, a log10 market-cap (in SOL) series from the marks. A least-squares
    slope over the last 2 hours of marks gives momentum; the slope is damped
    with the forecast horizon (momentum fades: exp(-H/6)) and a venue-dependent
    downward drift is added (curve coins decay faster than graduated pool
    coins). The projected log-cap at the mean answer read is Normal(mu, sigma)
    with sigma growing with the horizon (the practice history's hourly log-cap
    moves have sd ~0.72, median 0, so long-horizon caps are very uncertain).
  * A fixed-seed Monte Carlo (seed 20261112) draws every candidate coin's
    log-cap, ranks them, and counts how often each coin takes each of the 5
    places and how large each place's cap is. That gives honest, simulation
    based place probabilities and place-size distributions, including rank
    correlations (a coin cannot hold two places at once).
  * Coins born after the history ends cannot be named, and the practice shows
    they take a large share of the top 5 (3 of 5 at its last read). So a
    horizon-scaled share U of every place is left unnamed for them
    (U = 0.30 + 0.02*H, capped at 0.48); the named probabilities are scaled by
    (1 - U). The unnamed mass also covers known coins outside the candidate
    set. The size forecast mixes the simulated winner sizes with a broad
    newcomer size distribution centred on the simulated median winner size.
  * top5 is the greedy argmax assignment of the simulated place probabilities.

Deterministic: fixed PRNG seed, sorted iteration, no wall-clock, no network,
stdlib only. Two runs on the same history write byte-identical output.
"""

import bisect
import json
import math
import os
import random
import sys
from datetime import datetime

SEED = 20261112
N_SAMPLES = 5000
MAX_CANDIDATES = 240
MIN_P = 0.0015          # smallest named place probability
MIN_SIZE_P = 0.0005     # smallest named size-bin probability
MICRO = 1_000_000       # probabilities are floored to 6 decimals


def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def logcap_sol(market_cap_lamports):
    return math.log10(int(market_cap_lamports)) - 9.0


def ols_slope(xs, ys):
    """Least-squares slope of ys on xs; 0.0 when degenerate."""
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0.0:
        return 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / sxx


def bin_of(x, edges):
    """Size bin of log10 cap x: 0 below edges[0], len(edges) at/above last."""
    if x < edges[0]:
        return 0
    if x >= edges[-1]:
        return len(edges)
    return bisect.bisect_right(edges, x)


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

    with open(os.path.join(input_dir, "task.json"), "r", encoding="utf-8") as f:
        task = json.load(f)
    places = int(task["places"])
    edges = [float(e) for e in task["size"]["edges"]]
    output_name = task.get("output", "forecast.json")

    marks = []
    with open(os.path.join(input_dir, "history.ndjson"), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = json.loads(line)
            if m.get("ok") is True and isinstance(m.get("rows"), list):
                marks.append(m)
    if not marks:
        raise SystemExit("the history holds no ok mark")
    t0 = parse_ts(marks[0]["t0"])
    history_until = parse_ts(task["history_until"])

    # Forecast horizon H: mean answer read minus history end, in hours.
    reads = [parse_ts(r) for r in task.get("reads", [])]
    if reads:
        horizon_h = sum((r - history_until).total_seconds() for r in reads) / len(reads) / 3600.0
    else:
        horizon_h = 1.0
    horizon_h = max(0.05, horizon_h)

    # Per-mint time series of (t_hours, logcap, rank, venue, created_at).
    series = {}
    last_mark_with_rows = None
    for m in marks:
        if not m["rows"]:
            continue
        last_mark_with_rows = m
        t = (parse_ts(m["at"]) - t0).total_seconds() / 3600.0
        for r in m["rows"]:
            series.setdefault(r["mint"], []).append(
                (t, logcap_sol(r["market_cap_lamports"]), int(r["rank"]),
                 r.get("venue", "curve"), r.get("created_at"))
            )
    if last_mark_with_rows is None:
        raise SystemExit("the history holds no mark with rows")
    cut_t = (parse_ts(last_mark_with_rows["at"]) - t0).total_seconds() / 3600.0
    last_rows = sorted(last_mark_with_rows["rows"], key=lambda r: int(r["rank"]))

    # Candidate set: the last mark's ranked coins, plus recent risers that fell
    # out of the last mark's rows but peaked high in the last 2 hours.
    cand = {}
    for r in last_rows:
        cand[r["mint"]] = True
    for mint, pts in series.items():
        if mint in cand:
            continue
        recent = [p for p in pts if p[0] >= cut_t - 2.0]
        if recent and max(p[1] for p in recent) >= 2.5 and recent[-1][0] >= cut_t - 1.0:
            cand[mint] = True
    if len(cand) > MAX_CANDIDATES:
        # keep the highest last-observed caps (never an alphabetical slice)
        def last_cap(mint):
            pts = series.get(mint) or []
            return pts[-1][1] if pts else -99.0
        candidates = sorted(cand, key=lambda m: (-last_cap(m), m))[:MAX_CANDIDATES]
    else:
        candidates = sorted(cand)

    # Per-coin projected distribution at the mean answer read.
    damp = math.exp(-horizon_h / 6.0)
    sigma_base = math.sqrt(0.25 ** 2 + 0.25 ** 2 * horizon_h + (0.10 * horizon_h) ** 2)
    proj = {}  # mint -> (mu, sigma)
    for mint in candidates:
        pts = series[mint]
        y_last = pts[-1][1]
        venue = pts[-1][3]
        # slope over the last 2 hours of marks (or the whole span if shorter)
        win = [p for p in pts if p[0] >= cut_t - 2.0]
        if len(win) >= 3:
            slope = ols_slope([p[0] for p in win], [p[1] for p in win])
        elif len(pts) >= 3:
            slope = ols_slope([p[0] for p in pts], [p[1] for p in pts])
        else:
            slope = 0.0
        # clamp wild slopes from 10-minute moves
        slope = max(-1.5, min(1.5, slope))
        drift = -0.06 if venue == "pool" else -0.12
        mu = y_last + (slope * damp + drift) * horizon_h
        # a coin cannot be projected far above where it has ever been on trend
        # alone, nor below the bonding-curve floor the chart piles up at
        mu = max(1.2, min(mu, y_last + 1.0))
        sigma = sigma_base + (0.10 if len(win) < 3 else 0.0)
        proj[mint] = (mu, sigma)

    # Fixed-seed Monte Carlo over the candidates.
    rng = random.Random(SEED)
    n = len(candidates)
    mus = [proj[m][0] for m in candidates]
    sigs = [proj[m][1] for m in candidates]
    place_counts = [[0] * n for _ in range(places)]
    place_caps = [[] for _ in range(places)]  # winner log-caps per place
    for _ in range(N_SAMPLES):
        draws = [rng.gauss(mus[i], sigs[i]) for i in range(n)]
        order = sorted(range(n), key=lambda i: (-draws[i], candidates[i]))
        for k in range(places):
            idx = order[k]
            place_counts[k][idx] += 1
            place_caps[k].append(draws[idx])

    # Unnamed share for coins born after the cut (grows with the horizon).
    unnamed = min(0.30 + 0.02 * horizon_h, 0.48)
    named_mass = 1.0 - unnamed

    probabilities = []
    place_prob = []  # per place: list of (mint, p) final, for top5 assignment
    # Coins the simulation never saw win any place: naming them at p 0 (the
    # rules' own device) takes them out of a place's unnamed share, leaving
    # that share to the chart's remaining coins - the newcomers above all -
    # at the cost of the floor if one of them does win. Only the hopeless
    # ones qualify: never won in 5000 draws, projected far below the place's
    # typical winner, and ranked mid-chart (likely to stay on the chart).
    never_won = [i for i in range(n)
                 if all(place_counts[k][i] == 0 for k in range(places))]
    for k in range(places):
        rows = []
        for i in range(n):
            p = place_counts[k][i] / N_SAMPLES * named_mass
            if p >= MIN_P:
                rows.append((candidates[i], math.floor(p * MICRO) / MICRO))
        rows.sort(key=lambda r: (-r[1], r[0]))
        place_prob.append(rows)
        for mint, p in rows:
            if p > 0.0:
                probabilities.append({"mint": mint, "place": k + 1, "p": p})
        named_here = {mint for mint, _ in rows}
        caps_k = sorted(place_caps[k])
        med_k = caps_k[len(caps_k) // 2] if caps_k else 3.0
        last_rank = {r["mint"]: int(r["rank"]) for r in last_rows}
        hopeless = [candidates[i] for i in never_won
                    if candidates[i] not in named_here
                    and proj[candidates[i]][0] <= med_k - 1.5
                    and last_rank.get(candidates[i], 999) >= 50]
        hopeless.sort(key=lambda m: (proj[m][0], m))
        for mint in hopeless[:60]:
            probabilities.append({"mint": mint, "place": k + 1, "p": 0})

    # top5: greedy argmax per place, each coin used once.
    used = set()
    top5 = []
    for k in range(places):
        best = None
        for mint, p in place_prob[k]:
            if mint not in used:
                best = mint
                break
        if best is None:
            # fall back to the highest projected coin not yet used
            for mint in sorted(proj, key=lambda m: (-proj[m][0], m)):
                if mint not in used:
                    best = mint
                    break
        used.add(best)
        top5.append(best)

    # Sizes: simulated winner caps (weight named_mass) mixed with a broad
    # newcomer distribution centred on the simulated median winner cap.
    sizes = []
    n_bins = len(edges) + 1
    for k in range(places):
        caps = sorted(place_caps[k])
        med = caps[len(caps) // 2] if caps else 3.0
        hist = [0.0] * n_bins
        for c in caps:
            hist[bin_of(c, edges)] += named_mass / N_SAMPLES
        # newcomer mass: normal around the median winner size, sd 0.55
        # (integrated over the bins by sampling the same fixed-seed stream)
        new_hist = [0.0] * n_bins
        for _ in range(2000):
            new_hist[bin_of(rng.gauss(med, 0.55), edges)] += 1.0
        for b in range(n_bins):
            p = hist[b] + unnamed * (new_hist[b] / 2000.0)
            p = math.floor(p * MICRO) / MICRO
            if p >= MIN_SIZE_P:
                sizes.append({"place": k + 1, "bin": b, "p": p})
    sizes.sort(key=lambda r: (r["place"], r["bin"]))

    forecast = {"probabilities": probabilities, "top5": top5, "sizes": sizes}
    out_path = os.path.join(output_dir, output_name)
    os.makedirs(output_dir, exist_ok=True)
    text = json.dumps(forecast, separators=(",", ":")) + "\n"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
