"""Score the A/B run: does the decision model's p_hard predict the FAST model's real failure?

A/B gold (objective): a step is HARD if the fast model, run on the state, does NOT reproduce
the agent's actual next action (the tool it actually called / its text reply). This is the
quantity routing is supposed to predict.

Compares three "hard" notions on the same sample:
  - A/B gold     : fast model fails to reproduce the actual action (objective).
  - judge gold   : strong model's hard/routine judgment (subjective reference).
  - p_hard       : the decision model's calibrated probability of hard.

Usage: python ab_score.py <ab_input.jsonl> <ab_out.jsonl> [--tau 0.85]
"""
import json, sys, argparse


def auc(y, s):
    """y: 0/1 labels, s: scores. Rank-based AUC (1 - 0.5*U/n0/n1 form)."""
    pairs = sorted(zip(s, y))
    n = len(pairs)
    # average ranks for ties
    ranks = [0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][1] == pairs[i][1] and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + j) / 2 + 1  # 1-based average rank over the tie block [i..j]
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1
    n1 = sum(y)
    n0 = n - n1
    if n1 == 0 or n0 == 0:
        return None
    rank_sum_pos = sum(r for r, (_, yy) in zip(ranks, pairs) if yy == 1)
    return (rank_sum_pos - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--tau", type=float, default=0.85)
    a = ap.parse_args()

    inp = {json.loads(l)["id"]: json.loads(l) for l in open(a.inp) if l.strip()}
    out = [json.loads(l) for l in open(a.out) if l.strip()]

    p_hard, ab_gold, judge_gold, pred_ok = [], [], [], 0
    conf = {}
    for r in out:
        row = inp.get(r["id"])
        if row is None:
            continue
        actual = row["actual"]
        p = r.get("pred")
        if not r.get("ok") or p is None:
            pred_ok = 0
            ab = 1  # error counts as a failure
        else:
            pred_ok = 1
            pred_tool = None if str(p).startswith("TEXT") else p
            if actual == "reply_without_tool":
                fail = not (pred_tool is None or pred_tool == "reply_without_tool")
            else:
                fail = pred_tool != actual
            ab = 1 if fail else 0
        p_hard.append(row["p_hard"])
        ab_gold.append(ab)
        judge_gold.append(1 if row.get("judge_gold") == "hard" else 0)
        key = (row.get("judge_gold"), "hard" if ab else "routine")
        conf[key] = conf.get(key, 0) + 1

    n = len(out)
    nfail = sum(ab_gold)
    nerr = sum(1 for r in out if not r.get("ok") or r.get("pred") is None)
    ntext = sum(1 for r in out if r.get("ok") and str(r.get("pred")).startswith("TEXT"))
    agree_actual = sum(1 for r in out if r.get("ok") and r.get("pred") == inp[r["id"]]["actual"])

    print(f"n={n}  ok={n - nerr}  errors={nerr}  TEXT(no tool)={ntext}")
    print(f"fast reproduces actual action: {agree_actual}/{n} = {agree_actual / n:.1%}")
    print(f"A/B gold HARD rate (fast fails): {nfail}/{n} = {nfail / n:.1%}")
    print(f"judge gold HARD rate:            {sum(judge_gold)}/{n} = {sum(judge_gold) / n:.1%}")
    print()
    print("AUC of p_hard vs gold:")
    print(f"  vs A/B gold (fast fails) : {auc(ab_gold, p_hard):.3f}")
    print(f"  vs judge gold            : {auc(judge_gold, p_hard):.3f}")
    print()
    print(f"p_hard at tau={a.tau}: predicts hard on {sum(1 for p in p_hard if p >= a.tau)}/{n}")
    for name, g in (("A/B gold", ab_gold), ("judge gold", judge_gold)):
        acc = sum(1 for p, gg in zip(p_hard, g) if (1 if p >= a.tau else 0) == gg) / n
        print(f"  routing accuracy @tau {a.tau} vs {name}: {acc:.1%}")
    print()
    print("confusion: (judge_gold, A/B gold) -> count")
    for k in sorted(conf, key=lambda k: -conf[k]):
        print(f"  judge={k[0]:8}  ab={k[1]:8}  {conf[k]}")


if __name__ == "__main__":
    main()
