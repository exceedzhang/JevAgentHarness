#!/usr/bin/env python
"""Compare a gated in-loop run against its no-gate control: which episodes flipped."""
import json
import sys

def load(p):
    d = {}
    for l in open(p):
        l = l.strip()
        if not l:
            continue
        r = json.loads(l)
        d[r["task_id"]] = r
    return d

def succ(r):
    return bool(r.get("result", {}).get("success"))

def main(ctrl_p, gate_p):
    ctrl = load(ctrl_p)
    gate = load(gate_p)
    common = sorted(set(ctrl) & set(gate))
    cs = sum(1 for t in common if succ(ctrl[t]))
    gs = sum(1 for t in common if succ(gate[t]))
    print(f"control: {cs}/{len(common)} success")
    print(f"gated:   {gs}/{len(common)} success   (delta {gs-cs:+d})")

    recovered = [t for t in common if not succ(ctrl[t]) and succ(gate[t])]
    regressed = [t for t in common if succ(ctrl[t]) and not succ(gate[t])]
    print(f"\nRECOVERED (ctrl FAIL -> gate PASS): {len(recovered)}")
    for t in sorted(recovered):
        print(f"  + {t}")
    print(f"\nREGRESSED (ctrl PASS -> gate FAIL): {len(regressed)}")
    for t in sorted(regressed):
        print(f"  - {t}")

    def nblocks(t):
        return sum(1 for st in gate[t].get("steps", [])
                   if "blocked by pre-action gate" in str(st.get("observation", "")))

    total_blocks = sum(nblocks(t) for t in gate)
    eps_with_block = sum(1 for t in gate if nblocks(t) > 0)
    print(f"\ngate fired: total block observations={total_blocks}, episodes with >=1 block={eps_with_block}/{len(gate)}")

    # Among the 36 control-FAILS: how many got blocked, how many recovered
    cfails = [t for t in common if not succ(ctrl[t])]
    cf_blocked = [t for t in cfails if nblocks(t) > 0]
    print(f"\ncontrol-fails: {len(cfails)} total; gated-run had a block in {len(cf_blocked)} of them")
    # Among the 264 control-SUCCESSes: how many got blocked, how many regressed
    csuccs = [t for t in common if succ(ctrl[t])]
    cs_blocked = [t for t in csuccs if nblocks(t) > 0]
    print(f"control-passes: {len(csuccs)} total; gated-run had a block in {len(cs_blocked)} of them (regressed {len(regressed)})")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
