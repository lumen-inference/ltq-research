"""Summary of the CUDA C++ vs cuda-oxide comparison: bit-exact output check, median speed over runs, registers, SASS.

    python3 compare.py <directory with unpacked results.tgz> [results.json]
"""
import glob
import json
import os
import re
import statistics
import sys

d = sys.argv[1]
rows = []
for f in glob.glob(f"{d}/run_*/res.jsonl"):
    for line in open(f):
        rows.append(json.loads(line))

# 1. output check: Rust vs C++ (SASS and PTX) on identical data
same = {}
for f in sorted(glob.glob(f"{d}/run_rs_1/out/y_rs_*.bin")):
    tag = os.path.basename(f)[5:-4]
    yr = open(f, "rb").read()
    yc = open(f"{d}/run_cpp_1/out/y_cpp_{tag}.bin", "rb").read()
    yp = open(f"{d}/run_cpp_ptx_1/out/y_cpp_{tag}.bin", "rb").read()
    same[tag] = {"rs==cpp": yr == yc, "rs==cpp_ptx": yr == yp, "cpp==cpp_ptx": yc == yp}
print("y check (64 rows, bit for bit):", json.dumps(same, ensure_ascii=False))

# 2. speed: median over three runs
med = {}
for r in rows:
    if r["kind"] == "gemv":
        med.setdefault((r["lang"], r["mode"], r["K"], r["VAR"], r["R"]), []).append(r)
table = {}
for (lang, mode, K, VAR, R), rs in med.items():
    table[(lang, mode, K, VAR, R)] = {"GBps": statistics.median(x["GBps"] for x in rs), "regs": rs[0]["regs"],
                                      "grid": rs[0]["grid"], "local": rs[0]["local_bytes"]}
print(f"\n{'mode':5s} {'K':>2s} {'var.':>12s} {'R':>2s} | {'C++ SASS':>9s} {'C++ PTX':>9s} {'Rust':>9s} GB/s | Rust/C++ | regs C++/Rust | blocks C++/Rust")
ratios = {"DRAM": [], "L2": []}
for mode in ("DRAM", "L2"):
    for K in (2, 3, 4, 5):
        for VAR in (0, 1):
            for R in (2, 4):
                c, p, s = (table.get((l, mode, K, VAR, R)) for l in ("cpp", "cpp_ptx", "rs"))
                if not (c and s):
                    continue
                ratios[mode].append(s["GBps"] / c["GBps"])
                print(f"{mode:5s} {K:2d} {'shared table' if VAR == 0 else 'popc+dup':>12s} {R:2d} | {c['GBps']:9.1f} {p['GBps']:9.1f} {s['GBps']:9.1f}"
                      f" | {s['GBps'] / c['GBps']:8.3f} | {c['regs']:3d} / {s['regs']:3d} | {c['grid']} / {s['grid']}")
geo = lambda xs: statistics.geometric_mean(xs)
print(f"\ngeometric mean Rust/C++: DRAM {geo(ratios['DRAM']):.3f} (min {min(ratios['DRAM']):.3f}, max {max(ratios['DRAM']):.3f}),"
      f" L2 {geo(ratios['L2']):.3f} (min {min(ratios['L2']):.3f}, max {max(ratios['L2']):.3f})")

# 3. misc: memory ceiling, launch, wall time
for kind in ("read_bw", "launch", "wall"):
    for r in rows:
        if r["kind"] == kind and (kind != "wall" or True):
            pass
print("read ceiling:", statistics.median(r["GBps"] for r in rows if r["kind"] == "read_bw"), "GB/s")
for lang in ("cpp", "cpp_ptx", "rs"):
    L = [r for r in rows if r["kind"] == "launch" and r["lang"] == lang]
    W = [r for r in rows if r["kind"] == "wall" and r["lang"] == lang]
    print(f"{lang:8s} empty kernel launch: enqueue {statistics.median(x['enqueue_us'] for x in L):.3f} us, "
          f"with execution {statistics.median(x['total_us'] for x in L):.3f} us; whole run {statistics.median(x['s'] for x in W):.1f} s")

# 4. SASS: instruction count in the kernel body K=4 VAR=1 R=4
def sass_funcs(path):
    out, cur = {}, None
    for line in open(path):
        m = re.match(r"\s+Function : (\S+)", line)
        if m:
            cur = m.group(1); out[cur] = 0
        elif cur and re.match(r"\s+/\*[0-9a-f]{4}\*/", line):
            out[cur] += 1
    return out
sc, sr = sass_funcs(f"{d}/cpp.sass"), sass_funcs(f"{d}/rs.sass")
pick = lambda funcs, pat: {k: v for k, v in funcs.items() if re.search(pat, k)}
print("\nSASS, instructions per kernel (C++):", {k[-40:]: v for k, v in pick(sc, r"ltq_matvec").items()})
print("SASS, instructions per kernel (Rust):", {k[-60:]: v for k, v in pick(sr, r"ltq_matvec").items()})
if len(sys.argv) > 2:
    json.dump({"same_y": same, "table": {"|".join(map(str, k)): v for k, v in table.items()},
               "geo_rs_over_cpp": {m: geo(v) for m, v in ratios.items()}}, open(sys.argv[2], "w"), ensure_ascii=False, indent=1)
