#!/bin/bash
# Round 3: Rust with launch_bounds, ld.global.nc via ptx_asm! and the body inside the kernel itself, interleaved with C++ (SASS).
set -e
cd /content/kbench
chmod +x oxide_gemv_tuned
ptxas -arch=sm_120 -O3 -v oxide_gemv_tuned.ptx -o rstuned.cubin 2> res_rstuned.txt
cuobjdump -sass rstuned.cubin > rstuned.sass
for rep in 1 2 3; do
  for v in cpp rstuned; do
    rm -rf run3_${v}_$rep && mkdir -p run3_${v}_$rep && cd run3_${v}_$rep
    case $v in cpp) ../gemv_cpp ;; rstuned) ../oxide_gemv_tuned ;; esac > res.jsonl
    cd ..; echo "rep $rep $v done"
  done
done
tar czf results3.tgz run3_* res_rstuned.txt rstuned.sass oxide_gemv_tuned.ptx
echo DONE
