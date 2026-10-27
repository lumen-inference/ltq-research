#!/bin/bash
# Round 2: Rust with launch_bounds(256, min_blocks) matching the nvcc register budget, interleaved with C++ (SASS).
set -e
cd /content/kbench
chmod +x oxide_gemv_lb
ptxas -arch=sm_120 -O3 -v oxide_gemv_lb.ptx -o rslb.cubin 2> res_rslb.txt
cuobjdump -sass rslb.cubin > rslb.sass
for rep in 1 2 3; do
  for v in cpp rslb; do
    rm -rf run2_${v}_$rep && mkdir -p run2_${v}_$rep && cd run2_${v}_$rep
    case $v in cpp) ../gemv_cpp ;; rslb) ../oxide_gemv_lb ;; esac > res.jsonl
    cd ..; echo "rep $rep $v done"
  done
done
tar czf results2.tgz run2_* res_rslb.txt rslb.sass oxide_gemv_lb.ptx
echo DONE
