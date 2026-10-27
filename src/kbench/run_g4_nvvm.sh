#!/bin/bash
# Round 4: NVVM IR from cuda-oxide (--emit-nvvm-ir) is compiled by libNVVM 13.0 (the nvcc optimizer) to PTX,
# the Rust binary loads it via KB_PTX. The original port (rsnvvm) and the "expert" one (rstunednvvm).
set -e
cd /content/kbench
chmod +x oxide_gemv_nvvm oxide_gemv_tuned_nvvm
python3 nvvm_compile.py oxide_gemv_nvvm.ll rsnvvm.ptx
python3 nvvm_compile.py oxide_gemv_tuned_nvvm.ll rstunednvvm.ptx
head -20 rsnvvm.ptx | grep -E "version|target"
ptxas -arch=sm_120 -O3 -v rsnvvm.ptx -o rsnvvm.cubin 2> res_rsnvvm.txt
ptxas -arch=sm_120 -O3 -v rstunednvvm.ptx -o rstunednvvm.cubin 2> res_rstunednvvm.txt
for rep in 1 2 3; do
  for v in cpp rsnvvm rstunednvvm; do
    rm -rf run4_${v}_$rep && mkdir -p run4_${v}_$rep && cd run4_${v}_$rep
    case $v in
      cpp) ../gemv_cpp ;;
      rsnvvm) KB_PTX=../rsnvvm.ptx ../oxide_gemv_nvvm ;;
      rstunednvvm) KB_PTX=../rstunednvvm.ptx ../oxide_gemv_tuned_nvvm ;;
    esac > res.jsonl 2> err.txt || { echo "FAIL $v"; tail -5 err.txt; }
    cd ..; echo "rep $rep $v done"
  done
done
tar czf results4.tgz run4_* res_rsnvvm.txt res_rstunednvvm.txt rsnvvm.ptx
echo DONE
