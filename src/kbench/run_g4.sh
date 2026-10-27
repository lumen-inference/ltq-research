#!/bin/bash
# Kernel-language comparison run on G4: C++ is built here (nvcc 13.0) both directly to SASS and as PTX for driver JIT,
# Rust (cuda-oxide) arrives as a prebuilt binary with embedded PTX (driver JIT at load time).
set -e
cd /content/kbench
nvcc --version | tail -1
nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.max.sm --format=csv,noheader
nvcc -O3 -arch=sm_120 -o gemv_cpp cpp/gemv.cu
nvcc -O3 -gencode arch=compute_120,code=compute_120 -o gemv_cpp_ptx cpp/gemv.cu
nvcc -O3 -arch=sm_120 -ptx -o cpp.ptx cpp/gemv.cu
chmod +x oxide_gemv
cuobjdump -res-usage gemv_cpp > res_cpp.txt
ptxas -arch=sm_120 -O3 -v oxide_gemv.ptx -o rs.cubin 2> res_rs.txt
cuobjdump -sass gemv_cpp > cpp.sass
cuobjdump -sass rs.cubin > rs.sass
for rep in 1 2 3; do
  for v in cpp cpp_ptx rs; do
    rm -rf run_${v}_$rep && mkdir -p run_${v}_$rep && cd run_${v}_$rep
    s=$(date +%s.%N)
    case $v in
      cpp) ../gemv_cpp ;;
      cpp_ptx) ../gemv_cpp_ptx | sed 's/"lang":"cpp"/"lang":"cpp_ptx"/' ;;
      rs) ../oxide_gemv ;;
    esac > res.jsonl
    echo "{\"lang\":\"$v\",\"kind\":\"wall\",\"rep\":$rep,\"s\":$(echo "$(date +%s.%N) - $s" | bc)}" >> res.jsonl
    cd ..
    echo "rep $rep $v done"
  done
done
tar czf results.tgz run_* res_*.txt cpp.sass rs.sass cpp.ptx oxide_gemv.ptx
echo DONE
