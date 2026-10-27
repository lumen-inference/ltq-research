"""NVVM IR -> PTX via libNVVM (the nvcc optimizer): python3 nvvm_compile.py in.ll out.ptx [-arch=compute_120 ...]"""
import ctypes
import sys

nv = ctypes.CDLL("/usr/local/cuda/nvvm/lib64/libnvvm.so")
prog = ctypes.c_void_p()
assert nv.nvvmCreateProgram(ctypes.byref(prog)) == 0
src = open(sys.argv[1], "rb").read()
assert nv.nvvmAddModuleToProgram(prog, src, ctypes.c_size_t(len(src)), b"oxide") == 0
lib = open("/usr/local/cuda/nvvm/libdevice/libdevice.10.bc", "rb").read()
assert nv.nvvmLazyAddModuleToProgram(prog, lib, ctypes.c_size_t(len(lib)), b"libdevice") == 0
opts = [o.encode() for o in (sys.argv[3:] or ["-arch=compute_120", "-opt=3"])]
r = nv.nvvmCompileProgram(prog, len(opts), (ctypes.c_char_p * len(opts))(*opts))
n = ctypes.c_size_t(); nv.nvvmGetProgramLogSize(prog, ctypes.byref(n))
log = ctypes.create_string_buffer(n.value); nv.nvvmGetProgramLog(prog, log)
if log.value.strip():
    print(log.value.decode()[:3000])
assert r == 0, f"nvvmCompileProgram: {r}"
nv.nvvmGetCompiledResultSize(prog, ctypes.byref(n))
out = ctypes.create_string_buffer(n.value); nv.nvvmGetCompiledResult(prog, out)
open(sys.argv[2], "wb").write(out.value)
print("PTX:", len(out.value), "bytes")
