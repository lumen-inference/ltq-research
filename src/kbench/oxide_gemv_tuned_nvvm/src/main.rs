// Kernel-language comparison: cuda-oxide (Rust) vs CUDA C++. One-to-one port of the LTQ GEMV kernel from src/cuda/ltq_matvec_bench.cu;
// data, tables, launch configurations and timing are the same as in ../cpp/gemv.cu.
// Prints results as JSON lines, writes the outputs for cross-checking to out/y_rstunednvvm_K{K}_V{VAR}.bin.
//
//   cargo oxide build --arch sm_120 --materialize-cubin && ./target/release/oxide_gemv
#![feature(f16)]

use std::io::Write;
use std::time::Instant;

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use cuda_device::convert::cvt_f32_f16x2_lo;
use cuda_device::{device, kernel, launch_bounds, ptx_asm, thread, warp, DynamicSharedArray};
use cuda_host::cuda_module;

/// float4: 16-byte alignment yields the vector load ld.global.v4.f32.
#[repr(C, align(16))]
#[derive(Clone, Copy)]
pub struct F4 {
    pub a: [f32; 4],
}

#[cuda_module]
mod kernels {
    use super::*;

    #[device]
    pub fn dhash32(i: u64, seed: u32) -> u32 {
        let mut x = i
            .wrapping_mul(0x9E3779B97F4A7C15)
            .wrapping_add((seed as u64).wrapping_mul(0xD1B54A32D192ED03));
        x ^= x >> 31;
        x = x.wrapping_mul(0xBF58476D1CE4E5B9);
        x ^= x >> 29;
        x = x.wrapping_mul(0x94D049BB133111EB);
        x ^= x >> 32;
        x as u32
    }

    #[kernel]
    pub unsafe fn fill_u32(p: *mut u32, n: u64, seed: u32) {
        let stride = (thread::gridDim_x() * thread::blockDim_x()) as u64;
        let mut i = (thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x()) as u64;
        while i < n {
            unsafe { *p.add(i as usize) = dhash32(i, seed) };
            i += stride;
        }
    }

    #[kernel]
    pub unsafe fn fill_half(p: *mut u16, n: u64, seed: u32, lo: f32, hi: f32) {
        let stride = (thread::gridDim_x() * thread::blockDim_x()) as u64;
        let mut i = (thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x()) as u64;
        while i < n {
            // nvcc contracts lo + a * b into an fma; done explicitly here so the data match bit for bit
            let v = ((hi - lo) * (dhash32(i, seed) >> 8) as f32).mul_add(1.0 / 16777216.0, lo);
            unsafe { *p.add(i as usize) = (v as f16).to_bits() };
            i += stride;
        }
    }

    #[device]
    pub unsafe fn lane_unc<const K: u32>(unc: *const u32, unc1: *const u32, tile: usize, l: u32) -> (u32, u32) {
        let ub = K - 1;
        unsafe {
            if ub == 1 {
                (*unc.add(tile * 8 + (l >> 2) as usize) >> (8 * (l & 3)), 0)
            } else if ub == 2 {
                (*unc.add(tile * 16 + (l >> 1) as usize) >> (16 * (l & 1)), 0)
            } else if ub == 3 {
                let u1 = *unc1.add(tile * 8 + (l >> 2) as usize) >> (8 * (l & 3));
                (*unc.add(tile * 16 + (l >> 1) as usize) >> (16 * (l & 1)), u1)
            } else {
                (*unc.add(tile * 32 + l as usize), 0)
            }
        }
    }

    #[device]
    pub fn unc_of<const K: u32>(uw: u32, u1: u32, i: u32) -> u32 {
        let ub = K - 1;
        if ub == 1 {
            (uw >> i) & 1
        } else if ub == 2 {
            (uw >> (2 * i)) & 3
        } else if ub == 3 {
            ((uw >> (2 * i)) & 3) | (((u1 >> i) & 1) << 2)
        } else {
            (uw >> (4 * i)) & 15
        }
    }

    // Loads through the read-only cache (ld.global.nc), which C++ gets from __ldg / const __restrict__.
    #[device]
    #[inline(always)]
    pub unsafe fn ldg_u32(p: *const u32) -> u32 {
        let v: u32;
        unsafe { ptx_asm!("ld.global.nc.u32 %0, [%1];", out("=r") v, in("l") p as u64, options(register_only)) };
        v
    }

    #[device]
    #[inline(always)]
    pub unsafe fn ldg_u16(p: *const u16) -> u32 {
        let v: u32;
        unsafe { ptx_asm!("ld.global.nc.u16 %0, [%1];", out("=r") v, in("l") p as u64, options(register_only)) };
        v
    }

    #[device]
    #[inline(always)]
    pub unsafe fn ldg_f4(p: *const f32) -> [f32; 4] {
        let (a, b, c, d): (f32, f32, f32, f32);
        unsafe {
            ptx_asm!("ld.global.nc.v4.f32 {%0, %1, %2, %3}, [%4];", out("=f") a, out("=f") b, out("=f") c, out("=f") d,
                     in("l") p as u64, options(register_only))
        };
        [a, b, c, d]
    }

    #[device]
    #[inline(always)]
    pub unsafe fn lane_unc_nc<const K: u32>(unc: *const u32, unc1: *const u32, tile: usize, l: u32) -> (u32, u32) {
        let ub = K - 1;
        unsafe {
            if ub == 1 {
                (ldg_u32(unc.add(tile * 8 + (l >> 2) as usize)) >> (8 * (l & 3)), 0)
            } else if ub == 2 {
                (ldg_u32(unc.add(tile * 16 + (l >> 1) as usize)) >> (16 * (l & 1)), 0)
            } else if ub == 3 {
                let u1 = ldg_u32(unc1.add(tile * 8 + (l >> 2) as usize)) >> (8 * (l & 3));
                (ldg_u32(unc.add(tile * 16 + (l >> 1) as usize)) >> (16 * (l & 1)), u1)
            } else {
                (ldg_u32(unc.add(tile * 32 + l as usize)), 0)
            }
        }
    }

    /// R = 2: budget of 64 registers, as nvcc chose. The body is a copy of r4 (kernel attributes must be literals).
    #[kernel]
    #[launch_bounds(256, 4)]
    pub unsafe fn ltq_matvec_r2<const K: u32, const VAR: u32, const NLB: u32>(
        code: *const u32, unc: *const u32, unc1: *const u32, hdr: *const u32, scale: *const u16, x: *const f32, y: *mut f32,
        m: i32, kd: i32, gtab: *const u8, gains: *const f32, wm1: u32, wm2: u32, wm3: u32,
    ) {
        const R: usize = 2;
        let ub = K - 1;
        let smem = DynamicSharedArray::<u8>::get_raw();
        let sg = smem as *mut f32;
        let tid = thread::threadIdx_x();
        let bdim = thread::blockDim_x();
        unsafe {
            if tid < 4 {
                *sg.add(tid as usize) = *gains.add(tid as usize);
            }
            if VAR == 0 {
                let st = smem.add(16) as *mut u16;
                let g = gtab as *const u16;
                let mut i = tid;
                while i < (1024u32 << ub) {
                    *st.add(i as usize) = *g.add(i as usize);
                    i += bdim;
                }
            } else {
                let st = smem.add(16) as *mut f32;
                let g = gtab as *const f32;
                let nent = (1u32 << NLB) << ub;
                let mut i = tid;
                while i < nent * 32 {
                    *st.add(i as usize) = *g.add((i >> 5) as usize);
                    i += bdim;
                }
            }
        }
        thread::sync_threads();
        let tab16 = unsafe { smem.add(16) as *const u16 };
        let tab32 = unsafe { smem.add(16) as *const f32 };
        let l = tid & 31;
        let o = 8 * (l & 3);
        let warps = (thread::gridDim_x() * (bdim >> 5)) as usize;
        let nt = (kd >> 8) as usize;
        let mut row0 = (thread::blockIdx_x() * (bdim >> 5) + (tid >> 5)) as usize * R;
        while row0 < m as usize {
            let mut accs = [0f32; R];
            let mut j = 0usize;
            #[unroll(2)]
            while j < nt {
                let xa = unsafe { ldg_f4(x.add(j * 256 + 8 * l as usize)) };
                let xb = unsafe { ldg_f4(x.add(j * 256 + 8 * l as usize + 4)) };
                let xs = [xa[0], xa[1], xa[2], xa[3], xb[0], xb[1], xb[2], xb[3]];
                #[unroll]
                for r in 0..R {
                    let tile = (row0 + r) * nt + j;
                    let myw = unsafe { ldg_u32(code.add(tile * 8 + (l & 7) as usize)) };
                    let (uw, u1) = unsafe { lane_unc_nc::<K>(unc, unc1, tile, l) };
                    let h = unsafe { ldg_u32(hdr.add(tile)) };
                    let sc = cvt_f32_f16x2_lo(unsafe { ldg_u16(scale.add(tile)) });
                    let cw = warp::shuffle_sync(0xffffffff, myw, l >> 2);
                    let pw = warp::shuffle_sync(0xffffffff, myw, ((l >> 2) + 7) & 7);
                    let win17 = ((((cw as u64) << 32) | pw as u64) >> (23 + o)) as u32;
                    let mut part = 0f32;
                    #[unroll]
                    for i in 0..8u32 {
                        let w10 = (win17 >> i) & 0x3ff;
                        let u = unc_of::<K>(uw, u1, i);
                        let v = if VAR == 0 {
                            cvt_f32_f16x2_lo(unsafe { *tab16.add(((w10 << ub) | u) as usize) } as u32)
                        } else {
                            let mut lb = w10 >> 9;
                            if NLB > 1 {
                                lb |= ((w10 & wm1).count_ones() & 1) << 1;
                            }
                            if NLB > 2 {
                                lb |= ((w10 & wm2).count_ones() & 1) << 2;
                            }
                            if NLB > 3 {
                                lb |= ((w10 & wm3).count_ones() & 1) << 3;
                            }
                            unsafe { *tab32.add(((((lb << ub) | u) << 5) | l) as usize) }
                        };
                        part = v.mul_add(xs[i as usize], part);
                    }
                    let g = unsafe { *sg.add(((h >> (2 * (l >> 1))) & 3) as usize) };
                    accs[r] = part.mul_add(g * sc, accs[r]);
                }
                j += 1;
            }
            #[unroll]
            for r in 0..R {
                let mut acc = accs[r];
                #[unroll]
                for k in 0..5u32 {
                    acc += warp::shuffle_xor_f32_sync(0xffffffff, acc, 16 >> k);
                }
                if l == 0 {
                    unsafe { *y.add(row0 + r) = acc };
                }
            }
            row0 += warps * R;
        }
    }

    /// R = 4: budget of 128 registers.
    #[kernel]
    #[launch_bounds(256, 2)]
    pub unsafe fn ltq_matvec_r4<const K: u32, const VAR: u32, const NLB: u32>(
        code: *const u32, unc: *const u32, unc1: *const u32, hdr: *const u32, scale: *const u16, x: *const f32, y: *mut f32,
        m: i32, kd: i32, gtab: *const u8, gains: *const f32, wm1: u32, wm2: u32, wm3: u32,
    ) {
        const R: usize = 4;
        let ub = K - 1;
        let smem = DynamicSharedArray::<u8>::get_raw();
        let sg = smem as *mut f32;
        let tid = thread::threadIdx_x();
        let bdim = thread::blockDim_x();
        unsafe {
            if tid < 4 {
                *sg.add(tid as usize) = *gains.add(tid as usize);
            }
            if VAR == 0 {
                let st = smem.add(16) as *mut u16;
                let g = gtab as *const u16;
                let mut i = tid;
                while i < (1024u32 << ub) {
                    *st.add(i as usize) = *g.add(i as usize);
                    i += bdim;
                }
            } else {
                let st = smem.add(16) as *mut f32;
                let g = gtab as *const f32;
                let nent = (1u32 << NLB) << ub;
                let mut i = tid;
                while i < nent * 32 {
                    *st.add(i as usize) = *g.add((i >> 5) as usize);
                    i += bdim;
                }
            }
        }
        thread::sync_threads();
        let tab16 = unsafe { smem.add(16) as *const u16 };
        let tab32 = unsafe { smem.add(16) as *const f32 };
        let l = tid & 31;
        let o = 8 * (l & 3);
        let warps = (thread::gridDim_x() * (bdim >> 5)) as usize;
        let nt = (kd >> 8) as usize;
        let mut row0 = (thread::blockIdx_x() * (bdim >> 5) + (tid >> 5)) as usize * R;
        while row0 < m as usize {
            let mut accs = [0f32; R];
            let mut j = 0usize;
            #[unroll(2)]
            while j < nt {
                let xa = unsafe { ldg_f4(x.add(j * 256 + 8 * l as usize)) };
                let xb = unsafe { ldg_f4(x.add(j * 256 + 8 * l as usize + 4)) };
                let xs = [xa[0], xa[1], xa[2], xa[3], xb[0], xb[1], xb[2], xb[3]];
                #[unroll]
                for r in 0..R {
                    let tile = (row0 + r) * nt + j;
                    let myw = unsafe { ldg_u32(code.add(tile * 8 + (l & 7) as usize)) };
                    let (uw, u1) = unsafe { lane_unc_nc::<K>(unc, unc1, tile, l) };
                    let h = unsafe { ldg_u32(hdr.add(tile)) };
                    let sc = cvt_f32_f16x2_lo(unsafe { ldg_u16(scale.add(tile)) });
                    let cw = warp::shuffle_sync(0xffffffff, myw, l >> 2);
                    let pw = warp::shuffle_sync(0xffffffff, myw, ((l >> 2) + 7) & 7);
                    let win17 = ((((cw as u64) << 32) | pw as u64) >> (23 + o)) as u32;
                    let mut part = 0f32;
                    #[unroll]
                    for i in 0..8u32 {
                        let w10 = (win17 >> i) & 0x3ff;
                        let u = unc_of::<K>(uw, u1, i);
                        let v = if VAR == 0 {
                            cvt_f32_f16x2_lo(unsafe { *tab16.add(((w10 << ub) | u) as usize) } as u32)
                        } else {
                            let mut lb = w10 >> 9;
                            if NLB > 1 {
                                lb |= ((w10 & wm1).count_ones() & 1) << 1;
                            }
                            if NLB > 2 {
                                lb |= ((w10 & wm2).count_ones() & 1) << 2;
                            }
                            if NLB > 3 {
                                lb |= ((w10 & wm3).count_ones() & 1) << 3;
                            }
                            unsafe { *tab32.add(((((lb << ub) | u) << 5) | l) as usize) }
                        };
                        part = v.mul_add(xs[i as usize], part);
                    }
                    let g = unsafe { *sg.add(((h >> (2 * (l >> 1))) & 3) as usize) };
                    accs[r] = part.mul_add(g * sc, accs[r]);
                }
                j += 1;
            }
            #[unroll]
            for r in 0..R {
                let mut acc = accs[r];
                #[unroll]
                for k in 0..5u32 {
                    acc += warp::shuffle_xor_f32_sync(0xffffffff, acc, 16 >> k);
                }
                if l == 0 {
                    unsafe { *y.add(row0 + r) = acc };
                }
            }
            row0 += warps * R;
        }
    }

    #[kernel]
    pub unsafe fn empty_kernel(p: *mut i32) {
        if !p.is_null() && thread::threadIdx_x() == 1000 {
            unsafe { *p = 1 };
        }
    }
}

// ---------------- host ----------------

fn hash32(i: u64, seed: u32) -> u32 {
    let mut x = i
        .wrapping_mul(0x9E3779B97F4A7C15)
        .wrapping_add((seed as u64).wrapping_mul(0xD1B54A32D192ED03));
    x ^= x >> 31;
    x = x.wrapping_mul(0xBF58476D1CE4E5B9);
    x ^= x >> 29;
    x = x.wrapping_mul(0x94D049BB133111EB);
    x ^= x >> 32;
    x as u32
}

fn rev9(m: u32) -> u32 {
    (0..9).filter(|j| (m >> j) & 1 == 1).fold(0, |r, j| r | 1 << (8 - j))
}

fn nlb_of(k: u32) -> u32 {
    match k {
        2 => 4,
        3 => 3,
        _ => 2,
    }
}

fn masks_of(k: u32) -> [u32; 3] {
    match k {
        2 => [359, 58, 299],
        3 => [359, 241, 0],
        _ => [359, 0, 0],
    }
}

fn label_host(win: u32, wm: &[u32; 3], nlb: u32) -> u32 {
    let mut l = (win >> 9) & 1;
    for k in 1..nlb {
        l |= ((win & wm[(k - 1) as usize]).count_ones() & 1) << k;
    }
    l
}

type Kernels = kernels::LoadedModule;

struct Tabs {
    comb: DeviceBuffer<u16>,
    vt: DeviceBuffer<f32>,
    gains: DeviceBuffer<f32>,
    h_vt: Vec<f32>,
    h_gains: Vec<f32>,
    wm: [u32; 3],
    nlb: u32,
}

fn tabs(stream: &std::sync::Arc<CudaStream>, k: u32) -> Tabs {
    let nlb = nlb_of(k);
    let hm = masks_of(k);
    let wm = [0, 1, 2].map(|i| if hm[i] != 0 { rev9(hm[i]) } else { 0 });
    let ub = k - 1;
    let nent = (1u32 << nlb) << ub;
    let h_vt: Vec<f32> = (0..nent)
        .map(|e| half::f16::from_f32((((hash32(e as u64, 77) >> 8) as f32 * (1.0 / 16777216.0)) - 0.5) * 4.0).to_f32())
        .collect();
    let mut comb = vec![0u16; 1024 << ub];
    for w in 0..1024u32 {
        for u in 0..(1u32 << ub) {
            comb[((w << ub) | u) as usize] = half::f16::from_f32(h_vt[((label_host(w, &wm, nlb) << ub) | u) as usize]).to_bits();
        }
    }
    let h_gains = vec![0.7f32, 0.9, 1.1, 1.3];
    Tabs {
        comb: DeviceBuffer::from_host(stream, &comb).unwrap(),
        vt: DeviceBuffer::from_host(stream, &h_vt).unwrap(),
        gains: DeviceBuffer::from_host(stream, &h_gains).unwrap(),
        h_vt,
        h_gains,
        wm,
        nlb,
    }
}

struct Buf {
    code: DeviceBuffer<u32>,
    unc: DeviceBuffer<u32>,
    unc1: Option<DeviceBuffer<u32>>,
    hdr: DeviceBuffer<u32>,
    scale: DeviceBuffer<u16>,
    ntile: usize,
    k: u32,
}

impl Buf {
    fn bytes(&self) -> usize {
        self.ntile * (32 + 32 * (self.k as usize - 1) + 4 + 2)
    }
}

fn cfg(grid: u32, smem: u32) -> LaunchConfig {
    LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: smem }
}

fn fill(m: &Kernels, s: &std::sync::Arc<CudaStream>, b: &DeviceBuffer<u32>, n: usize, seed: u32) {
    unsafe { m.fill_u32(s, cfg(1024, 0), b.cu_deviceptr() as *mut u32, n as u64, seed).unwrap() };
}

fn alloc(m: &Kernels, s: &std::sync::Arc<CudaStream>, k: u32, ntile: usize, seed: u32) -> Buf {
    let ub = k - 1;
    let uw = if ub == 3 { 16 } else { 8 * ub as usize };
    // like cudaMalloc + fill in C++: memory is not zeroed, the fill_* kernels fill it
    let code = unsafe { DeviceBuffer::<u32>::uninitialized_async(s, ntile * 8).unwrap() };
    let unc = unsafe { DeviceBuffer::<u32>::uninitialized_async(s, ntile * uw).unwrap() };
    let hdr = unsafe { DeviceBuffer::<u32>::uninitialized_async(s, ntile).unwrap() };
    let scale = unsafe { DeviceBuffer::<u16>::uninitialized_async(s, ntile).unwrap() };
    fill(m, s, &code, ntile * 8, seed);
    fill(m, s, &unc, ntile * uw, seed + 1);
    fill(m, s, &hdr, ntile, seed + 2);
    unsafe {
        m.fill_half(s, cfg(1024, 0), scale.cu_deviceptr() as *mut u16, ntile as u64, seed + 3, 0.5, 1.5).unwrap()
    };
    let unc1 = (ub == 3).then(|| {
        let b = unsafe { DeviceBuffer::<u32>::uninitialized_async(s, ntile * 8).unwrap() };
        fill(m, s, &b, ntile * 8, seed + 4);
        b
    });
    Buf { code, unc, unc1, hdr, scale, ntile, k }
}

fn smem_of(k: u32, var: u32) -> u32 {
    let ub = k - 1;
    let nent = (1u32 << nlb_of(k)) << ub;
    16 + if var == 0 { (1024u32 << ub) * 2 } else { nent * 32 * 4 }
}

/// Dispatch on R: R = 2 and R = 4 have different launch_bounds, so they are two kernels.
macro_rules! call_r {
    ($R:literal, $m:expr, $K:literal, $VAR:literal, $N:literal, $($a:expr),* $(,)?) => {
        if $R == 2 { $m.ltq_matvec_r2::<$K, $VAR, $N>($($a),*) } else { $m.ltq_matvec_r4::<$K, $VAR, $N>($($a),*) }
    };
}

/// Launch of a specific specialization: kernel const arguments must be literals, hence the macro.
macro_rules! launch {
    ($m:expr, $s:expr, $b:expr, $t:expr, $x:expr, $y:expr, $mrows:expr, $grid:expr, $K:literal, $VAR:literal, $R:literal) => {{
        let nlb = nlb_of($K);
        let gtab = if $VAR == 0 { $t.comb.cu_deviceptr() } else { $t.vt.cu_deviceptr() };
        let unc1 = $b.unc1.as_ref().map_or(0, |u| u.cu_deviceptr());
        let c = cfg($grid, smem_of($K, $VAR));
        unsafe {
            match nlb {
                4 => call_r!($R, $m, $K, $VAR, 4, 
                    $s, c, $b.code.cu_deviceptr() as _, $b.unc.cu_deviceptr() as _, unc1 as _, $b.hdr.cu_deviceptr() as _,
                    $b.scale.cu_deviceptr() as _, $x as _, $y as _, $mrows, KD, gtab as _, $t.gains.cu_deviceptr() as _,
                    $t.wm[0], $t.wm[1], $t.wm[2]),
                3 => call_r!($R, $m, $K, $VAR, 3, 
                    $s, c, $b.code.cu_deviceptr() as _, $b.unc.cu_deviceptr() as _, unc1 as _, $b.hdr.cu_deviceptr() as _,
                    $b.scale.cu_deviceptr() as _, $x as _, $y as _, $mrows, KD, gtab as _, $t.gains.cu_deviceptr() as _,
                    $t.wm[0], $t.wm[1], $t.wm[2]),
                _ => call_r!($R, $m, $K, $VAR, 2, 
                    $s, c, $b.code.cu_deviceptr() as _, $b.unc.cu_deviceptr() as _, unc1 as _, $b.hdr.cu_deviceptr() as _,
                    $b.scale.cu_deviceptr() as _, $x as _, $y as _, $mrows, KD, gtab as _, $t.gains.cu_deviceptr() as _,
                    $t.wm[0], $t.wm[1], $t.wm[2]),
            }
            .unwrap()
        }
    }};
}

macro_rules! func {
    ($m:expr, $K:literal, $VAR:literal, $R:literal) => {{
        let name = match nlb_of($K) {
            4 => if $R == 2 { kernels::ltq_matvec_r2_ptx_name::<$K, $VAR, 4>() } else { kernels::ltq_matvec_r4_ptx_name::<$K, $VAR, 4>() },
            3 => if $R == 2 { kernels::ltq_matvec_r2_ptx_name::<$K, $VAR, 3>() } else { kernels::ltq_matvec_r4_ptx_name::<$K, $VAR, 3>() },
            _ => if $R == 2 { kernels::ltq_matvec_r2_ptx_name::<$K, $VAR, 2>() } else { kernels::ltq_matvec_r4_ptx_name::<$K, $VAR, 2>() },
        };
        $m.as_cuda_module().load_function(name).unwrap()
    }};
}

const KD: i32 = 8192;

struct Ctx {
    m: Kernels,
    s: std::sync::Arc<CudaStream>,
    ctx: std::sync::Arc<CudaContext>,
    x: DeviceBuffer<f32>,
    y: DeviceBuffer<f32>,
    hx: Vec<f32>,
    sms: u32,
}

fn time_ms(c: &Ctx, iters: usize, mut launch: impl FnMut()) -> f32 {
    let a = c.ctx.new_event(Some(0)).unwrap();
    let b = c.ctx.new_event(Some(0)).unwrap();
    for _ in 0..3 {
        launch();
    }
    a.record(&c.s).unwrap();
    for _ in 0..iters {
        launch();
    }
    b.record(&c.s).unwrap();
    b.synchronize().unwrap();
    a.elapsed_ms(&b).unwrap() / iters as f32
}

fn ref_row(b: &Buf, t: &Tabs, hx: &[f32], row: usize, h: &(Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>, Vec<u16>)) -> f64 {
    let (code, unc, unc1, hdr, scale) = h;
    let k = b.k;
    let ub = k - 1;
    let nt = KD as usize / 256;
    let mut acc = 0f64;
    for j in 0..nt {
        let tile = row * nt + j;
        let cbit = |p: i32| {
            let p = (p & 255) as usize;
            (code[tile * 8 + (p >> 5)] >> (p & 31)) & 1
        };
        for p in 0..256usize {
            let mut w = 0u32;
            for bb in 0..10 {
                w |= cbit(p as i32 - 9 + bb) << bb;
            }
            let u = match ub {
                1 => (unc[tile * 8 + (p >> 5)] >> (p & 31)) & 1,
                2 => (unc[tile * 16 + (p >> 4)] >> (2 * (p & 15))) & 3,
                3 => ((unc[tile * 16 + (p >> 4)] >> (2 * (p & 15))) & 3) | (((unc1[tile * 8 + (p >> 5)] >> (p & 31)) & 1) << 2),
                _ => (unc[tile * 32 + (p >> 3)] >> (4 * (p & 7))) & 15,
            };
            let v = t.h_vt[((label_host(w, &t.wm, t.nlb) << ub) | u) as usize];
            let g = t.h_gains[((hdr[tile] >> (2 * (p >> 4))) & 3) as usize] * half::f16::from_bits(scale[tile]).to_f32();
            acc += v as f64 * g as f64 * hx[j * 256 + p] as f64;
        }
    }
    acc
}

macro_rules! check {
    ($c:expr, $t:expr, $K:literal, $VAR:literal) => {{
        let c: &Ctx = $c;
        let m = 64usize;
        let b = alloc(&c.m, &c.s, $K, m * KD as usize / 256, 1000 + $K);
        let dy = DeviceBuffer::<f32>::zeroed(&c.s, m).unwrap();
        launch!(c.m, &c.s, b, $t, c.x.cu_deviceptr(), dy.cu_deviceptr(), m as i32, 8, $K, $VAR, 4);
        let y = dy.to_host_vec(&c.s).unwrap();
        let h = (
            b.code.to_host_vec(&c.s).unwrap(),
            b.unc.to_host_vec(&c.s).unwrap(),
            b.unc1.as_ref().map_or(vec![], |u| u.to_host_vec(&c.s).unwrap()),
            b.hdr.to_host_vec(&c.s).unwrap(),
            b.scale.to_host_vec(&c.s).unwrap(),
        );
        let mut maxrel = 0f64;
        for r in 0..m {
            let rf = ref_row(&b, $t, &c.hx, r, &h);
            maxrel = maxrel.max((y[r] as f64 - rf).abs() / (rf.abs() + 1e-3 * (KD as f64).sqrt()));
        }
        let bytes: Vec<u8> = y.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(format!("out/y_rstunednvvm_K{}_V{}.bin", $K, $VAR), bytes).unwrap();
        println!("{{\"lang\":\"rs_tuned_nvvm\",\"kind\":\"check\",\"K\":{},\"VAR\":{},\"maxrel_vs_cpu\":{:.3e}}}", $K, $VAR, maxrel);
    }};
}

macro_rules! bench {
    ($c:expr, $t:expr, $tb:expr, $mode:expr, $K:literal, $VAR:literal, $R:literal) => {{
        let c: &Ctx = $c;
        let bpt = 32 + 32 * ($K - 1) + 6;
        let m = (($tb / (bpt * (KD as usize / 256))) & !7) as i32;
        let b = alloc(&c.m, &c.s, $K, m as usize * KD as usize / 256, 7 + $K);
        let f = func!(c.m, $K, $VAR, $R);
        let grid = f.max_active_blocks_per_multiprocessor(256, smem_of($K, $VAR)).unwrap() * c.sms;
        let regs = f.num_registers().unwrap();
        let local = f.local_size_bytes().unwrap();
        let iters = if $tb > (256usize << 20) { 20 } else { 200 };
        let ms = time_ms(c, iters, || launch!(c.m, &c.s, b, $t, c.x.cu_deviceptr(), c.y.cu_deviceptr(), m, grid, $K, $VAR, $R));
        println!(
            "{{\"lang\":\"rs_tuned_nvvm\",\"kind\":\"gemv\",\"mode\":\"{}\",\"K\":{},\"VAR\":{},\"R\":{},\"grid\":{},\"regs\":{},\"local_bytes\":{},\"ms\":{:.5},\"GBps\":{:.1},\"Gw_s\":{:.1}}}",
            $mode, $K, $VAR, $R, grid, regs, local, ms, b.bytes() as f64 / ms as f64 / 1e6, m as f64 * KD as f64 / ms as f64 / 1e6
        );
        std::io::stdout().flush().unwrap();
    }};
}

macro_rules! bench_k {
    ($c:expr, $t:expr, $tb:expr, $mode:expr, $K:literal) => {{
        bench!($c, $t, $tb, $mode, $K, 0, 2);
        bench!($c, $t, $tb, $mode, $K, 0, 4);
        bench!($c, $t, $tb, $mode, $K, 1, 2);
        bench!($c, $t, $tb, $mode, $K, 1, 4);
    }};
}

fn main() {
    std::fs::create_dir_all("out").unwrap();
    let ctx = CudaContext::new(0).unwrap();
    let s = ctx.default_stream();
    // KB_PTX: PTX built by libNVVM from the dumped NVVM IR (--emit-nvvm-ir does not embed it in the module)
    let m = match std::env::var("KB_PTX") {
        Ok(p) => unsafe { kernels::from_module(ctx.load_module_from_file(&p).unwrap()).unwrap() },
        Err(_) => kernels::load(&ctx).unwrap(),
    };
    let sms = ctx.multiprocessor_count().unwrap();
    let mut l2i = 0i32;
    unsafe {
        cuda_core::sys::cuDeviceGetAttribute(&mut l2i, cuda_core::sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, 0);
    }
    let l2 = l2i as usize;
    println!("{{\"lang\":\"rs_tuned_nvvm\",\"kind\":\"device\",\"sms\":{},\"l2_mb\":{}}}", sms, l2 >> 20);

    let hx: Vec<f32> = (0..KD as u64).map(|i| ((hash32(i, 3) >> 8) as f32 * (1.0 / 16777216.0)) - 0.5).collect();
    let x = DeviceBuffer::from_host(&s, &hx).unwrap();
    let y = DeviceBuffer::<f32>::zeroed(&s, 16 << 20).unwrap();
    let c = Ctx { m, s, ctx, x, y, hx, sms };
    let (t2, t3, t4, t5) = (tabs(&c.s, 2), tabs(&c.s, 3), tabs(&c.s, 4), tabs(&c.s, 5));

    check!(&c, &t2, 2, 0);
    check!(&c, &t2, 2, 1);
    check!(&c, &t3, 3, 0);
    check!(&c, &t3, 3, 1);
    check!(&c, &t4, 4, 0);
    check!(&c, &t4, 4, 1);
    check!(&c, &t5, 5, 0);
    check!(&c, &t5, 5, 1);

    {
        // launch overhead: enqueueing and executing N empty kernels
        let n = 20000;
        for _ in 0..100 {
            unsafe { c.m.empty_kernel(&c.s, LaunchConfig { grid_dim: (1, 1, 1), block_dim: (32, 1, 1), shared_mem_bytes: 0 }, std::ptr::null_mut()).unwrap() };
        }
        c.s.synchronize().unwrap();
        let t0 = Instant::now();
        for _ in 0..n {
            unsafe { c.m.empty_kernel(&c.s, LaunchConfig { grid_dim: (1, 1, 1), block_dim: (32, 1, 1), shared_mem_bytes: 0 }, std::ptr::null_mut()).unwrap() };
        }
        let t1 = Instant::now();
        c.s.synchronize().unwrap();
        let t2 = Instant::now();
        println!(
            "{{\"lang\":\"rs_tuned_nvvm\",\"kind\":\"launch\",\"enqueue_us\":{:.3},\"total_us\":{:.3}}}",
            (t1 - t0).as_secs_f64() * 1e6 / n as f64,
            (t2 - t0).as_secs_f64() * 1e6 / n as f64
        );
    }

    let dram = 2048usize << 20;
    let l2b = (l2 as f64 * 0.4) as usize;
    for (tb, mode) in [(dram, "DRAM"), (l2b, "L2")] {
        bench_k!(&c, &t2, tb, mode, 2);
        bench_k!(&c, &t3, tb, mode, 3);
        bench_k!(&c, &t4, tb, mode, 4);
        bench_k!(&c, &t5, tb, mode, 5);
    }
}
