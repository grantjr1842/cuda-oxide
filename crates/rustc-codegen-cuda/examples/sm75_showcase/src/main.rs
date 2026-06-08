/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! SM75 Baseline Showcase Example
//!
//! This example demonstrates the features that are supported by the `sm_75` (Turing)
//! virtual architecture baseline in `cuda-oxide`. Because `sm_75` is the oldest supported
//! target in this toolchain, compiling for `sm_75` by default ensures maximum JIT compatibility
//! across all subsequent architectures (Ampere, Ada, Hopper, Blackwell).
//!
//! Features showcased:
//! 1. **Grid, block, thread hierarchy** (Turing baseline)
//! 2. **Warp shuffles** (`warp::shuffle_down_f32`, Turing baseline)
//! 3. **Block-shared memory** (`SharedArray`, Turing baseline)
//! 4. **Ordinary barriers** (`sync_threads`, Turing baseline)
//! 5. **Floating-point atomics** (`DeviceAtomicF32`, Turing baseline)
//! 6. **Console printing** (`gpu_printf`, Turing baseline)
//!
//! Build and run:
//!   cargo oxide run sm75_showcase
//!
//! Force compilation failure with negative gating:
//!   - Trigger AmpereAsync gate (requires sm_80+):
//!     cargo oxide run sm75_showcase --arch sm_75 --features trigger_ampere_gate
//!   - Trigger Tma gate (requires sm_100+):
//!     cargo oxide run sm75_showcase --arch sm_75 --features trigger_tma_gate
//!

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::atomic::{AtomicOrdering, DeviceAtomicF32};
use cuda_device::{SharedArray, cuda_module, kernel, thread, warp};

// =============================================================================
// KERNEL - Compiled to PTX by rustc-codegen-cuda
// =============================================================================

#[cuda_module]
mod kernels {
    use super::*;

    /// Parallel reduction kernel: sums elements of `data` and adds to `global_sum`
    #[kernel]
    pub fn reduction_kernel(
        data: &[f32],
        global_sum: &[f32],
        block_sums_out: *mut f32,
    ) {
        // Shared memory for block-level reduction (Turing baseline)
        static mut SHARED_MEM: SharedArray<f32, 32> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x();
        let gid = thread::index_1d();
        let lane = warp::lane_id();
        let warp = thread::threadIdx_x() / 32;

        let num_elements = data.len();
        let mut val = if gid.get() < num_elements {
            data[gid.get()]
        } else {
            0.0
        };

        // 1. Showcase console printing (Turing baseline)
        if gid.get() == 0 {
            // Print a welcome message from the GPU
            cuda_device::gpu_printf!("Hello from thread {} inside the reduction_kernel!\n", 0);
        }

        // 2. Showcase warp-level reduction using warp shuffles (Turing baseline)
        // Butterfly/down reduction across the warp (32 threads)
        val += warp::shuffle_down_f32(val, 16);
        val += warp::shuffle_down_f32(val, 8);
        val += warp::shuffle_down_f32(val, 4);
        val += warp::shuffle_down_f32(val, 2);
        val += warp::shuffle_down_f32(val, 1);

        // First lane of each warp writes the warp sum to shared memory
        if lane == 0 {
            unsafe {
                SHARED_MEM[warp as usize] = val;
            }
        }

        // 3. Showcase ordinary block synchronization (Turing baseline)
        thread::sync_threads();

        // 4. Showcase gate triggers (to verify negative feature gating)
        #[cfg(feature = "trigger_ampere_gate")]
        {
            // Triggers "AmpereAsync" gate since bar.warp.sync is an Ampere+ feature
            warp::sync_mask(0xFFFFFFFF);
        }

        #[cfg(feature = "trigger_tma_gate")]
        {
            // Triggers "Tma" gate since fence.proxy.async is a Hopper+ feature
            unsafe {
                cuda_device::barrier::fence_proxy_async_shared_cta();
            }
        }

        // Let the first warp sum the warp sums stored in shared memory
        if warp == 0 {
            let mut warp_sum = if lane < (thread::blockDim_x() / 32) {
                unsafe { SHARED_MEM[lane as usize] }
            } else {
                0.0
            };

            // Warp shuffle reduction again
            warp_sum += warp::shuffle_down_f32(warp_sum, 16);
            warp_sum += warp::shuffle_down_f32(warp_sum, 8);
            warp_sum += warp::shuffle_down_f32(warp_sum, 4);
            warp_sum += warp::shuffle_down_f32(warp_sum, 2);
            warp_sum += warp::shuffle_down_f32(warp_sum, 1);

            // Thread 0 adds the block sum to the global sum atomically
            if tid == 0 {
                // Write block sum to debug output slice
                unsafe {
                    *block_sums_out.add(thread::blockIdx_x() as usize) = warp_sum;
                }

                // 5. Showcase floating-point atomics (Turing baseline)
                // Get a reference to the atomic type from the raw pointer
                let global_sum_atomic = unsafe {
                    &*(global_sum.as_ptr() as *const DeviceAtomicF32)
                };
                global_sum_atomic.fetch_add(warp_sum, AtomicOrdering::Relaxed);
            }
        }
    }
}

// =============================================================================
// HOST CODE - Compiled to native x86_64 by LLVM
// =============================================================================

fn main() {
    println!("=== SM75 Showcase (Turing baseline support) ===");

    // Initialize CUDA context
    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    // Verify host compute capability matches or exceeds sm_75
    let (major, minor) = ctx.compute_capability().unwrap();
    println!("Detected host GPU compute capability: {}.{}", major, minor);

    // Setup input data (size = 1024 floats, all 1.0 -> sum should be 1024.0)
    const N: usize = 1024;
    let input_host = vec![1.0f32; N];

    // Allocate device buffers
    let input_dev = DeviceBuffer::from_host(&stream, &input_host).unwrap();
    let debug_block_sums = DeviceBuffer::<f32>::zeroed(&stream, 8).unwrap();
    let global_sum_dev = DeviceBuffer::<f32>::zeroed(&stream, 1).unwrap();

    // Load kernel and launch
    let module = kernels::load(&ctx).expect("Failed to load embedded module");
    println!("Launching reduction_kernel on {} elements...", N);

    module
        .reduction_kernel(
            &stream,
            LaunchConfig {
                grid_dim: (8, 1, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            },
            &input_dev,
            &global_sum_dev,
            debug_block_sums.cu_deviceptr() as *mut f32,
        )
        .expect("Kernel launch failed");

    // Wait for GPU to finish and flush printf buffer
    stream.synchronize().expect("Stream sync failed");

    // Copy results back to host
    let global_sum_result: Vec<f32> = global_sum_dev.to_host_vec(&stream).unwrap();
    let block_sums = debug_block_sums.to_host_vec(&stream).unwrap();

    println!("\nVerification:");
    println!("  Block sums: {:?}", block_sums);
    println!("  Global sum: {} (Expected: {})", global_sum_result[0], N as f32);

    assert!((global_sum_result[0] - N as f32).abs() < 1e-5, "Sum mismatch!");
    println!("✓ Showcase successfully completed!");
}
