/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use cuda_core::{CudaContext, DeviceBuffer};

#[test]
fn device_buffer_supports_empty_allocations() {
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    let stream = ctx.new_stream().expect("failed to create CUDA stream");

    let device =
        DeviceBuffer::<u32>::zeroed(&stream, 0).expect("failed to create empty device buffer");

    assert_eq!(device.len(), 0);
    assert_eq!(device.num_bytes(), 0);
    assert!(device.is_empty());
}

#[test]
fn device_buffer_from_host_supports_empty_input() {
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    let stream = ctx.new_stream().expect("failed to create CUDA stream");

    let device = DeviceBuffer::<u32>::from_host(&stream, &[])
        .expect("failed to create empty device buffer from host");

    assert_eq!(device.len(), 0);
    assert_eq!(device.num_bytes(), 0);
    assert!(device.is_empty());
}
