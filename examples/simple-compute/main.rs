use gpgpu::BufOps;

// Simple compute example that multiplies 2 vectors A and B, storing the result in a vector C.
fn main() {
    let fw = gpgpu::Framework::default(); // Framework initialization.

    let shader = gpgpu::Shader::from_wgsl_file(&fw, "examples/simple-compute/shader.wgsl").unwrap(); // Shader loading.

    // Creation of a kernel. This represents the current function that will be executed
    let kernel = gpgpu::Kernel::new(
        &fw,
        &shader,
        "main", // The kernel needs the name function to be executed
        // We have to tell the GPU how the layout of the data with SetLayout
        // We create SetLayout using the new_set_layout macro
        vec![gpgpu::new_set_layout!(
            0: Buffer(gpgpu::GpuBufferUsage::ReadOnly),
            1: Buffer(gpgpu::GpuBufferUsage::ReadOnly),
            2: Buffer(gpgpu::GpuBufferUsage::ReadWrite)
        )],
    );

    let size = 10000; // Size of the vectors

    let data_a = (0..size).into_iter().collect::<Vec<u32>>(); // Vector A data. 0, 1, 2, ..., 9999 (size - 1).
    let data_b = (0..size).into_iter().rev().collect::<Vec<u32>>(); // Vector B data. 9999 (size - 1), 9998, ..., 0.

    // Allocation of new vectors on the GPU
    let gpu_vec_a = gpgpu::GpuBuffer::from_slice(&fw, &data_a); // Input vector A.
    let gpu_vec_b = gpgpu::GpuBuffer::from_slice(&fw, &data_b); // Input vector B.
    let gpu_vec_c = gpgpu::GpuBuffer::with_capacity(&fw, size as u64); // Output vector C. Empty.

    let bindings = gpgpu::SetBindings::default()
        .add_buffer(0, &gpu_vec_a)
        .add_buffer(1, &gpu_vec_b)
        .add_buffer(2, &gpu_vec_c);

    // Execution of the kernel. It needs 3 dimmensions, x y and z.
    // Since we are using single-dim vectors, only x is required.
    kernel.run(&fw, vec![bindings], size as u32, 1, 1);

    // After the kernel execution, we can read the results from the GPU.
    let gpu_result = gpu_vec_c.read_vec_blocking().unwrap();

    // We test that the results are correct.
    for (idx, (a, b)) in data_a.into_iter().zip(data_b).enumerate() {
        let cpu_mult = a * b;
        let gpu_mult = gpu_result[idx];

        assert_eq!(cpu_mult, gpu_mult);
    }
}
