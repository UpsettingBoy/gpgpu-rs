// Simple compute example that multiplies 2 vectors A and B, storing the result in a vector C.
fn main() {
    let fw = gpgpu::Framework::default(); // Framework initialization.

    let shader = gpgpu::Shader::from_wgsl_file(&fw, "examples/simple-compute/mult.wgsl").unwrap(); // Shader loading.

    let size = 10000; // Size of the vectors

    let data_a = (0..size).into_iter().collect::<Vec<u32>>(); // Vector A data. 0, 1, 2, ..., 9999 (size - 1).
    let data_b = (0..size).into_iter().rev().collect::<Vec<u32>>(); // Vector B data. 9999 (size - 1), 9998, ..., 0.

    // Allocation of new vectors on the GPU
    let gpu_vec_a = gpgpu::GpuBuffer::from_slice(&fw, &data_a); // Input vector A.
    let gpu_vec_b = gpgpu::GpuBuffer::from_slice(&fw, &data_b); // Input vector B.
    let gpu_vec_c = gpgpu::GpuBuffer::new(&fw, size as usize); // Output vector C. Empty.

    // We have to tell the GPU how the data is sent. Take a look at the shader (mult.wgsl).
    // The boolean indicates wether the vector is read-only or not.
    let bindings = gpgpu::DescriptorSet::default() // Group 0
        .bind_buffer(&gpu_vec_a, gpgpu::GpuBufferUsage::ReadOnly) // Binding 0
        .bind_buffer(&gpu_vec_b, gpgpu::GpuBufferUsage::ReadOnly) // Binding 1
        .bind_buffer(&gpu_vec_c, gpgpu::GpuBufferUsage::ReadWrite); // Binding 2. read_write in shader. No write-only yet.

    // Match a shader entry point with its descriptor (the bindings).
    // A program represents a function on a GPU with an already set of inputs and outputs following a layout (the variable `bindings` above).
    let program = gpgpu::Program::new(&shader, "main").add_descriptor_set(bindings);

    // Creation of a kernel. This represents the `program` function and its `enqueuing` parameters,
    let kernel = gpgpu::Kernel::new(&fw, program);

    // Execution of the kernel. It needs 3 dimmensions, x y and z.
    // Since we are using single-dim vectors, only x is required.
    kernel.enqueue(size as u32, 1, 1);

    // After the kernel execution, we can read the results from the GPU.
    let gpu_result = gpu_vec_c.read().unwrap();

    // We test that the results are correct.
    for (idx, (a, b)) in data_a.into_iter().zip(data_b).enumerate() {
        let cpu_mult = a * b;
        let gpu_mult = gpu_result[idx];

        assert_eq!(cpu_mult, gpu_mult);
    }
}
