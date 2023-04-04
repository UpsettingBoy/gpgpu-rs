use gpgpu::BufOps;

// Simple compute example that multiplies 2 square arrays (matrixes)  A and B, storing the result in another array C using ndarray.
fn main() {
    let fw = gpgpu::Framework::default();

    let shader = gpgpu::Shader::from_wgsl_file(&fw, "examples/ndarray/shader.wgsl").unwrap();

    let dims = (3200, 3200); // X and Y dimensions. Must be multiple of the enqueuing dimensions

    let array_src = ndarray::Array::<i32, _>::ones(dims) * 2;
    let src_view = array_src.view();

    let gpu_arrays_len = gpgpu::GpuUniformBuffer::from_slice(&fw, &[dims.0, dims.1]); // Send the ndarray dimensions

    let gpu_array_a = gpgpu::GpuArray::from_array(&fw, src_view).unwrap(); // Array A
    let gpu_array_b = gpgpu::GpuArray::from_array(&fw, src_view).unwrap(); // Array B
    let gpu_array_c = gpgpu::GpuArray::from_array(&fw, ndarray::Array::zeros(dims).view()).unwrap(); // Array C: result storage

    let desc_0 = gpgpu::DescriptorSet::default().bind_uniform_buffer(&gpu_arrays_len, 0); // Descriptor set of arrays dimensions.

    let desc_1 = gpgpu::DescriptorSet::default() // Descriptor set of all the arrays
        .bind_array(&gpu_array_a, gpgpu::GpuBufferUsage::ReadOnly, 0)
        .bind_array(&gpu_array_b, gpgpu::GpuBufferUsage::ReadOnly, 1)
        .bind_array(&gpu_array_c, gpgpu::GpuBufferUsage::ReadWrite, 2);

    let program = gpgpu::Program::new(&shader, "main_fn_2")
        .add_descriptor_set(desc_0)
        .add_descriptor_set(desc_1);

    gpgpu::Kernel::new(&fw, program)
        // .enqueue((dims.0 * dims.1) as u32, 1, 1); // Kernel main_fn 1. Enqueuing in a single dimension
        .enqueue(dims.0 as u32 / 32, dims.1 as u32 / 32, 1); // Kernel main_fn 2. Enqueuing in x and y dimensions (array dimensions are needed)

    let array_output = gpu_array_c.read_blocking().unwrap();

    for (cpu, gpu) in array_src.iter().zip(array_output) {
        assert_eq!(2 * cpu, gpu)
    }
}
