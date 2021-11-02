use std::sync::Arc;

lazy_static::lazy_static! {
    static ref FW: gpgpu::Framework = gpgpu::Framework::default();
}

fn main() {
    let shader = Arc::new(
        gpgpu::utils::shader::from_wgsl_file(&FW, "examples/parallel-compute/shader.wgsl").unwrap(),
    );

    let threading = 4;
    let size = 32000; // Must be multiple of 32

    let cpu_data = (0..size).into_iter().collect::<Vec<u32>>();
    let shader_input_buffer = Arc::new(gpgpu::GpuBuffer::from_slice(&FW, &cpu_data));

    let mut handles = Vec::with_capacity(threading);
    for _ in 0..threading {
        let local_shader = shader.clone();
        let local_shader_input_buffer = shader_input_buffer.clone();

        let handle = std::thread::spawn(move || {
            let local_cpu_data = (0..size).into_iter().collect::<Vec<u32>>();
            let local_input_buffer = gpgpu::GpuBuffer::from_slice(&FW, &local_cpu_data);
            let local_output_buffer = gpgpu::GpuBuffer::<u32>::new(&FW, size as usize);

            let binds = gpgpu::DescriptorSet::default()
                .bind_buffer(&local_shader_input_buffer, gpgpu::GpuBufferUsage::ReadOnly)
                .bind_buffer(&local_input_buffer, gpgpu::GpuBufferUsage::ReadOnly)
                .bind_buffer(&local_output_buffer, gpgpu::GpuBufferUsage::ReadWrite);

            let kernel = &FW
                .create_kernel_builder(&local_shader, "main")
                .add_descriptor_set(binds)
                .build();

            kernel.enqueue(size / 32, 1, 1);
            local_output_buffer.read().unwrap()
        });

        handles.push(handle);
    }

    for handle in handles {
        let output = handle.join().unwrap();

        for (idx, a) in cpu_data.iter().enumerate() {
            let cpu_mult = a.pow(2);
            let gpu_mult = output[idx];

            assert_eq!(cpu_mult, gpu_mult);
        }
    }
}
