use std::sync::Arc;

use gpgpu::{BindGroupLayoutBuilder, BufOps};

// Framework is required to be static because of std::thread::spawn lifetime requirements.
// By using crossbeam ScopedThreads this could be avoided.
lazy_static::lazy_static! {
    static ref FW: gpgpu::Framework =
        gpgpu::Framework::default().set_bind_group_layouts(vec![
            BindGroupLayoutBuilder::new()
                .add_buffer(gpgpu::GpuBufferUsage::ReadOnly)
                .add_buffer(gpgpu::GpuBufferUsage::ReadOnly)
                .add_buffer(gpgpu::GpuBufferUsage::ReadWrite),
        ]);
}

fn main() {
    let shader = Arc::new(
        gpgpu::Shader::from_wgsl_file(&FW, "examples/parallel-compute/shader.wgsl").unwrap(),
    );

    let threading = 4; // Threading level
    let size = 32000; // Must be multiple of 32

    let cpu_data = (0..size).into_iter().collect::<Vec<u32>>();
    let shader_input_buffer = Arc::new(gpgpu::GpuBuffer::from_slice(&FW, &cpu_data)); // Data shared across threads shader invocations

    let mut handles = Vec::with_capacity(threading);
    for _ in 0..threading {
        let local_shader = shader.clone();
        let local_shader_input_buffer = shader_input_buffer.clone();

        // Threads spawn
        let handle = std::thread::spawn(move || {
            // Current thread GPU objects
            let local_cpu_data = (0..size).into_iter().collect::<Vec<u32>>();
            let local_input_buffer = gpgpu::GpuBuffer::from_slice(&FW, &local_cpu_data);
            let local_output_buffer = gpgpu::GpuBuffer::<u32>::with_capacity(&FW, size as u64);

            let desc = gpgpu::DescriptorSet::new(0)
                .bind_buffer(&local_shader_input_buffer)
                .bind_buffer(&local_input_buffer)
                .bind_buffer(&local_output_buffer);
            let program = gpgpu::Program::new(&local_shader, "main").add_descriptor_set(desc);

            gpgpu::Kernel::new(&FW, program).enqueue(size / 32, 1, 1);

            local_output_buffer.read_vec_blocking().unwrap()
        });

        handles.push(handle);
    }

    // Join threads
    for handle in handles {
        let output = handle.join().unwrap();

        for (idx, a) in cpu_data.iter().enumerate() {
            let cpu_mult = a.pow(2);
            let gpu_mult = output[idx];

            assert_eq!(cpu_mult, gpu_mult);
        }
    }
}
