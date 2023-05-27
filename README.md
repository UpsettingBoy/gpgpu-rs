# gpgpu
![GitHub Workflow Status (branch)](https://img.shields.io/github/actions/workflow/status/UpsettingBoy/gpgpu-rs/ci.yml?branch=dev&label=Actions&style=flat-square)
![Crates.io](https://img.shields.io/crates/l/gpgpu?style=flat-square)
![Crates.io](https://img.shields.io/crates/v/gpgpu?style=flat-square)
[![docs.rs](https://img.shields.io/static/v1?label=docs.rs&message=read&color=brightgreen&style=flat-square)](https://docs.rs/gpgpu)

<!-- cargo-sync-readme start -->

An experimental async GPU compute library based on [`wgpu`](https://github.com/gfx-rs/wgpu).
It is meant to be used alongside `wgpu` if desired.

To start using `gpgpu`, just create a [`Framework`](https://docs.rs/gpgpu/latest/gpgpu/struct.Framework.html) instance
and follow the [examples](https://github.com/UpsettingBoy/gpgpu-rs/tree/dev/examples) in the main repository.

# Example
Small program that multiplies 2 vectors A and B; and stores the
result in another vector C.
## Rust program
```rust
    use gpgpu::*;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Framework initialization
    let fw = Framework::default();

    // Original CPU data
    let cpu_data = (0..10000).into_iter().collect::<Vec<u32>>();

    // GPU buffer creation
    let buf_a = GpuBuffer::from_slice(&fw, &cpu_data);       // Input
    let buf_b = GpuBuffer::from_slice(&fw, &cpu_data);       // Input
    let buf_c = GpuBuffer::<u32>::with_capacity(&fw, cpu_data.len() as u64);  // Output

    // Shader load from SPIR-V binary file
    let shader = Shader::from_spirv_file(&fw, "<SPIR-V shader path>")?;
    //  or from a WGSL source file
    let shader = Shader::from_wgsl_file(&fw, "<WGSL shader path>")?;    

    // Descriptor set and program creation
    let desc = DescriptorSet::default()
        .bind_buffer(&buf_a, GpuBufferUsage::ReadOnly)
        .bind_buffer(&buf_b, GpuBufferUsage::ReadOnly)
        .bind_buffer(&buf_c, GpuBufferUsage::ReadWrite);
    let program = Program::new(&shader, "main").add_descriptor_set(desc); // Entry point

    // Kernel creation and enqueuing
    Kernel::new(&fw, program).enqueue(cpu_data.len() as u32, 1, 1); // Enqueuing, not very optimus 😅

    let output = buf_c.read_vec_blocking()?;                        // Read back C from GPU
    for (a, b) in cpu_data.into_iter().zip(output) {
        assert_eq!(a.pow(2), b);
    }

    Ok(())
}
```

## Shader program
The shader is written in [WGSL](https://gpuweb.github.io/gpuweb/wgsl/)
```rust
// Vector type definition. Used for both input and output
struct Vector {
    data: array<u32>,
}

// A, B and C vectors
@group(0) @binding(0) var<storage, read>  a: Vector;
@group(0) @binding(1) var<storage, read>  b: Vector;
@group(0) @binding(2) var<storage, read_write> c: Vector;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    c.data[global_id.x] = a.data[global_id.x] * b.data[global_id.x];
}
```


<!-- cargo-sync-readme end -->