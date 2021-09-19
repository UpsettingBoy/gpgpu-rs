# gpgpu

<!-- cargo-sync-readme start -->

A simple GPU compute library based on [`wgpu`](https://github.com/gfx-rs/wgpu).
It is meant to be used alongside `wgpu` if desired.

Right now `gpgpu` uses some of `wgpu`'s type on its public API.
It may be removed in the future.

To start using `gpgpu`, just create a [`Framework`](https://docs.rs/gpgpu/latest/gpgpu/struct.Framework.html) instance
and follow the examples in the main repository.

# Example
Small program that multiplies 2 vectors A and B; and stores the
result in another vector C.
## Rust program
```rust
 use gpgpu::*;
 use gpgpu::wgpu;

 fn main() -> GpuResult<()> {
    let fw = Framework::default();
    
    // Original CPU data
    let cpu_data = (0..10000).into_iter().collect::<Vec<u32>>();

    // GPU buffer creation
    let buf_a = fw.create_buffer_from_slice(&cpu_data);     // Input
    let buf_b = fw.create_buffer_from_slice(&cpu_data);     // Input
    let buf_c = fw.create_buffer::<u32>(cpu_data.len());    // Output

    // Shader load from SPIR-V file
    let shader_module = utils::shader::from_spirv_file(&fw, "<shader path>")?;

    // Descriptor set creation
    let desc_set = DescriptorSet::default()
        .bind_storage_buffer(&buf_a, true)      // Read-only
        .bind_storage_buffer(&buf_b, true)      // Read-only
        .bind_storage_buffer(&buf_c, false);    // Read-write
    
    // Kernel creation and enqueuing
    fw.create_kernel_builder(&shader_module, "main")   // Entry point
        .add_descriptor_set(desc_set)                      
        .build()
        .enqueue(cpu_data.len() as u32, 1, 1);         // Enqueuing, not very optimus 😅

    let output = buf_c.read()?;                        // Read back C from GPU
    for (a, b) in cpu_data.into_iter().zip(output) {
        assert_eq!(a.pow(2), b);
    }

    Ok(())
}
```

## Shader program
The shader is writen in [WGSL](https://gpuweb.github.io/gpuweb/wgsl/)
```rust
// Matrix type definition. Used for both input and output
[[block]]
struct Vector {
    data: [[stride(4)]] array<u32>;
};

// A, B and C vectors
[[group(0), binding(0)]] var<storage, read>  a: Vector;
[[group(0), binding(1)]] var<storage, read>  b: Vector;
[[group(0), binding(2)]] var<storage, read_write> c: Vector;

[[stage(compute), workgroup_size(1)]]
fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
    c.data[global_id.x] = a.data[global_id.x] * b.data[global_id.x];
}
```


<!-- cargo-sync-readme end -->