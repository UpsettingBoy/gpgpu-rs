/// Contains some helper functions to create [`wgpu::ShaderModule`](wgpu::ShaderModule) from SPIR-V files or bytes.
pub mod shader {
    use std::{borrow::Cow, path::Path};

    use wgpu::ShaderModule;

    use crate::{Framework, GpuResult};

    /// Creates a [`wgpu::ShaderModule`](wgpu::ShaderModule) instance from a SPIR-V file.
    pub fn from_spirv_file(fw: &Framework, path: impl AsRef<Path>) -> GpuResult<ShaderModule> {
        let bytes = std::fs::read(&path)?;
        let shader_name = path.as_ref().to_str();

        Ok(from_spirv_bytes(fw, &bytes, shader_name))
    }

    /// Creates a [`wgpu::ShaderModule`](wgpu::ShaderModule) instance from SPIR-V bytes
    /// and an optional `shader_name`.
    pub fn from_spirv_bytes(
        fw: &Framework,
        bytes: &[u8],
        shader_name: Option<&str>,
    ) -> ShaderModule {
        let source = wgpu::util::make_spirv(bytes);

        fw.device
            .create_shader_module(&wgpu::ShaderModuleDescriptor {
                label: shader_name,
                source,
            })
    }

    /// Creates a [`wgpu::ShaderModule`](wgpu::ShaderModule) instance from a `WGSL` file.
    /// Intended use is `examples` crates.
    pub fn from_wgsl_file(fw: &Framework, path: impl AsRef<Path>) -> GpuResult<ShaderModule> {
        let source_string = std::fs::read_to_string(path)?;

        Ok(fw
            .device
            .create_shader_module(&wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(Cow::Owned(source_string)),
            }))
    }
}

// TODO: CHECK CHECK CHECK
pub(crate) fn primitive_slice_to_bytes<P>(primitive: &[P]) -> &[u8]
where
    P: image::Primitive,
{
    let times = std::mem::size_of::<P>() / std::mem::size_of::<u8>();

    unsafe {
        let input_ptr = primitive.as_ptr();
        let new_ptr: *const u8 = std::mem::transmute(input_ptr);

        std::slice::from_raw_parts(new_ptr, primitive.len() * times)
    }
}
