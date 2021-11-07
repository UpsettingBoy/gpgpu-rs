use gpgpu::features::integrate_ndarray::GpuArray;

// Simple compute example that multiplies 2 vectors A and B, storing the result in a vector C using ndarray.
fn main() {
    let fw = gpgpu::Framework::default();

    let shader_mod =
        gpgpu::utils::shader::from_wgsl_file(&fw, "examples/ndarray/shader.wgsl").unwrap();

    let array_src = ndarray::array![[0u32, 1, 2, 3], [0u32, 1, 2, 3]];
    let mut array_zeroed = ndarray::Array::from(array_src.clone());

    array_zeroed *= 0;

    let array_src_view = array_src.view();
    let array_zeroed_view = array_zeroed.view();

    let gpu_array_a = gpgpu::GpuArray::from_array(&fw, array_src_view.clone()).unwrap();
    let gpu_array_b = gpgpu::GpuArray::from_array(&fw, array_src_view.clone()).unwrap();
    let gpu_array_c = gpgpu::GpuArray::from_array(&fw, array_zeroed_view).unwrap();

    // let desc = gpgpu::DescriptorSet::default()
}
