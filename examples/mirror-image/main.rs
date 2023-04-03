use gpgpu::{primitives::pixels::Rgba8Uint, BindGroupLayoutBuilder, ImgOps};

// This example simply mirrors an image.
fn main() {
    let fw = gpgpu::Framework::default()
        .set_bind_group_layouts(vec![BindGroupLayoutBuilder::new()
            .add_const_image::<Rgba8Uint>()
            .add_image::<Rgba8Uint>()]);

    let shader = gpgpu::Shader::from_wgsl_file(&fw, "examples/mirror-image/shader.wgsl").unwrap();

    let dynamic_img = image::open("examples/mirror-image/monke.jpg").unwrap(); // RGB8 image ...
    let rgba = dynamic_img.into_rgba8(); // ... converted to RGBA8

    let (width, height) = rgba.dimensions();

    // GPU image creation
    let input_img = gpgpu::GpuConstImage::<Rgba8Uint>::new(&fw, width, height); // Input
    let output_img = gpgpu::GpuImage::<Rgba8Uint>::new(&fw, width, height); // Output

    // Write input image into the GPU
    input_img.write(&rgba).unwrap();

    let desc = gpgpu::DescriptorSet::new(0)
        .bind_const_image(&input_img)
        .bind_image(&output_img);
    let program = gpgpu::Program::new(&shader, "main").add_descriptor_set(desc);

    gpgpu::Kernel::new(&fw, program).enqueue(width / 32, height / 32, 1); // Since the kernel workgroup size is (32, 32, 1) dims are divided

    let output_bytes = output_img.read_vec_blocking().unwrap();
    image::save_buffer(
        "examples/mirror-image/mirror-monke.png",
        &output_bytes,
        width,
        height,
        image::ColorType::Rgba8,
    )
    .unwrap();
}
