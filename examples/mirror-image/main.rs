use gpgpu::primitives::pixels::Rgba8Uint;

// This example simply mirrors an image.
fn main() {
    let fw = gpgpu::Framework::default();
    let shader_mod =
        gpgpu::utils::shader::from_wgsl_file(&fw, "examples/mirror-image/mirror.wgsl").unwrap();

    let dynamic_img = image::open("examples/mirror-image/monke.jpg").unwrap(); // RGB8 image ...
    let rgba = dynamic_img.into_rgba8(); // ... converted to RGBA8

    let (width, height) = rgba.dimensions();

    // GPU image creation
    let mut input_img = gpgpu::GpuImage::<Rgba8Uint>::new(&fw, width, height); // Input
    let output_img = gpgpu::GpuImage::<Rgba8Uint>::new(&fw, width, height); // Output

    // Write input image into the GPU
    input_img.write(&rgba);

    let binds = gpgpu::DescriptorSet::default()
        .bind_image(&input_img)
        .bind_storage_image(&output_img, gpgpu::AccessMode::WriteOnly);

    fw.create_kernel_builder(&shader_mod, "main")
        .add_descriptor_set(binds)
        .build()
        .enqueue(width / 32, height / 32, 1); // Since the kernel workgroup size is (32,32,1) dims are divided

    let output_bytes = output_img.read().unwrap();

    image::save_buffer(
        "examples/mirror-image/mirror-monke.png",
        &output_bytes,
        width,
        height,
        image::ColorType::Rgba8,
    )
    .unwrap();
}
