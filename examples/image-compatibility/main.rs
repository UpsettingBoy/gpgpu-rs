use gpgpu::primitives::pixels::Rgba8Uint;

// This example simply mirrors an image using the image crate compatibility feature.
fn main() {
    let fw = gpgpu::Framework::default();
    let shader_mod =
        gpgpu::utils::shader::from_wgsl_file(&fw, "examples/image-compatibility/mirror.wgsl")
            .unwrap();

    let dynamic_img = image::open("examples/image-compatibility/monke.jpg").unwrap(); // RGB8 image ...
    let rgba = dynamic_img.into_rgba8(); // ... converted to RGBA8

    let (width, height) = rgba.dimensions();

    // GPU image creation
    let input_img = gpgpu::GpuImage::from_image_crate(&fw, &rgba); // Input
    let output_img = gpgpu::GpuImage::<Rgba8Uint>::new(&fw, width, height); // Output

    let binds = gpgpu::DescriptorSet::default()
        .bind_image(&input_img)
        .bind_storage_image(&output_img, gpgpu::ImageUsage::WriteOnly);

    fw.create_kernel_builder(&shader_mod, "main")
        .add_descriptor_set(binds)
        .build()
        .enqueue(width / 32, height / 32, 1); // Since the kernel workgroup size is (32,32,1) dims are divided

    let output = output_img.read_to_image_buffer().unwrap();
    output
        .save("examples/image-compatibility/mirror-monke.png")
        .unwrap();
}
