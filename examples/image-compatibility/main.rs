use gpgpu::{primitives::pixels::Rgba8Uint, ImgOps};

// This example simply mirrors an image using the image crate compatibility feature.
fn main() {
    let fw = gpgpu::Framework::default();
    let shader =
        gpgpu::Shader::from_wgsl_file(&fw, "examples/image-compatibility/shader.wgsl").unwrap();

    let kernel = gpgpu::Kernel::new(
        &fw,
        &shader,
        "main",
        vec![gpgpu::new_set_layout!(
            0: ConstImage<Rgba8Uint>,
            1: Image<Rgba8Uint>
        )],
    );

    let dynamic_img = image::open("examples/image-compatibility/monke.jpg").unwrap(); // RGB8 image ...
    let rgba = dynamic_img.into_rgba8(); // ... converted to RGBA8

    let (width, height) = rgba.dimensions();

    // GPU image creation
    let input_img = gpgpu::GpuConstImage::from_image_buffer(&fw, &rgba); // Input
    let output_img = gpgpu::GpuImage::<Rgba8Uint>::new(&fw, width, height); // Output

    let binds = gpgpu::SetBindings::default()
        .add_const_image(0, &input_img)
        .add_image(1, &output_img);

    kernel.run(&fw, vec![binds], width / 32, height / 32, 1); // Since the kernel workgroup size is (32,32,1) dims are divided

    let output = output_img.read_to_image_buffer_blocking().unwrap();
    output
        .save("examples/image-compatibility/mirror-monke.png")
        .unwrap();
}
