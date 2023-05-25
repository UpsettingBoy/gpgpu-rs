use gpgpu::{primitives::pixels::Rgba8UintNorm, ImgOps};

// This example upscales some pixel art using 2 different samplers
fn main() {
    let fw = gpgpu::Framework::default();
    let shader = gpgpu::Shader::from_wgsl_file(&fw, "examples/samplers/shader.wgsl").unwrap();

    let dynamic_img = image::open("examples/samplers/pixelart.jpg").unwrap(); // RGB8 image ...
    let rgba = dynamic_img.into_rgba8(); // ... converted to RGBA8

    let (width, height) = rgba.dimensions();
    let scale = 30;
    let scaled_width = width * scale;
    let scaled_height = height * scale;

    // GPU image creation
    let input_img = gpgpu::GpuConstImage::<Rgba8UintNorm>::new(&fw, width, height); // Input
    let output_img = gpgpu::GpuImage::<Rgba8UintNorm>::new(&fw, scaled_width, scaled_height); // Output

    // Write input image into the GPU
    input_img.write(&rgba).unwrap();

    // Create a sampler
    let linear_sampler = gpgpu::Sampler::new(
        &fw,
        gpgpu::SamplerWrapMode::ClampToBorder,
        gpgpu::SamplerFilterMode::Linear,
    );
    let nearest_neighbor_sampler = gpgpu::Sampler::new(
        &fw,
        gpgpu::SamplerWrapMode::ClampToBorder,
        gpgpu::SamplerFilterMode::Nearest,
    );

    let desc = gpgpu::DescriptorSet::default()
        .bind_const_image(&input_img)
        .bind_image(&output_img)
        .bind_sampler(&linear_sampler)
        .bind_sampler(&nearest_neighbor_sampler);
    let program = gpgpu::Program::new(&shader, "main").add_descriptor_set(desc);

    gpgpu::Kernel::new(&fw, program).enqueue(scaled_width / 32, scaled_height / 32, 1); // Since the kernel workgroup size is (32, 32, 1) dims are divided

    let output_bytes = output_img.read_vec_blocking().unwrap();
    image::save_buffer(
        "examples/samplers/samplers.png",
        &output_bytes,
        scaled_width,
        scaled_height,
        image::ColorType::Rgba8,
    )
    .unwrap();
}
