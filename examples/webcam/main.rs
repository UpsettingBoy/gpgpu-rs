use std::io::Write;

use gpgpu::{
    new_set_layout, primitives::pixels::Rgba8UintNorm, BufOps, DescriptorSet, Framework,
    GpuConstImage, GpuImage, GpuUniformBuffer, ImgOps,
};

use minifb::{Key, Window, WindowOptions};
use nokhwa::{Camera, CameraFormat, Resolution};

const WIDTH: usize = 1280;
const HEIGHT: usize = 720;

fn main() {
    let fw = Framework::default();
    let shader = gpgpu::Shader::from_wgsl_file(&fw, "examples/webcam/shader.wgsl").unwrap();

    let kernel = gpgpu::Kernel::new(
        &fw,
        &shader,
        "main",
        vec![new_set_layout! {
            0: ConstImage<Rgba8UintNorm>,
            1: Image<Rgba8UintNorm>,
            2: UniformBuffer,
        }],
    );

    // Camera initilization. Config may not work if not same cam as the Thinkpad T480 one.
    // Change parameters accordingly
    let mut camera = {
        let camera_format = CameraFormat::new(
            Resolution {
                width_x: WIDTH as u32,
                height_y: HEIGHT as u32,
            },
            nokhwa::FrameFormat::MJPEG,
            30,
        );

        Camera::new(0, Some(camera_format)).unwrap()
    };

    // Window initialization
    let mut window = Window::new(
        "gpgpu webcam example",
        WIDTH,
        HEIGHT,
        WindowOptions::default(),
    )
    .unwrap();

    camera.open_stream().unwrap();
    window.limit_update_rate(Some(std::time::Duration::from_secs_f32(1.0 / 60.0)));

    // Since the same GPU resources could be used during the whole execution
    // of the program, they are outside of the event loop
    let gpu_input = GpuConstImage::<Rgba8UintNorm>::new(&fw, WIDTH as u32, HEIGHT as u32); // Cam frame texture
    let buf_time = GpuUniformBuffer::<f32>::with_capacity(&fw, 1); // Elapsed time buffer (single element) for fancy shaders 😁

    let gpu_output = GpuImage::<Rgba8UintNorm>::new(&fw, WIDTH as u32, HEIGHT as u32); // Shader output

    let binds = gpgpu::SetBindings::default()
        .add_const(0, &gpu_input)
        .add_image(1, &gpu_output)
        .add_uniform_buffer(2, &buf_time);

    let time = std::time::Instant::now();

    let mut frame_buffer = vec![0u32; WIDTH * HEIGHT * 4];

    let mut total = 0.0;
    let mut count = 0;

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let fps = std::time::Instant::now();

        // Adapted for new (using image 0.24) nokhwa version
        let cam_raw = camera.frame().unwrap();
        let cam_buf = image::DynamicImage::ImageRgb8(cam_raw).into_rgba8(); // Obtain cam current frame

        gpu_input.write_image_buffer(&cam_buf).unwrap(); // Upload cam frame into the cam frame texture
        buf_time.write(&[time.elapsed().as_secs_f32()]).unwrap(); // Upload elapsed time into elapsed time buffer

        kernel.run(binds, WIDTH as u32 / 32, HEIGHT as u32 / 31, 1);

        gpu_output
            .read_blocking(bytemuck::cast_slice_mut(&mut frame_buffer))
            .unwrap();

        // Write processed cam frame into window frame buffer
        window
            .update_with_buffer(&frame_buffer, WIDTH, HEIGHT)
            .unwrap();

        print_fps(fps.elapsed().as_secs_f32(), &mut total, &mut count);
    }
}

fn print_fps(elapsed: f32, total: &mut f32, count: &mut u32) {
    let fps = 1.0 / elapsed;

    *total += fps;
    *count += 1;

    print!(
        "\rFPS: {:00.0}\tAverage: {:00.2}",
        fps,
        *total / *count as f32
    );

    std::io::stdout().flush().unwrap();
}
