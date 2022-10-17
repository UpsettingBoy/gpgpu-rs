struct Time {
    time: f32,
}

@group(0) @binding(0) var input: texture_2d<f32>;
@group(0) @binding(1) var output: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> time: Time; 

let pi: f32 = 3.14159;

@compute @workgroup_size(32, 32, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let coord = vec2<i32>(global_id.xy);
    let pixel = textureLoad(input, coord, 0);

    let dims = textureDimensions(input);
    let mirror_coord = vec2<i32>(dims.x - coord.x, coord.y);

    let t = pi * time.time;
    let colour = vec4<f32>(sin(t), sin(0.25 * t), sin(0.5 * t), 1.0);

    textureStore(output, mirror_coord, pixel.bgra * colour);
}