[[group(0), binding(0)]] var input: texture_2d<u32>;
[[group(0), binding(1)]] var output: texture_storage_2d<rgba8uint, write>;

[[stage(compute), workgroup_size(32, 32, 1)]]
fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
    let coord = vec2<i32>(global_id.xy);
    let pixel = textureLoad(input, coord, 0);

    let dims =  textureDimensions(input);
    let mirror_coord = vec2<i32>(dims.x - coord.x, coord.y);
    
    textureStore(output, mirror_coord, pixel);
}