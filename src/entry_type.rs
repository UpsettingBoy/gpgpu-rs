#[derive(Clone, Copy)]
pub(crate) enum EntryType {
    Buffer,
    Uniform,
    ConstImage,
    Image,
}
