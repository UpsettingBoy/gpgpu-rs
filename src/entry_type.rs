#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum EntryType {
    Buffer,
    Uniform,
    ConstImage,
    Image,
}
