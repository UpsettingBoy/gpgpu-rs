//! This modules controls the enablement of all the features
//! of the `gpgpu` crate.

#[cfg(feature = "integrate-image")]
pub mod integrate_image;

#[cfg(feature = "integrate-ndarray")]
pub mod integrate_ndarray;
