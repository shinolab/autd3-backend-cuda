#![allow(unknown_lints)]
#![allow(clippy::manual_slice_size_calculation)]

mod cusolver;

use std::{collections::HashMap, ffi::CStr, fmt::Display};

use autd3_core::{
    acoustics::directivity::Sphere,
    gain::BitVec,
    geometry::{Geometry, Point3},
};
use autd3_gain_holo::{
    Complex, HoloError, LinAlgBackend, MatrixX, MatrixXc, Trans, VectorX, VectorXc,
};
use cuda_sys::cublas::{
    cublasOperation_t_CUBLAS_OP_C, cublasOperation_t_CUBLAS_OP_N, cublasOperation_t_CUBLAS_OP_T,
};
use cusolver::cudaDataType_t_CUDA_R_32F as CUDA_R_32F;
use thiserror::Error;

#[repr(C)]
#[repr(align(8))]
struct CuComplex(cuda_sys::cublas::cuFloatComplex);

fn make_complex(x: f32, y: f32) -> CuComplex {
    CuComplex(cuda_sys::cublas::cuFloatComplex {
        x,
        y,
        __bindgen_align: [],
    })
}

#[link(name = "autd3_cuda_kernel", kind = "static")]
unsafe extern "C" {
    fn cu_generate_propagation_matrix(
        positions: *const f32,
        foci: *const f32,
        wavenums: *const f32,
        row: u32,
        col: u32,
        dst: *mut CuComplex,
    );

    fn cu_scaled_to(
        a: *const CuComplex,
        b: *const CuComplex,
        row: u32,
        col: u32,
        c: *mut CuComplex,
    );

    fn cu_get_diagonal(x: *const f32, row: u32, col: u32, y: *mut f32);
    fn cu_get_diagonal_c(x: *const CuComplex, row: u32, col: u32, y: *mut CuComplex);
    fn cu_set_diagonal(x: *const f32, n: u32, y: *mut f32);
    fn cu_set_diagonal_c(x: *const CuComplex, n: u32, y: *mut CuComplex);
    fn cu_reciprocal(x: *const CuComplex, row: u32, col: u32, y: *mut CuComplex);
    fn cu_hadamard_product(
        x: *const CuComplex,
        y: *const CuComplex,
        row: u32,
        col: u32,
        z: *mut CuComplex,
    );

    fn cu_norm_squared(a: *const CuComplex, row: u32, col: u32, b: *mut f32);
    fn cu_make_complex(re: *const f32, row: u32, col: u32, dst: *mut CuComplex);
    fn cu_make_complex2(re: *const f32, im: *const f32, row: u32, col: u32, dst: *mut CuComplex);

    fn cu_conj(a: *const CuComplex, row: u32, col: u32, b: *mut CuComplex);

    fn cu_exp(a: *const CuComplex, row: u32, col: u32, b: *mut CuComplex);
    fn cu_real(a: *const CuComplex, row: u32, col: u32, b: *mut f32);
    fn cu_imag(a: *const CuComplex, row: u32, col: u32, b: *mut f32);

    fn cu_reduce_col(mat: *const f32, m: u32, n: u32, result: *mut f32);
}

fn convert_trans(trans: autd3_gain_holo::Trans) -> u32 {
    match trans {
        autd3_gain_holo::Trans::NoTrans => cublasOperation_t_CUBLAS_OP_N,
        autd3_gain_holo::Trans::Trans => cublasOperation_t_CUBLAS_OP_T,
        autd3_gain_holo::Trans::ConjTrans => cublasOperation_t_CUBLAS_OP_C,
    }
}

#[derive(Error, Debug)]
pub enum CUDABackendError {
    CuBLASError(cuda_sys::cublas::cublasStatus_t),
    CUDAError(cuda_sys::cudart::cudaError_t),
    CuSOLVERError(cusolver::cusolverStatus_t),
}

impl Display for CUDABackendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        unsafe {
            match self {
                CUDABackendError::CuBLASError(err) => write!(f, "cuBLAS Error: {:?}", err),
                CUDABackendError::CuSOLVERError(err) => write!(f, "cuSOLVER Error: {:?}", err),
                CUDABackendError::CUDAError(err) => write!(
                    f,
                    "CUDA Error: {}",
                    CStr::from_ptr(cuda_sys::cudart::cudaGetErrorString(*err))
                        .to_str()
                        .unwrap()
                ),
            }
        }
    }
}

impl From<CUDABackendError> for HoloError {
    fn from(value: CUDABackendError) -> Self {
        HoloError::BackendError(value.to_string())
    }
}

macro_rules! cu_call {
    ($f:expr) => {{
        let res = $f;
        let err = cuda_sys::cudart::cudaGetLastError();
        if err != cuda_sys::cudart::cudaError_t::Success {
            return Err(CUDABackendError::CUDAError(err).into());
        }
        res
    }};
}

macro_rules! cuda_call {
    ($f:expr) => {{
        let err = $f;
        if err != cuda_sys::cudart::cudaError_t::Success {
            return Err(CUDABackendError::CUDAError(err).into());
        }
    }};
}

macro_rules! cublas_call {
    ($f:expr) => {{
        let err = $f;
        if err != cuda_sys::cublas::cublasStatus_t::SUCCESS {
            return Err(CUDABackendError::CuBLASError(err).into());
        }
    }};
}

macro_rules! cusolver_call {
    ($f:expr) => {{
        let err = $f;
        if err != cusolver::cusolverStatus_t_CUSOLVER_STATUS_SUCCESS {
            return Err(CUDABackendError::CuSOLVERError(err).into());
        }
    }};
}

macro_rules! alloc_uninitialized {
    ($ty:ty, $len:expr) => {{
        let mut v: *mut $ty = std::ptr::null_mut();
        cuda_call!(cuda_sys::cudart::cudaMalloc(
            &mut v as *mut *mut $ty as _,
            std::mem::size_of::<$ty>() * $len,
        ));
        v
    }};
    ($ty:ty, $r:expr, $c:expr) => {{
        let mut v: *mut $ty = std::ptr::null_mut();
        cuda_call!(cuda_sys::cudart::cudaMalloc(
            &mut v as *mut *mut $ty as _,
            std::mem::size_of::<$ty>() * $r * $c,
        ));
        v
    }};
}

macro_rules! alloc_zeroed {
    ($ty:ty, $len:expr) => {{
        let v = alloc_uninitialized!($ty, $len);
        cuda_call!(cuda_sys::cudart::cudaMemset(
            v as _,
            0,
            std::mem::size_of::<$ty>() * $len
        ));
        v
    }};
    ($ty:ty, $r:expr, $c:expr) => {{
        let v = alloc_uninitialized!($ty, $r, $c);
        cuda_call!(cuda_sys::cudart::cudaMemset(
            v as _,
            0,
            std::mem::size_of::<$ty>() * $r * $c
        ));
        v
    }};
}

macro_rules! free {
    ($p:expr) => {{ cuda_call!(cuda_sys::cudart::cudaFree($p as _)) }};
}

macro_rules! cpy_host_to_device {
    ($ty:ty, $src:expr, $dst:expr, $len:expr) => {{
        cuda_call!(cuda_sys::cudart::cudaMemcpy(
            $dst as _,
            $src as _,
            std::mem::size_of::<$ty>() * $len,
            cuda_sys::cudart::cudaMemcpyKind_cudaMemcpyHostToDevice,
        ))
    }};
}

macro_rules! cpy_device_to_device {
    ($ty:ty, $src:expr, $dst:expr, $len:expr) => {{
        cuda_call!(cuda_sys::cudart::cudaMemcpy(
            $dst as _,
            $src as _,
            std::mem::size_of::<$ty>() * $len,
            cuda_sys::cudart::cudaMemcpyKind_cudaMemcpyDeviceToDevice,
        ))
    }};
}

macro_rules! cpy_device_to_host {
    ($ty:ty, $src:expr, $dst:expr, $len:expr) => {{
        cuda_call!(cuda_sys::cudart::cudaMemcpy(
            $dst as _,
            $src as _,
            std::mem::size_of::<$ty>() * $len,
            cuda_sys::cudart::cudaMemcpyKind_cudaMemcpyDeviceToHost,
        ))
    }};
}

pub struct CuVectorX {
    pub(crate) ptr: *mut f32,
    pub(crate) len: usize,
}

impl Drop for CuVectorX {
    fn drop(&mut self) {
        unsafe {
            let _ = cuda_sys::cudart::cudaFree(self.ptr as _);
        }
    }
}

pub struct CuVectorXc {
    pub(crate) ptr: *mut CuComplex,
    pub(crate) len: usize,
}

impl Drop for CuVectorXc {
    fn drop(&mut self) {
        unsafe {
            let _ = cuda_sys::cudart::cudaFree(self.ptr as _);
        }
    }
}

pub struct CuMatrixX {
    pub(crate) ptr: *mut f32,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
}

impl Drop for CuMatrixX {
    fn drop(&mut self) {
        unsafe {
            let _ = cuda_sys::cudart::cudaFree(self.ptr as _);
        }
    }
}

pub struct CuMatrixXc {
    pub(crate) ptr: *mut CuComplex,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
}

impl Drop for CuMatrixXc {
    fn drop(&mut self) {
        unsafe {
            let _ = cuda_sys::cudart::cudaFree(self.ptr as _);
        }
    }
}

/// Backend using CUDA
pub struct CUDABackend {
    handle: cuda_sys::cublas::cublasHandle_t,
    handle_s: cusolver::cusolverDnHandle_t,
}

impl CUDABackend {
    pub fn new() -> Result<Self, HoloError> {
        let mut handle: cuda_sys::cublas::cublasHandle_t = std::ptr::null_mut();
        unsafe {
            cublas_call!(cuda_sys::cublas::cublasCreate_v2(&mut handle as _));
        }

        let mut handle_s: cusolver::cusolverDnHandle_t = std::ptr::null_mut();
        unsafe { cusolver_call!(cusolver::cusolverDnCreate(&mut handle_s as _)) }

        Ok(Self { handle, handle_s })
    }
}

unsafe impl Send for CUDABackend {}
unsafe impl Sync for CUDABackend {}

impl Drop for CUDABackend {
    fn drop(&mut self) {
        unsafe {
            cuda_sys::cublas::cublasDestroy_v2(self.handle);
            cusolver::cusolverDnDestroy(self.handle_s);
        }
    }
}

impl LinAlgBackend<Sphere> for CUDABackend {
    type MatrixXc = CuMatrixXc;
    type MatrixX = CuMatrixX;
    type VectorXc = CuVectorXc;
    type VectorX = CuVectorX;

    fn generate_propagation_matrix(
        &self,
        geometry: &Geometry,
        foci: &[Point3],
        filter: Option<&HashMap<usize, BitVec>>,
    ) -> Result<Self::MatrixXc, HoloError> {
        let cols = geometry
            .devices()
            .map(|dev| dev.num_transducers())
            .sum::<usize>();
        let rows = foci.len();

        let foci = foci
            .iter()
            .flat_map(|f| f.iter().copied())
            .collect::<Vec<_>>();

        let mut positions = Vec::with_capacity(cols * 3);
        let mut wavenums = Vec::with_capacity(cols);

        if let Some(filter) = filter {
            geometry.devices().for_each(|dev| {
                if let Some(filter) = filter.get(&dev.idx()) {
                    let wavenumber = dev.wavenumber();
                    dev.iter().for_each(|tr| {
                        if filter[tr.idx()] {
                            let p = tr.position();
                            positions.push(p.x);
                            positions.push(p.y);
                            positions.push(p.z);
                            wavenums.push(wavenumber);
                        }
                    })
                }
            });
        } else {
            geometry.devices().for_each(|dev| {
                let wavenumber = dev.wavenumber();
                dev.iter().for_each(|tr| {
                    let p = tr.position();
                    positions.push(p.x);
                    positions.push(p.y);
                    positions.push(p.z);
                    wavenums.push(wavenumber);
                })
            });
        }

        let cols = wavenums.len();

        unsafe {
            let p_positions = alloc_uninitialized!(f32, positions.len());
            cpy_host_to_device!(f32, positions.as_ptr(), p_positions, positions.len());
            let p_foci = alloc_uninitialized!(f32, foci.len());
            cpy_host_to_device!(f32, foci.as_ptr(), p_foci, foci.len());
            let p_wavenums = alloc_uninitialized!(f32, wavenums.len());
            cpy_host_to_device!(f32, wavenums.as_ptr(), p_wavenums, wavenums.len());
            let ptr = alloc_uninitialized!(CuComplex, rows, cols);
            cu_call!(cu_generate_propagation_matrix(
                p_positions,
                p_foci,
                p_wavenums,
                rows as _,
                cols as _,
                ptr
            ));
            Ok(Self::MatrixXc { ptr, rows, cols })
        }
    }

    fn alloc_v(&self, size: usize) -> Result<Self::VectorX, HoloError> {
        unsafe {
            Ok(Self::VectorX {
                ptr: alloc_uninitialized!(f32, size),
                len: size,
            })
        }
    }

    fn alloc_m(&self, rows: usize, cols: usize) -> Result<Self::MatrixX, HoloError> {
        unsafe {
            Ok(Self::MatrixX {
                ptr: alloc_uninitialized!(f32, rows * cols),
                rows,
                cols,
            })
        }
    }

    fn alloc_cv(&self, size: usize) -> Result<Self::VectorXc, HoloError> {
        unsafe {
            Ok(Self::VectorXc {
                ptr: alloc_uninitialized!(CuComplex, size),
                len: size,
            })
        }
    }

    fn alloc_cm(&self, rows: usize, cols: usize) -> Result<Self::MatrixXc, HoloError> {
        unsafe {
            Ok(Self::MatrixXc {
                ptr: alloc_uninitialized!(CuComplex, rows * cols),
                rows,
                cols,
            })
        }
    }

    fn alloc_zeros_v(&self, size: usize) -> Result<Self::VectorX, HoloError> {
        unsafe {
            Ok(Self::VectorX {
                ptr: alloc_zeroed!(f32, size),
                len: size,
            })
        }
    }

    fn alloc_zeros_cv(&self, size: usize) -> Result<Self::VectorXc, HoloError> {
        unsafe {
            Ok(Self::VectorXc {
                ptr: alloc_zeroed!(CuComplex, size),
                len: size,
            })
        }
    }

    fn alloc_zeros_cm(&self, rows: usize, cols: usize) -> Result<Self::MatrixXc, HoloError> {
        unsafe {
            Ok(Self::MatrixXc {
                ptr: alloc_zeroed!(CuComplex, rows * cols),
                rows,
                cols,
            })
        }
    }

    fn to_host_v(&self, v: Self::VectorX) -> Result<VectorX, HoloError> {
        let mut dst = VectorX::zeros(v.len);
        unsafe { cpy_device_to_host!(f32, v.ptr, dst.as_mut_ptr(), v.len) }
        Ok(dst)
    }

    fn to_host_m(&self, v: Self::MatrixX) -> Result<MatrixX, HoloError> {
        let mut dst = MatrixX::zeros(v.rows, v.cols);
        unsafe { cpy_device_to_host!(f32, v.ptr, dst.as_mut_ptr(), v.rows * v.cols) }
        Ok(dst)
    }

    fn to_host_cv(&self, v: Self::VectorXc) -> Result<VectorXc, HoloError> {
        let mut dst = VectorXc::zeros(v.len);
        unsafe { cpy_device_to_host!(CuComplex, v.ptr, dst.as_mut_ptr(), v.len) }
        Ok(dst)
    }

    fn to_host_cm(&self, v: Self::MatrixXc) -> Result<MatrixXc, HoloError> {
        let mut dst = MatrixXc::zeros(v.rows, v.cols);
        unsafe { cpy_device_to_host!(CuComplex, v.ptr, dst.as_mut_ptr(), v.rows * v.cols) }
        Ok(dst)
    }

    fn from_slice_v(&self, v: &[f32]) -> Result<Self::VectorX, HoloError> {
        unsafe {
            let len = v.len();
            let ptr = alloc_uninitialized!(f32, len);
            cpy_host_to_device!(f32, v.as_ptr(), ptr, len);
            Ok(Self::VectorX { ptr, len })
        }
    }

    fn from_slice_m(
        &self,
        rows: usize,
        cols: usize,
        v: &[f32],
    ) -> Result<Self::MatrixX, HoloError> {
        unsafe {
            let len = v.len();
            let ptr = alloc_uninitialized!(f32, len);
            cpy_host_to_device!(f32, v.as_ptr(), ptr, len);
            Ok(Self::MatrixX { ptr, rows, cols })
        }
    }

    fn from_slice_cv(&self, v: &[f32]) -> Result<Self::VectorXc, HoloError> {
        unsafe {
            let len = v.len();
            let re = alloc_uninitialized!(f32, len);
            cpy_host_to_device!(f32, v.as_ptr(), re, len);
            let ptr = alloc_uninitialized!(CuComplex, len);
            cu_call!(cu_make_complex(re, len as _, 1, ptr));
            Ok(Self::VectorXc { ptr, len })
        }
    }

    fn from_slice2_cv(&self, r: &[f32], i: &[f32]) -> Result<Self::VectorXc, HoloError> {
        unsafe {
            let len = r.len();
            let re = alloc_uninitialized!(f32, len);
            cpy_host_to_device!(f32, r.as_ptr(), re, len);
            let im = alloc_uninitialized!(f32, len);
            cpy_host_to_device!(f32, i.as_ptr(), im, len);
            let ptr = alloc_uninitialized!(CuComplex, len);
            cu_call!(cu_make_complex2(re, im, len as _, 1, ptr));
            Ok(Self::VectorXc { ptr, len })
        }
    }

    fn from_slice2_cm(
        &self,
        rows: usize,
        cols: usize,
        r: &[f32],
        i: &[f32],
    ) -> Result<Self::MatrixXc, HoloError> {
        unsafe {
            let len = r.len();
            let re = alloc_uninitialized!(f32, len);
            cpy_host_to_device!(f32, r.as_ptr(), re, len);
            let im = alloc_uninitialized!(f32, len);
            cpy_host_to_device!(f32, i.as_ptr(), im, len);
            let ptr = alloc_uninitialized!(CuComplex, len);
            cu_call!(cu_make_complex2(re, im, rows as _, cols as _, ptr));
            Ok(Self::MatrixXc { ptr, rows, cols })
        }
    }

    fn copy_from_slice_v(&self, v: &[f32], dst: &mut Self::VectorX) -> Result<(), HoloError> {
        unsafe {
            cpy_host_to_device!(f32, v.as_ptr(), dst.ptr, v.len());
        }
        Ok(())
    }

    fn copy_to_v(&self, src: &Self::VectorX, dst: &mut Self::VectorX) -> Result<(), HoloError> {
        unsafe {
            cpy_device_to_device!(f32, src.ptr, dst.ptr, src.len);
        }
        Ok(())
    }

    fn copy_to_m(&self, src: &Self::MatrixX, dst: &mut Self::MatrixX) -> Result<(), HoloError> {
        unsafe {
            cpy_device_to_device!(f32, src.ptr, dst.ptr, src.rows * src.cols);
        }
        Ok(())
    }

    fn clone_v(&self, v: &Self::VectorX) -> Result<Self::VectorX, HoloError> {
        unsafe {
            let len = v.len;
            let ptr = alloc_uninitialized!(f32, len);
            cpy_device_to_device!(f32, v.ptr, ptr, len);
            Ok(Self::VectorX { ptr, len })
        }
    }

    fn clone_m(&self, v: &Self::MatrixX) -> Result<Self::MatrixX, HoloError> {
        unsafe {
            let len = v.rows * v.cols;
            let ptr = alloc_uninitialized!(f32, len);
            cpy_device_to_device!(f32, v.ptr, ptr, len);
            Ok(Self::MatrixX {
                ptr,
                rows: v.rows,
                cols: v.cols,
            })
        }
    }

    fn clone_cv(&self, v: &Self::VectorXc) -> Result<Self::VectorXc, HoloError> {
        unsafe {
            let len = v.len;
            let ptr = alloc_uninitialized!(CuComplex, len);
            cpy_device_to_device!(CuComplex, v.ptr, ptr, len);
            Ok(Self::VectorXc { ptr, len })
        }
    }

    fn clone_cm(&self, v: &Self::MatrixXc) -> Result<Self::MatrixXc, HoloError> {
        unsafe {
            let len = v.rows * v.cols;
            let ptr = alloc_uninitialized!(CuComplex, len);
            cpy_device_to_device!(CuComplex, v.ptr, ptr, len);
            Ok(Self::MatrixXc {
                ptr,
                rows: v.rows,
                cols: v.cols,
            })
        }
    }

    fn make_complex2_v(
        &self,
        real: &Self::VectorX,
        imag: &Self::VectorX,
        v: &mut Self::VectorXc,
    ) -> Result<(), HoloError> {
        unsafe {
            cu_call!(cu_make_complex2(
                real.ptr,
                imag.ptr,
                real.len as _,
                1,
                v.ptr
            ));
        }
        Ok(())
    }

    fn create_diagonal(&self, v: &Self::VectorX, a: &mut Self::MatrixX) -> Result<(), HoloError> {
        unsafe {
            cuda_call!(cuda_sys::cudart::cudaMemset(
                a.ptr as _,
                0,
                std::mem::size_of::<f32>() * a.rows * a.cols
            ));
            cu_call!(cu_set_diagonal(v.ptr as _, v.len as _, a.ptr));
        }
        Ok(())
    }

    fn create_diagonal_c(
        &self,
        v: &Self::VectorXc,
        a: &mut Self::MatrixXc,
    ) -> Result<(), HoloError> {
        unsafe {
            cuda_call!(cuda_sys::cudart::cudaMemset(
                a.ptr as _,
                0,
                std::mem::size_of::<CuComplex>() * a.rows * a.cols
            ));
            cu_call!(cu_set_diagonal_c(v.ptr as _, v.len as _, a.ptr as _));
        }
        Ok(())
    }

    fn get_diagonal(&self, a: &Self::MatrixX, v: &mut Self::VectorX) -> Result<(), HoloError> {
        unsafe {
            cu_call!(cu_get_diagonal(
                a.ptr as _,
                a.rows as _,
                a.cols as _,
                v.ptr as _
            ));
        }
        Ok(())
    }

    fn norm_squared_cv(&self, a: &Self::VectorXc, b: &mut Self::VectorX) -> Result<(), HoloError> {
        unsafe {
            cu_call!(cu_norm_squared(a.ptr as _, a.len as _, 1, b.ptr as _));
        }
        Ok(())
    }

    fn real_cm(&self, a: &Self::MatrixXc, b: &mut Self::MatrixX) -> Result<(), HoloError> {
        unsafe {
            cu_call!(cu_real(a.ptr as _, a.rows as _, a.cols as _, b.ptr as _));
        }
        Ok(())
    }

    fn imag_cm(&self, a: &Self::MatrixXc, b: &mut Self::MatrixX) -> Result<(), HoloError> {
        unsafe {
            cu_call!(cu_imag(a.ptr as _, a.rows as _, a.cols as _, b.ptr as _));
        }
        Ok(())
    }

    fn scale_assign_cv(
        &self,
        a: autd3_gain_holo::Complex,
        b: &mut Self::VectorXc,
    ) -> Result<(), HoloError> {
        let a = make_complex(a.re, a.im);
        unsafe {
            cublas_call!(cuda_sys::cublas::cublasCscal_v2(
                self.handle,
                b.len as _,
                &a as *const _ as _,
                b.ptr as _,
                1
            ));
        }
        Ok(())
    }

    fn conj_assign_v(&self, b: &mut Self::VectorXc) -> Result<(), HoloError> {
        unsafe {
            cu_call!(cu_conj(b.ptr as _, b.len as _, 1, b.ptr as _));
        }
        Ok(())
    }

    fn exp_assign_cv(&self, v: &mut Self::VectorXc) -> Result<(), HoloError> {
        unsafe {
            cu_call!(cu_exp(v.ptr as _, v.len as _, 1, v.ptr as _));
        }
        Ok(())
    }

    fn concat_col_cm(
        &self,
        a: &Self::MatrixXc,
        b: &Self::MatrixXc,
        c: &mut Self::MatrixXc,
    ) -> Result<(), HoloError> {
        unsafe {
            cuda_call!(cuda_sys::cudart::cudaMemcpy(
                c.ptr as _,
                a.ptr as _,
                a.rows * a.cols * std::mem::size_of::<CuComplex>(),
                cuda_sys::cudart::cudaMemcpyKind_cudaMemcpyDeviceToDevice
            ));
            cuda_call!(cuda_sys::cudart::cudaMemcpy(
                c.ptr.add(a.rows * a.cols) as _,
                b.ptr as _,
                b.rows * b.cols * std::mem::size_of::<CuComplex>(),
                cuda_sys::cudart::cudaMemcpyKind_cudaMemcpyDeviceToDevice
            ));
        }
        Ok(())
    }

    fn max_v(&self, m: &Self::VectorX) -> Result<f32, HoloError> {
        unsafe {
            let mut idx: i32 = 0;
            cublas_call!(cuda_sys::cublas::cublasIsamax_v2(
                self.handle,
                m.len as _,
                m.ptr as _,
                1,
                &mut idx as _,
            ));
            let mut res = 0.;
            cuda_call!(cuda_sys::cudart::cudaMemcpy(
                &mut res as *mut _ as _,
                m.ptr.add(idx as usize - 1) as _,
                std::mem::size_of::<f32>(),
                cuda_sys::cudart::cudaMemcpyKind_cudaMemcpyDeviceToHost,
            ));
            Ok(res)
        }
    }

    fn hadamard_product_cm(
        &self,
        x: &Self::MatrixXc,
        y: &Self::MatrixXc,
        z: &mut Self::MatrixXc,
    ) -> Result<(), HoloError> {
        unsafe {
            cu_call!(cu_hadamard_product(
                x.ptr as _,
                y.ptr as _,
                x.rows as _,
                x.cols as _,
                z.ptr as _
            ));
        }
        Ok(())
    }

    fn dot(&self, x: &Self::VectorX, y: &Self::VectorX) -> Result<f32, HoloError> {
        unsafe {
            let mut d: f32 = 0.;
            cublas_call!(cuda_sys::cublas::cublasSdot_v2(
                self.handle,
                x.len as _,
                x.ptr as _,
                1,
                y.ptr as _,
                1,
                &mut d as _,
            ));
            Ok(d)
        }
    }

    fn dot_c(
        &self,
        x: &Self::VectorXc,
        y: &Self::VectorXc,
    ) -> Result<autd3_gain_holo::Complex, HoloError> {
        unsafe {
            let mut d = autd3_gain_holo::Complex::new(0., 0.);
            cublas_call!(cuda_sys::cublas::cublasCdotc_v2(
                self.handle,
                x.len as _,
                x.ptr as _,
                1,
                y.ptr as _,
                1,
                &mut d as *mut _ as _,
            ));
            Ok(d)
        }
    }

    fn add_v(&self, alpha: f32, a: &Self::VectorX, b: &mut Self::VectorX) -> Result<(), HoloError> {
        unsafe {
            cublas_call!(cuda_sys::cublas::cublasSaxpy_v2(
                self.handle,
                a.len as _,
                &alpha as _,
                a.ptr as _,
                1,
                b.ptr as _,
                1
            ));
            Ok(())
        }
    }

    fn add_m(&self, alpha: f32, a: &Self::MatrixX, b: &mut Self::MatrixX) -> Result<(), HoloError> {
        unsafe {
            cublas_call!(cuda_sys::cublas::cublasSaxpy_v2(
                self.handle,
                (a.rows * a.cols) as _,
                &alpha as _,
                a.ptr as _,
                1,
                b.ptr as _,
                1
            ));
            Ok(())
        }
    }

    fn gevv_c(
        &self,
        trans_a: autd3_gain_holo::Trans,
        trans_b: autd3_gain_holo::Trans,
        alpha: autd3_gain_holo::Complex,
        a: &Self::VectorXc,
        x: &Self::VectorXc,
        beta: autd3_gain_holo::Complex,
        y: &mut Self::MatrixXc,
    ) -> Result<(), HoloError> {
        let transa = convert_trans(trans_a);
        let transb = convert_trans(trans_b);
        let alpha = make_complex(alpha.re, alpha.im);
        let beta = make_complex(beta.re, beta.im);

        let m = if transa == cublasOperation_t_CUBLAS_OP_N {
            a.len
        } else {
            1
        };
        if m != y.rows {
            return Err(CUDABackendError::CuBLASError(
                cuda_sys::cublas::cublasStatus_t::INVALID_VALUE,
            )
            .into());
        }
        let n = if transb == cublasOperation_t_CUBLAS_OP_N {
            1
        } else {
            x.len
        };
        if n != y.cols {
            return Err(CUDABackendError::CuBLASError(
                cuda_sys::cublas::cublasStatus_t::INVALID_VALUE,
            )
            .into());
        }
        let ka = if transa == cublasOperation_t_CUBLAS_OP_N {
            1
        } else {
            a.len
        };
        let kb = if transb == cublasOperation_t_CUBLAS_OP_N {
            x.len
        } else {
            1
        };
        if ka != kb {
            return Err(CUDABackendError::CuBLASError(
                cuda_sys::cublas::cublasStatus_t::INVALID_VALUE,
            )
            .into());
        }

        unsafe {
            cublas_call!(cuda_sys::cublas::cublasCgemm_v2(
                self.handle,
                transa,
                transb,
                m as _,
                n as _,
                ka as _,
                &alpha as *const _ as _,
                a.ptr as _,
                a.len as _,
                x.ptr as _,
                x.len as _,
                &beta as *const _ as _,
                y.ptr as _,
                y.rows as _,
            ));
        }
        Ok(())
    }

    fn gemv_c(
        &self,
        trans: autd3_gain_holo::Trans,
        alpha: autd3_gain_holo::Complex,
        a: &Self::MatrixXc,
        x: &Self::VectorXc,
        beta: autd3_gain_holo::Complex,
        y: &mut Self::VectorXc,
    ) -> Result<(), HoloError> {
        let trans = convert_trans(trans);

        let alpha = make_complex(alpha.re, alpha.im);
        let beta = make_complex(beta.re, beta.im);

        unsafe {
            let m = a.rows;
            let n = a.cols;
            let lda = m;

            let ap = a.ptr;
            let bp = x.ptr;
            let cp = y.ptr;

            cublas_call!(cuda_sys::cublas::cublasCgemv_v2(
                self.handle,
                trans,
                m as _,
                n as _,
                &alpha as *const _ as _,
                ap as _,
                lda as _,
                bp as _,
                1,
                &beta as *const _ as _,
                cp as _,
                1,
            ));

            Ok(())
        }
    }

    fn gemm_c(
        &self,
        trans_a: autd3_gain_holo::Trans,
        trans_b: autd3_gain_holo::Trans,
        alpha: autd3_gain_holo::Complex,
        a: &Self::MatrixXc,
        b: &Self::MatrixXc,
        beta: autd3_gain_holo::Complex,
        y: &mut Self::MatrixXc,
    ) -> Result<(), HoloError> {
        let transa = convert_trans(trans_a);
        let transb = convert_trans(trans_b);

        let alpha = make_complex(alpha.re, alpha.im);
        let beta = make_complex(beta.re, beta.im);

        unsafe {
            cublas_call!(cuda_sys::cublas::cublasCgemm_v2(
                self.handle,
                transa,
                transb,
                y.rows as _,
                y.cols as _,
                if transa == cublasOperation_t_CUBLAS_OP_N {
                    a.cols
                } else {
                    a.rows
                } as _,
                &alpha as *const _ as _,
                a.ptr as _,
                a.rows as _,
                b.ptr as _,
                b.rows as _,
                &beta as *const _ as _,
                y.ptr as _,
                y.rows as _,
            ));
        }
        Ok(())
    }

    fn solve_inplace(&self, a: &Self::MatrixX, x: &mut Self::VectorX) -> Result<(), HoloError> {
        unsafe {
            let n = a.cols;
            let lda = a.rows;
            let ldb = x.len;

            let ap = a.ptr;
            let bp = x.ptr;

            let mut workspace_in_bytes_on_device: usize = 0;
            let mut workspace_in_bytes_on_host: usize = 0;
            cusolver_call!(cusolver::cusolverDnXpotrf_bufferSize(
                self.handle_s,
                std::ptr::null_mut(),
                cusolver::cublasFillMode_t_CUBLAS_FILL_MODE_UPPER,
                n as _,
                CUDA_R_32F,
                ap as _,
                lda as _,
                CUDA_R_32F,
                &mut workspace_in_bytes_on_device as _,
                &mut workspace_in_bytes_on_host as _,
            ));

            let workspace_buffer_on_device = alloc_uninitialized!(u8, workspace_in_bytes_on_device);
            let mut workspace_buffer_on_host_v = vec![0u8; workspace_in_bytes_on_host];
            let workspace_buffer_on_host = if workspace_in_bytes_on_host > 0 {
                workspace_buffer_on_host_v.as_mut_ptr()
            } else {
                std::ptr::null_mut()
            };

            let info = alloc_uninitialized!(i32, 1);

            cusolver_call!(cusolver::cusolverDnXpotrf(
                self.handle_s,
                std::ptr::null_mut(),
                cusolver::cublasFillMode_t_CUBLAS_FILL_MODE_UPPER,
                n as _,
                CUDA_R_32F,
                ap as _,
                lda as _,
                CUDA_R_32F,
                workspace_buffer_on_device as _,
                workspace_in_bytes_on_device,
                workspace_buffer_on_host as _,
                workspace_in_bytes_on_host,
                info as _,
            ));
            cusolver_call!(cusolver::cusolverDnXpotrs(
                self.handle_s,
                std::ptr::null_mut(),
                cusolver::cublasFillMode_t_CUBLAS_FILL_MODE_UPPER,
                n as _,
                1,
                CUDA_R_32F,
                ap as _,
                lda as _,
                CUDA_R_32F,
                bp as _,
                ldb as _,
                info as _,
            ));

            free!(info);
            free!(workspace_buffer_on_device);
        }
        Ok(())
    }

    fn reduce_col(&self, a: &Self::MatrixX, b: &mut Self::VectorX) -> Result<(), HoloError> {
        unsafe {
            cu_call!(cu_reduce_col(
                a.ptr as _,
                a.rows as _,
                a.cols as _,
                b.ptr as _,
            ));
        }
        Ok(())
    }

    fn scaled_to_cv(
        &self,
        a: &Self::VectorXc,
        b: &Self::VectorXc,
        c: &mut Self::VectorXc,
    ) -> Result<(), HoloError> {
        unsafe {
            cu_call!(cu_scaled_to(
                a.ptr as _, b.ptr as _, a.len as _, 1, c.ptr as _
            ));
        }
        Ok(())
    }

    fn scaled_to_assign_cv(
        &self,
        a: &Self::VectorXc,
        b: &mut Self::VectorXc,
    ) -> Result<(), HoloError> {
        unsafe {
            cu_call!(cu_scaled_to(
                b.ptr as _, a.ptr as _, a.len as _, 1, b.ptr as _
            ));
        }
        Ok(())
    }

    fn cols_c(&self, m: &Self::MatrixXc) -> Result<usize, HoloError> {
        Ok(m.cols)
    }

    fn gen_back_prop(
        &self,
        m: usize,
        n: usize,
        transfer: &Self::MatrixXc,
    ) -> Result<Self::MatrixXc, HoloError> {
        let mut tmp = self.alloc_zeros_cm(n, n)?;

        self.gemm_c(
            Trans::NoTrans,
            Trans::ConjTrans,
            Complex::new(1., 0.),
            transfer,
            transfer,
            Complex::new(0., 0.),
            &mut tmp,
        )?;

        let denominator = self.alloc_cv(n)?;
        unsafe {
            cu_call!(cu_get_diagonal_c(
                tmp.ptr as _,
                tmp.rows as _,
                tmp.cols as _,
                denominator.ptr as _
            ));
        }
        unsafe {
            cu_call!(cu_reciprocal(
                denominator.ptr as _,
                denominator.len as _,
                1,
                denominator.ptr as _
            ));
        }

        self.create_diagonal_c(&denominator, &mut tmp)?;

        let mut b = self.alloc_zeros_cm(m, n)?;
        self.gemm_c(
            Trans::ConjTrans,
            Trans::NoTrans,
            Complex::new(1., 0.),
            transfer,
            &tmp,
            Complex::new(0., 0.),
            &mut b,
        )?;
        Ok(b)
    }
}

#[cfg(test)]
mod tests {
    use autd3::driver::autd3_device::AUTD3;
    use autd3_core::{
        acoustics::{directivity::Sphere, propagate},
        defined::PI,
        geometry::UnitQuaternion,
    };

    use nalgebra::{ComplexField, Normed};

    use autd3_gain_holo::{Amplitude, Pa, Trans};

    use super::*;

    use rand::Rng;

    const N: usize = 10;
    const EPS: f32 = 1e-3;

    fn generate_geometry(size: usize) -> Geometry {
        Geometry::new(
            (0..size)
                .flat_map(|i| {
                    (0..size).map(move |j| {
                        AUTD3 {
                            pos: Point3::new(
                                i as f32 * AUTD3::DEVICE_WIDTH,
                                j as f32 * AUTD3::DEVICE_HEIGHT,
                                0.,
                            ),
                            rot: UnitQuaternion::identity(),
                        }
                        .into()
                    })
                })
                .collect(),
        )
    }

    fn gen_foci(n: usize) -> impl Iterator<Item = (Point3, Amplitude)> {
        (0..n).map(move |i| {
            (
                Point3::new(
                    90. + 10. * (2.0 * PI * i as f32 / n as f32).cos(),
                    70. + 10. * (2.0 * PI * i as f32 / n as f32).sin(),
                    150.,
                ),
                10e3 * Pa,
            )
        })
    }

    fn make_random_v(backend: &CUDABackend, size: usize) -> Result<CuVectorX, HoloError> {
        let mut rng = rand::rng();
        let v: Vec<f32> = (&mut rng)
            .sample_iter(rand::distr::StandardUniform)
            .take(size)
            .collect();
        backend.from_slice_v(&v)
    }

    fn make_random_m(
        backend: &CUDABackend,
        rows: usize,
        cols: usize,
    ) -> Result<CuMatrixX, HoloError> {
        let mut rng = rand::rng();
        let v: Vec<f32> = (&mut rng)
            .sample_iter(rand::distr::StandardUniform)
            .take(rows * cols)
            .collect();
        backend.from_slice_m(rows, cols, &v)
    }

    fn make_random_cv(backend: &CUDABackend, size: usize) -> Result<CuVectorXc, HoloError> {
        let mut rng = rand::rng();
        let real: Vec<f32> = (&mut rng)
            .sample_iter(rand::distr::StandardUniform)
            .take(size)
            .collect();
        let imag: Vec<f32> = (&mut rng)
            .sample_iter(rand::distr::StandardUniform)
            .take(size)
            .collect();
        backend.from_slice2_cv(&real, &imag)
    }

    fn make_random_cm(
        backend: &CUDABackend,
        rows: usize,
        cols: usize,
    ) -> Result<CuMatrixXc, HoloError> {
        let mut rng = rand::rng();
        let real: Vec<f32> = (&mut rng)
            .sample_iter(rand::distr::StandardUniform)
            .take(rows * cols)
            .collect();
        let imag: Vec<f32> = (&mut rng)
            .sample_iter(rand::distr::StandardUniform)
            .take(rows * cols)
            .collect();
        backend.from_slice2_cm(rows, cols, &real, &imag)
    }

    #[rstest::fixture]
    fn backend() -> CUDABackend {
        CUDABackend::new().unwrap()
    }

    #[rstest::rstest]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_alloc_v(backend: CUDABackend) -> Result<(), HoloError> {
        let v = backend.alloc_v(N)?;
        let v = backend.to_host_v(v)?;

        assert_eq!(N, v.len());
        Ok(())
    }

    #[rstest::rstest]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_alloc_m(backend: CUDABackend) -> Result<(), HoloError> {
        let m = backend.alloc_m(N, 2 * N)?;
        let m = backend.to_host_m(m)?;

        assert_eq!(N, m.nrows());
        assert_eq!(2 * N, m.ncols());
        Ok(())
    }

    #[rstest::rstest]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_alloc_cv(backend: CUDABackend) -> Result<(), HoloError> {
        let v = backend.alloc_cv(N)?;
        let v = backend.to_host_cv(v)?;

        assert_eq!(N, v.len());
        Ok(())
    }

    #[rstest::rstest]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_alloc_cm(backend: CUDABackend) -> Result<(), HoloError> {
        let m = backend.alloc_cm(N, 2 * N)?;
        let m = backend.to_host_cm(m)?;

        assert_eq!(N, m.nrows());
        assert_eq!(2 * N, m.ncols());
        Ok(())
    }

    #[rstest::rstest]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_alloc_zeros_v(backend: CUDABackend) -> Result<(), HoloError> {
        let v = backend.alloc_zeros_v(N)?;
        let v = backend.to_host_v(v)?;

        assert_eq!(N, v.len());
        assert!(v.iter().all(|&v| v == 0.));
        Ok(())
    }

    #[rstest::rstest]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_alloc_zeros_cv(backend: CUDABackend) -> Result<(), HoloError> {
        let v = backend.alloc_zeros_cv(N)?;
        let v = backend.to_host_cv(v)?;

        assert_eq!(N, v.len());
        assert!(v.iter().all(|&v| v == Complex::new(0., 0.)));
        Ok(())
    }

    #[rstest::rstest]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_alloc_zeros_cm(backend: CUDABackend) -> Result<(), HoloError> {
        let m = backend.alloc_zeros_cm(N, 2 * N)?;
        let m = backend.to_host_cm(m)?;

        assert_eq!(N, m.nrows());
        assert_eq!(2 * N, m.ncols());
        assert!(m.iter().all(|&v| v == Complex::new(0., 0.)));
        Ok(())
    }

    #[rstest::rstest]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_cols_c(backend: CUDABackend) -> Result<(), HoloError> {
        let m = backend.alloc_cm(N, 2 * N)?;

        assert_eq!(2 * N, backend.cols_c(&m)?);

        Ok(())
    }

    #[rstest::rstest]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_from_slice_v(backend: CUDABackend) -> Result<(), HoloError> {
        let rng = rand::rng();

        let v: Vec<f32> = rng
            .sample_iter(rand::distr::StandardUniform)
            .take(N)
            .collect();

        let c = backend.from_slice_v(&v)?;
        let c = backend.to_host_v(c)?;

        assert_eq!(N, c.len());
        v.iter().zip(c.iter()).for_each(|(&r, &c)| {
            assert_eq!(r, c);
        });
        Ok(())
    }

    #[rstest::rstest]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_from_slice_m(backend: CUDABackend) -> Result<(), HoloError> {
        let rng = rand::rng();

        let v: Vec<f32> = rng
            .sample_iter(rand::distr::StandardUniform)
            .take(N * 2 * N)
            .collect();

        let c = backend.from_slice_m(N, 2 * N, &v)?;
        let c = backend.to_host_m(c)?;

        assert_eq!(N, c.nrows());
        assert_eq!(2 * N, c.ncols());
        (0..2 * N).for_each(|col| {
            (0..N).for_each(|row| {
                assert_eq!(v[col * N + row], c[(row, col)]);
            })
        });
        Ok(())
    }

    #[rstest::rstest]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_from_slice_cv(backend: CUDABackend) -> Result<(), HoloError> {
        let rng = rand::rng();

        let real: Vec<f32> = rng
            .sample_iter(rand::distr::StandardUniform)
            .take(N)
            .collect();

        let c = backend.from_slice_cv(&real)?;
        let c = backend.to_host_cv(c)?;

        assert_eq!(N, c.len());
        real.iter().zip(c.iter()).for_each(|(r, c)| {
            assert_eq!(r, &c.re);
            assert_eq!(0.0, c.im);
        });
        Ok(())
    }

    #[rstest::rstest]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_from_slice2_cv(backend: CUDABackend) -> Result<(), HoloError> {
        let mut rng = rand::rng();

        let real: Vec<f32> = (&mut rng)
            .sample_iter(rand::distr::StandardUniform)
            .take(N)
            .collect();
        let imag: Vec<f32> = (&mut rng)
            .sample_iter(rand::distr::StandardUniform)
            .take(N)
            .collect();

        let c = backend.from_slice2_cv(&real, &imag)?;
        let c = backend.to_host_cv(c)?;

        assert_eq!(N, c.len());
        real.iter()
            .zip(imag.iter())
            .zip(c.iter())
            .for_each(|((r, i), c)| {
                assert_eq!(r, &c.re);
                assert_eq!(i, &c.im);
            });
        Ok(())
    }

    #[rstest::rstest]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_from_slice2_cm(backend: CUDABackend) -> Result<(), HoloError> {
        let mut rng = rand::rng();

        let real: Vec<f32> = (&mut rng)
            .sample_iter(rand::distr::StandardUniform)
            .take(N * 2 * N)
            .collect();
        let imag: Vec<f32> = (&mut rng)
            .sample_iter(rand::distr::StandardUniform)
            .take(N * 2 * N)
            .collect();

        let c = backend.from_slice2_cm(N, 2 * N, &real, &imag)?;
        let c = backend.to_host_cm(c)?;

        assert_eq!(N, c.nrows());
        assert_eq!(2 * N, c.ncols());
        (0..2 * N).for_each(|col| {
            (0..N).for_each(|row| {
                assert_eq!(real[col * N + row], c[(row, col)].re);
                assert_eq!(imag[col * N + row], c[(row, col)].im);
            })
        });
        Ok(())
    }

    #[rstest::rstest]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_copy_from_slice_v(backend: CUDABackend) -> Result<(), HoloError> {
        {
            let mut a = backend.alloc_zeros_v(N)?;
            let mut rng = rand::rng();
            let v = (&mut rng)
                .sample_iter(rand::distr::StandardUniform)
                .take(N / 2)
                .collect::<Vec<f32>>();

            backend.copy_from_slice_v(&v, &mut a)?;

            let a = backend.to_host_v(a)?;
            (0..N / 2).for_each(|i| {
                assert_eq!(v[i], a[i]);
            });
            (N / 2..N).for_each(|i| {
                assert_eq!(0., a[i]);
            });
        }

        {
            let mut a = backend.alloc_zeros_v(N)?;
            let v = [];

            backend.copy_from_slice_v(&v, &mut a)?;

            let a = backend.to_host_v(a)?;
            a.iter().for_each(|&a| {
                assert_eq!(0., a);
            });
        }

        Ok(())
    }

    #[rstest::rstest]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_copy_to_v(backend: CUDABackend) -> Result<(), HoloError> {
        let a = make_random_v(&backend, N)?;
        let mut b = backend.alloc_v(N)?;

        backend.copy_to_v(&a, &mut b)?;

        let a = backend.to_host_v(a)?;
        let b = backend.to_host_v(b)?;
        a.iter().zip(b.iter()).for_each(|(a, b)| {
            assert_eq!(a, b);
        });
        Ok(())
    }

    #[rstest::rstest]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_copy_to_m(backend: CUDABackend) -> Result<(), HoloError> {
        let a = make_random_m(&backend, N, N)?;
        let mut b = backend.alloc_m(N, N)?;

        backend.copy_to_m(&a, &mut b)?;

        let a = backend.to_host_m(a)?;
        let b = backend.to_host_m(b)?;
        a.iter().zip(b.iter()).for_each(|(a, b)| {
            assert_eq!(a, b);
        });
        Ok(())
    }

    #[rstest::rstest]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_clone_v(backend: CUDABackend) -> Result<(), HoloError> {
        let c = make_random_v(&backend, N)?;
        let c2 = backend.clone_v(&c)?;

        let c = backend.to_host_v(c)?;
        let c2 = backend.to_host_v(c2)?;

        c.iter().zip(c2.iter()).for_each(|(c, c2)| {
            assert_eq!(c, c2);
        });
        Ok(())
    }

    #[rstest::rstest]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_clone_m(backend: CUDABackend) -> Result<(), HoloError> {
        let c = make_random_m(&backend, N, N)?;
        let c2 = backend.clone_m(&c)?;

        let c = backend.to_host_m(c)?;
        let c2 = backend.to_host_m(c2)?;

        c.iter().zip(c2.iter()).for_each(|(c, c2)| {
            assert_eq!(c, c2);
        });
        Ok(())
    }

    #[rstest::rstest]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_clone_cv(backend: CUDABackend) -> Result<(), HoloError> {
        let c = make_random_cv(&backend, N)?;
        let c2 = backend.clone_cv(&c)?;

        let c = backend.to_host_cv(c)?;
        let c2 = backend.to_host_cv(c2)?;

        c.iter().zip(c2.iter()).for_each(|(c, c2)| {
            assert_eq!(c.re, c2.re);
            assert_eq!(c.im, c2.im);
        });
        Ok(())
    }

    #[rstest::rstest]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_clone_cm(backend: CUDABackend) -> Result<(), HoloError> {
        let c = make_random_cm(&backend, N, N)?;
        let c2 = backend.clone_cm(&c)?;

        let c = backend.to_host_cm(c)?;
        let c2 = backend.to_host_cm(c2)?;

        c.iter().zip(c2.iter()).for_each(|(c, c2)| {
            assert_eq!(c.re, c2.re);
            assert_eq!(c.im, c2.im);
        });
        Ok(())
    }

    #[rstest::rstest]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_make_complex2_v(backend: CUDABackend) -> Result<(), HoloError> {
        let real = make_random_v(&backend, N)?;
        let imag = make_random_v(&backend, N)?;

        let mut c = backend.alloc_cv(N)?;
        backend.make_complex2_v(&real, &imag, &mut c)?;

        let real = backend.to_host_v(real)?;
        let imag = backend.to_host_v(imag)?;
        let c = backend.to_host_cv(c)?;
        real.iter()
            .zip(imag.iter())
            .zip(c.iter())
            .for_each(|((r, i), c)| {
                assert_eq!(r, &c.re);
                assert_eq!(i, &c.im);
            });
        Ok(())
    }

    #[rstest::rstest]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_create_diagonal(backend: CUDABackend) -> Result<(), HoloError> {
        let diagonal = make_random_v(&backend, N)?;

        let mut c = backend.alloc_m(N, N)?;

        backend.create_diagonal(&diagonal, &mut c)?;

        let diagonal = backend.to_host_v(diagonal)?;
        let c = backend.to_host_m(c)?;
        (0..N).for_each(|i| {
            (0..N).for_each(|j| {
                if i == j {
                    assert_eq!(diagonal[i], c[(i, j)]);
                } else {
                    assert_eq!(0.0, c[(i, j)]);
                }
            })
        });
        Ok(())
    }

    #[rstest::rstest]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_create_diagonal_c(backend: CUDABackend) -> Result<(), HoloError> {
        let diagonal = make_random_cv(&backend, N)?;

        let mut c = backend.alloc_cm(N, N)?;

        backend.create_diagonal_c(&diagonal, &mut c)?;

        let diagonal = backend.to_host_cv(diagonal)?;
        let c = backend.to_host_cm(c)?;
        (0..N).for_each(|i| {
            (0..N).for_each(|j| {
                if i == j {
                    assert_eq!(diagonal[i].re, c[(i, j)].re);
                    assert_eq!(diagonal[i].im, c[(i, j)].im);
                } else {
                    assert_eq!(0.0, c[(i, j)].re);
                    assert_eq!(0.0, c[(i, j)].im);
                }
            })
        });
        Ok(())
    }

    #[rstest::rstest]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_get_diagonal(backend: CUDABackend) -> Result<(), HoloError> {
        let m = make_random_m(&backend, N, N)?;
        let mut diagonal = backend.alloc_v(N)?;

        backend.get_diagonal(&m, &mut diagonal)?;

        let m = backend.to_host_m(m)?;
        let diagonal = backend.to_host_v(diagonal)?;
        (0..N).for_each(|i| {
            assert_eq!(m[(i, i)], diagonal[i]);
        });
        Ok(())
    }

    #[rstest::rstest]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_norm_squared_cv(backend: CUDABackend) -> Result<(), HoloError> {
        let v = make_random_cv(&backend, N)?;

        let mut abs = backend.alloc_v(N)?;
        backend.norm_squared_cv(&v, &mut abs)?;

        let v = backend.to_host_cv(v)?;
        let abs = backend.to_host_v(abs)?;
        v.iter().zip(abs.iter()).for_each(|(v, abs)| {
            assert_approx_eq::assert_approx_eq!(v.norm_squared(), abs, EPS);
        });
        Ok(())
    }

    #[rstest::rstest]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_real_cm(backend: CUDABackend) -> Result<(), HoloError> {
        let v = make_random_cm(&backend, N, N)?;
        let mut r = backend.alloc_m(N, N)?;

        backend.real_cm(&v, &mut r)?;

        let v = backend.to_host_cm(v)?;
        let r = backend.to_host_m(r)?;
        (0..N).for_each(|i| {
            (0..N).for_each(|j| {
                assert_approx_eq::assert_approx_eq!(v[(i, j)].re, r[(i, j)], EPS);
            })
        });
        Ok(())
    }

    #[rstest::rstest]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_imag_cm(backend: CUDABackend) -> Result<(), HoloError> {
        let v = make_random_cm(&backend, N, N)?;
        let mut r = backend.alloc_m(N, N)?;

        backend.imag_cm(&v, &mut r)?;

        let v = backend.to_host_cm(v)?;
        let r = backend.to_host_m(r)?;
        (0..N).for_each(|i| {
            (0..N).for_each(|j| {
                assert_approx_eq::assert_approx_eq!(v[(i, j)].im, r[(i, j)], EPS);
            })
        });
        Ok(())
    }

    #[rstest::rstest]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_scale_assign_cv(backend: CUDABackend) -> Result<(), HoloError> {
        let mut v = make_random_cv(&backend, N)?;
        let vc = backend.clone_cv(&v)?;
        let mut rng = rand::rng();
        let scale = Complex::new(rng.random(), rng.random());

        backend.scale_assign_cv(scale, &mut v)?;

        let v = backend.to_host_cv(v)?;
        let vc = backend.to_host_cv(vc)?;
        v.iter().zip(vc.iter()).for_each(|(&v, &vc)| {
            assert_approx_eq::assert_approx_eq!(scale * vc, v, EPS);
        });
        Ok(())
    }

    #[rstest::rstest]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_conj_assign_v(backend: CUDABackend) -> Result<(), HoloError> {
        let mut v = make_random_cv(&backend, N)?;
        let vc = backend.clone_cv(&v)?;

        backend.conj_assign_v(&mut v)?;

        let v = backend.to_host_cv(v)?;
        let vc = backend.to_host_cv(vc)?;
        v.iter().zip(vc.iter()).for_each(|(&v, &vc)| {
            assert_eq!(vc.re, v.re);
            assert_eq!(vc.im, -v.im);
        });
        Ok(())
    }

    #[rstest::rstest]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_exp_assign_cv(backend: CUDABackend) -> Result<(), HoloError> {
        let mut v = make_random_cv(&backend, N)?;
        let vc = backend.clone_cv(&v)?;

        backend.exp_assign_cv(&mut v)?;

        let v = backend.to_host_cv(v)?;
        let vc = backend.to_host_cv(vc)?;
        v.iter().zip(vc.iter()).for_each(|(v, vc)| {
            assert_approx_eq::assert_approx_eq!(vc.exp(), v, EPS);
        });
        Ok(())
    }

    #[rstest::rstest]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_concat_col_cm(backend: CUDABackend) -> Result<(), HoloError> {
        let a = make_random_cm(&backend, N, N)?;
        let b = make_random_cm(&backend, N, 2 * N)?;
        let mut c = backend.alloc_cm(N, N + 2 * N)?;

        backend.concat_col_cm(&a, &b, &mut c)?;

        let a = backend.to_host_cm(a)?;
        let b = backend.to_host_cm(b)?;
        let c = backend.to_host_cm(c)?;
        (0..N).for_each(|col| (0..N).for_each(|row| assert_eq!(a[(row, col)], c[(row, col)])));
        (0..2 * N)
            .for_each(|col| (0..N).for_each(|row| assert_eq!(b[(row, col)], c[(row, N + col)])));
        Ok(())
    }

    #[rstest::rstest]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_max_v(backend: CUDABackend) -> Result<(), HoloError> {
        let v = make_random_v(&backend, N)?;

        let max = backend.max_v(&v)?;

        let v = backend.to_host_v(v)?;
        assert_eq!(
            *v.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap(),
            max
        );
        Ok(())
    }

    #[rstest::rstest]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_hadamard_product_cm(backend: CUDABackend) -> Result<(), HoloError> {
        let a = make_random_cm(&backend, N, N)?;
        let b = make_random_cm(&backend, N, N)?;
        let mut c = backend.alloc_cm(N, N)?;

        backend.hadamard_product_cm(&a, &b, &mut c)?;

        let a = backend.to_host_cm(a)?;
        let b = backend.to_host_cm(b)?;
        let c = backend.to_host_cm(c)?;
        c.iter()
            .zip(a.iter())
            .zip(b.iter())
            .for_each(|((c, a), b)| {
                assert_approx_eq::assert_approx_eq!(a.re * b.re - a.im * b.im, c.re, EPS);
                assert_approx_eq::assert_approx_eq!(a.re * b.im + a.im * b.re, c.im, EPS);
            });
        Ok(())
    }

    #[rstest::rstest]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_dot(backend: CUDABackend) -> Result<(), HoloError> {
        let a = make_random_v(&backend, N)?;
        let b = make_random_v(&backend, N)?;

        let dot = backend.dot(&a, &b)?;

        let a = backend.to_host_v(a)?;
        let b = backend.to_host_v(b)?;
        let expect = a.iter().zip(b.iter()).map(|(a, b)| a * b).sum::<f32>();
        assert_approx_eq::assert_approx_eq!(dot, expect, EPS);
        Ok(())
    }

    #[rstest::rstest]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_dot_c(backend: CUDABackend) -> Result<(), HoloError> {
        let a = make_random_cv(&backend, N)?;
        let b = make_random_cv(&backend, N)?;

        let dot = backend.dot_c(&a, &b)?;

        let a = backend.to_host_cv(a)?;
        let b = backend.to_host_cv(b)?;
        let expect = a
            .iter()
            .zip(b.iter())
            .map(|(a, b)| a.conj() * b)
            .sum::<Complex>();
        assert_approx_eq::assert_approx_eq!(dot.re, expect.re, EPS);
        assert_approx_eq::assert_approx_eq!(dot.im, expect.im, EPS);
        Ok(())
    }

    #[rstest::rstest]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_add_v(backend: CUDABackend) -> Result<(), HoloError> {
        let a = make_random_v(&backend, N)?;
        let mut b = make_random_v(&backend, N)?;
        let bc = backend.clone_v(&b)?;

        let mut rng = rand::rng();
        let alpha = rng.random();

        backend.add_v(alpha, &a, &mut b)?;

        let a = backend.to_host_v(a)?;
        let b = backend.to_host_v(b)?;
        let bc = backend.to_host_v(bc)?;
        b.iter()
            .zip(a.iter())
            .zip(bc.iter())
            .for_each(|((b, a), bc)| {
                assert_approx_eq::assert_approx_eq!(alpha * a + bc, b, EPS);
            });
        Ok(())
    }

    #[rstest::rstest]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_add_m(backend: CUDABackend) -> Result<(), HoloError> {
        let a = make_random_m(&backend, N, N)?;
        let mut b = make_random_m(&backend, N, N)?;
        let bc = backend.clone_m(&b)?;

        let mut rng = rand::rng();
        let alpha = rng.random();

        backend.add_m(alpha, &a, &mut b)?;

        let a = backend.to_host_m(a)?;
        let b = backend.to_host_m(b)?;
        let bc = backend.to_host_m(bc)?;
        b.iter()
            .zip(a.iter())
            .zip(bc.iter())
            .for_each(|((b, a), bc)| {
                assert_approx_eq::assert_approx_eq!(alpha * a + bc, b, EPS);
            });
        Ok(())
    }

    #[rstest::rstest]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_gevv_c(backend: CUDABackend) -> Result<(), HoloError> {
        let mut rng = rand::rng();

        {
            let a = make_random_cv(&backend, N)?;
            let b = make_random_cv(&backend, N)?;
            let mut c = make_random_cm(&backend, N, N)?;

            let alpha = Complex::new(rng.random(), rng.random());
            let beta = Complex::new(rng.random(), rng.random());
            assert!(
                backend
                    .gevv_c(Trans::NoTrans, Trans::NoTrans, alpha, &a, &b, beta, &mut c)
                    .is_err()
            );
        }

        {
            let a = make_random_cv(&backend, N)?;
            let b = make_random_cv(&backend, N)?;
            let mut c = make_random_cm(&backend, N, N)?;
            let cc = backend.clone_cm(&c)?;

            let alpha = Complex::new(rng.random(), rng.random());
            let beta = Complex::new(rng.random(), rng.random());
            backend.gevv_c(Trans::NoTrans, Trans::Trans, alpha, &a, &b, beta, &mut c)?;

            let a = backend.to_host_cv(a)?;
            let b = backend.to_host_cv(b)?;
            let c = backend.to_host_cm(c)?;
            let cc = backend.to_host_cm(cc)?;
            let expected = a * b.transpose() * alpha + cc * beta;
            c.iter().zip(expected.iter()).for_each(|(c, expected)| {
                assert_approx_eq::assert_approx_eq!(c.re, expected.re, EPS);
                assert_approx_eq::assert_approx_eq!(c.im, expected.im, EPS);
            });
        }

        {
            let a = make_random_cv(&backend, N)?;
            let b = make_random_cv(&backend, N)?;
            let mut c = make_random_cm(&backend, N, N)?;
            let cc = backend.clone_cm(&c)?;

            let alpha = Complex::new(rng.random(), rng.random());
            let beta = Complex::new(rng.random(), rng.random());
            backend.gevv_c(
                Trans::NoTrans,
                Trans::ConjTrans,
                alpha,
                &a,
                &b,
                beta,
                &mut c,
            )?;

            let a = backend.to_host_cv(a)?;
            let b = backend.to_host_cv(b)?;
            let c = backend.to_host_cm(c)?;
            let cc = backend.to_host_cm(cc)?;
            let expected = a * b.adjoint() * alpha + cc * beta;
            c.iter().zip(expected.iter()).for_each(|(c, expected)| {
                assert_approx_eq::assert_approx_eq!(c.re, expected.re, EPS);
                assert_approx_eq::assert_approx_eq!(c.im, expected.im, EPS);
            });
        }

        {
            let a = make_random_cv(&backend, N)?;
            let b = make_random_cv(&backend, N)?;
            let mut c = make_random_cm(&backend, 1, 1)?;
            let cc = backend.clone_cm(&c)?;

            let alpha = Complex::new(rng.random(), rng.random());
            let beta = Complex::new(rng.random(), rng.random());
            backend.gevv_c(Trans::Trans, Trans::NoTrans, alpha, &a, &b, beta, &mut c)?;

            let a = backend.to_host_cv(a)?;
            let b = backend.to_host_cv(b)?;
            let c = backend.to_host_cm(c)?;
            let cc = backend.to_host_cm(cc)?;
            let expected = a.transpose() * b * alpha + cc * beta;
            c.iter().zip(expected.iter()).for_each(|(c, expected)| {
                assert_approx_eq::assert_approx_eq!(c.re, expected.re, EPS);
                assert_approx_eq::assert_approx_eq!(c.im, expected.im, EPS);
            });
        }

        {
            let a = make_random_cv(&backend, N)?;
            let b = make_random_cv(&backend, N)?;
            let mut c = make_random_cm(&backend, N, N)?;

            let alpha = Complex::new(rng.random(), rng.random());
            let beta = Complex::new(rng.random(), rng.random());
            assert!(
                backend
                    .gevv_c(Trans::Trans, Trans::Trans, alpha, &a, &b, beta, &mut c)
                    .is_err()
            );
        }

        {
            let a = make_random_cv(&backend, N)?;
            let b = make_random_cv(&backend, N)?;
            let mut c = make_random_cm(&backend, N, N)?;

            let alpha = Complex::new(rng.random(), rng.random());
            let beta = Complex::new(rng.random(), rng.random());
            assert!(
                backend
                    .gevv_c(Trans::Trans, Trans::ConjTrans, alpha, &a, &b, beta, &mut c)
                    .is_err()
            );
        }

        {
            let a = make_random_cv(&backend, N)?;
            let b = make_random_cv(&backend, N)?;
            let mut c = make_random_cm(&backend, 1, 1)?;
            let cc = backend.clone_cm(&c)?;

            let alpha = Complex::new(rng.random(), rng.random());
            let beta = Complex::new(rng.random(), rng.random());
            backend.gevv_c(
                Trans::ConjTrans,
                Trans::NoTrans,
                alpha,
                &a,
                &b,
                beta,
                &mut c,
            )?;

            let a = backend.to_host_cv(a)?;
            let b = backend.to_host_cv(b)?;
            let c = backend.to_host_cm(c)?;
            let cc = backend.to_host_cm(cc)?;
            let expected = a.adjoint() * b * alpha + cc * beta;
            c.iter().zip(expected.iter()).for_each(|(c, expected)| {
                assert_approx_eq::assert_approx_eq!(c.re, expected.re, EPS);
                assert_approx_eq::assert_approx_eq!(c.im, expected.im, EPS);
            });
        }

        {
            let a = make_random_cv(&backend, N)?;
            let b = make_random_cv(&backend, N)?;
            let mut c = make_random_cm(&backend, N, N)?;

            let alpha = Complex::new(rng.random(), rng.random());
            let beta = Complex::new(rng.random(), rng.random());
            assert!(
                backend
                    .gevv_c(Trans::ConjTrans, Trans::Trans, alpha, &a, &b, beta, &mut c)
                    .is_err()
            );
        }

        {
            let a = make_random_cv(&backend, N)?;
            let b = make_random_cv(&backend, N)?;
            let mut c = make_random_cm(&backend, N, N)?;

            let alpha = Complex::new(rng.random(), rng.random());
            let beta = Complex::new(rng.random(), rng.random());
            assert!(
                backend
                    .gevv_c(
                        Trans::ConjTrans,
                        Trans::ConjTrans,
                        alpha,
                        &a,
                        &b,
                        beta,
                        &mut c,
                    )
                    .is_err()
            );
        }

        Ok(())
    }

    #[rstest::rstest]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_gemv_c(backend: CUDABackend) -> Result<(), HoloError> {
        let m = N;
        let n = 2 * N;

        let mut rng = rand::rng();

        {
            let a = make_random_cm(&backend, m, n)?;
            let b = make_random_cv(&backend, n)?;
            let mut c = make_random_cv(&backend, m)?;
            let cc = backend.clone_cv(&c)?;

            let alpha = Complex::new(rng.random(), rng.random());
            let beta = Complex::new(rng.random(), rng.random());
            backend.gemv_c(Trans::NoTrans, alpha, &a, &b, beta, &mut c)?;

            let a = backend.to_host_cm(a)?;
            let b = backend.to_host_cv(b)?;
            let c = backend.to_host_cv(c)?;
            let cc = backend.to_host_cv(cc)?;
            let expected = a * b * alpha + cc * beta;
            c.iter().zip(expected.iter()).for_each(|(c, expected)| {
                assert_approx_eq::assert_approx_eq!(c.re, expected.re, EPS);
                assert_approx_eq::assert_approx_eq!(c.im, expected.im, EPS);
            });
        }

        {
            let a = make_random_cm(&backend, n, m)?;
            let b = make_random_cv(&backend, n)?;
            let mut c = make_random_cv(&backend, m)?;
            let cc = backend.clone_cv(&c)?;

            let alpha = Complex::new(rng.random(), rng.random());
            let beta = Complex::new(rng.random(), rng.random());
            backend.gemv_c(Trans::Trans, alpha, &a, &b, beta, &mut c)?;

            let a = backend.to_host_cm(a)?;
            let b = backend.to_host_cv(b)?;
            let c = backend.to_host_cv(c)?;
            let cc = backend.to_host_cv(cc)?;
            let expected = a.transpose() * b * alpha + cc * beta;
            c.iter().zip(expected.iter()).for_each(|(c, expected)| {
                assert_approx_eq::assert_approx_eq!(c.re, expected.re, EPS);
                assert_approx_eq::assert_approx_eq!(c.im, expected.im, EPS);
            });
        }

        {
            let a = make_random_cm(&backend, n, m)?;
            let b = make_random_cv(&backend, n)?;
            let mut c = make_random_cv(&backend, m)?;
            let cc = backend.clone_cv(&c)?;

            let alpha = Complex::new(rng.random(), rng.random());
            let beta = Complex::new(rng.random(), rng.random());
            backend.gemv_c(Trans::ConjTrans, alpha, &a, &b, beta, &mut c)?;

            let a = backend.to_host_cm(a)?;
            let b = backend.to_host_cv(b)?;
            let c = backend.to_host_cv(c)?;
            let cc = backend.to_host_cv(cc)?;
            let expected = a.adjoint() * b * alpha + cc * beta;
            c.iter().zip(expected.iter()).for_each(|(c, expected)| {
                assert_approx_eq::assert_approx_eq!(c.re, expected.re, EPS);
                assert_approx_eq::assert_approx_eq!(c.im, expected.im, EPS);
            });
        }
        Ok(())
    }

    #[rstest::rstest]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_gemm_c(backend: CUDABackend) -> Result<(), HoloError> {
        let m = N;
        let n = 2 * N;
        let k = 3 * N;

        let mut rng = rand::rng();

        {
            let a = make_random_cm(&backend, m, k)?;
            let b = make_random_cm(&backend, k, n)?;
            let mut c = make_random_cm(&backend, m, n)?;
            let cc = backend.clone_cm(&c)?;

            let alpha = Complex::new(rng.random(), rng.random());
            let beta = Complex::new(rng.random(), rng.random());
            backend.gemm_c(Trans::NoTrans, Trans::NoTrans, alpha, &a, &b, beta, &mut c)?;

            let a = backend.to_host_cm(a)?;
            let b = backend.to_host_cm(b)?;
            let c = backend.to_host_cm(c)?;
            let cc = backend.to_host_cm(cc)?;
            let expected = a * b * alpha + cc * beta;
            c.iter().zip(expected.iter()).for_each(|(c, expected)| {
                assert_approx_eq::assert_approx_eq!(c.re, expected.re, EPS);
                assert_approx_eq::assert_approx_eq!(c.im, expected.im, EPS);
            });
        }

        {
            let a = make_random_cm(&backend, m, k)?;
            let b = make_random_cm(&backend, n, k)?;
            let mut c = make_random_cm(&backend, m, n)?;
            let cc = backend.clone_cm(&c)?;

            let alpha = Complex::new(rng.random(), rng.random());
            let beta = Complex::new(rng.random(), rng.random());
            backend.gemm_c(Trans::NoTrans, Trans::Trans, alpha, &a, &b, beta, &mut c)?;

            let a = backend.to_host_cm(a)?;
            let b = backend.to_host_cm(b)?;
            let c = backend.to_host_cm(c)?;
            let cc = backend.to_host_cm(cc)?;
            let expected = a * b.transpose() * alpha + cc * beta;
            c.iter().zip(expected.iter()).for_each(|(c, expected)| {
                assert_approx_eq::assert_approx_eq!(c.re, expected.re, EPS);
                assert_approx_eq::assert_approx_eq!(c.im, expected.im, EPS);
            });
        }

        {
            let a = make_random_cm(&backend, m, k)?;
            let b = make_random_cm(&backend, n, k)?;
            let mut c = make_random_cm(&backend, m, n)?;
            let cc = backend.clone_cm(&c)?;

            let alpha = Complex::new(rng.random(), rng.random());
            let beta = Complex::new(rng.random(), rng.random());
            backend.gemm_c(
                Trans::NoTrans,
                Trans::ConjTrans,
                alpha,
                &a,
                &b,
                beta,
                &mut c,
            )?;

            let a = backend.to_host_cm(a)?;
            let b = backend.to_host_cm(b)?;
            let c = backend.to_host_cm(c)?;
            let cc = backend.to_host_cm(cc)?;
            let expected = a * b.adjoint() * alpha + cc * beta;
            c.iter().zip(expected.iter()).for_each(|(c, expected)| {
                assert_approx_eq::assert_approx_eq!(c.re, expected.re, EPS);
                assert_approx_eq::assert_approx_eq!(c.im, expected.im, EPS);
            });
        }

        {
            let a = make_random_cm(&backend, k, m)?;
            let b = make_random_cm(&backend, k, n)?;
            let mut c = make_random_cm(&backend, m, n)?;
            let cc = backend.clone_cm(&c)?;

            let alpha = Complex::new(rng.random(), rng.random());
            let beta = Complex::new(rng.random(), rng.random());
            backend.gemm_c(Trans::Trans, Trans::NoTrans, alpha, &a, &b, beta, &mut c)?;

            let a = backend.to_host_cm(a)?;
            let b = backend.to_host_cm(b)?;
            let c = backend.to_host_cm(c)?;
            let cc = backend.to_host_cm(cc)?;
            let expected = a.transpose() * b * alpha + cc * beta;
            c.iter().zip(expected.iter()).for_each(|(c, expected)| {
                assert_approx_eq::assert_approx_eq!(c.re, expected.re, EPS);
                assert_approx_eq::assert_approx_eq!(c.im, expected.im, EPS);
            });
        }

        {
            let a = make_random_cm(&backend, k, m)?;
            let b = make_random_cm(&backend, n, k)?;
            let mut c = make_random_cm(&backend, m, n)?;
            let cc = backend.clone_cm(&c)?;

            let alpha = Complex::new(rng.random(), rng.random());
            let beta = Complex::new(rng.random(), rng.random());
            backend.gemm_c(Trans::Trans, Trans::Trans, alpha, &a, &b, beta, &mut c)?;

            let a = backend.to_host_cm(a)?;
            let b = backend.to_host_cm(b)?;
            let c = backend.to_host_cm(c)?;
            let cc = backend.to_host_cm(cc)?;
            let expected = a.transpose() * b.transpose() * alpha + cc * beta;
            c.iter().zip(expected.iter()).for_each(|(c, expected)| {
                assert_approx_eq::assert_approx_eq!(c.re, expected.re, EPS);
                assert_approx_eq::assert_approx_eq!(c.im, expected.im, EPS);
            });
        }

        {
            let a = make_random_cm(&backend, k, m)?;
            let b = make_random_cm(&backend, n, k)?;
            let mut c = make_random_cm(&backend, m, n)?;
            let cc = backend.clone_cm(&c)?;

            let alpha = Complex::new(rng.random(), rng.random());
            let beta = Complex::new(rng.random(), rng.random());
            backend.gemm_c(Trans::Trans, Trans::ConjTrans, alpha, &a, &b, beta, &mut c)?;

            let a = backend.to_host_cm(a)?;
            let b = backend.to_host_cm(b)?;
            let c = backend.to_host_cm(c)?;
            let cc = backend.to_host_cm(cc)?;
            let expected = a.transpose() * b.adjoint() * alpha + cc * beta;
            c.iter().zip(expected.iter()).for_each(|(c, expected)| {
                assert_approx_eq::assert_approx_eq!(c.re, expected.re, EPS);
                assert_approx_eq::assert_approx_eq!(c.im, expected.im, EPS);
            });
        }

        {
            let a = make_random_cm(&backend, k, m)?;
            let b = make_random_cm(&backend, k, n)?;
            let mut c = make_random_cm(&backend, m, n)?;
            let cc = backend.clone_cm(&c)?;

            let alpha = Complex::new(rng.random(), rng.random());
            let beta = Complex::new(rng.random(), rng.random());
            backend.gemm_c(
                Trans::ConjTrans,
                Trans::NoTrans,
                alpha,
                &a,
                &b,
                beta,
                &mut c,
            )?;

            let a = backend.to_host_cm(a)?;
            let b = backend.to_host_cm(b)?;
            let c = backend.to_host_cm(c)?;
            let cc = backend.to_host_cm(cc)?;
            let expected = a.adjoint() * b * alpha + cc * beta;
            c.iter().zip(expected.iter()).for_each(|(c, expected)| {
                assert_approx_eq::assert_approx_eq!(c.re, expected.re, EPS);
                assert_approx_eq::assert_approx_eq!(c.im, expected.im, EPS);
            });
        }

        {
            let a = make_random_cm(&backend, k, m)?;
            let b = make_random_cm(&backend, n, k)?;
            let mut c = make_random_cm(&backend, m, n)?;
            let cc = backend.clone_cm(&c)?;

            let alpha = Complex::new(rng.random(), rng.random());
            let beta = Complex::new(rng.random(), rng.random());
            backend.gemm_c(Trans::ConjTrans, Trans::Trans, alpha, &a, &b, beta, &mut c)?;

            let a = backend.to_host_cm(a)?;
            let b = backend.to_host_cm(b)?;
            let c = backend.to_host_cm(c)?;
            let cc = backend.to_host_cm(cc)?;
            let expected = a.adjoint() * b.transpose() * alpha + cc * beta;
            c.iter().zip(expected.iter()).for_each(|(c, expected)| {
                assert_approx_eq::assert_approx_eq!(c.re, expected.re, EPS);
                assert_approx_eq::assert_approx_eq!(c.im, expected.im, EPS);
            });
        }

        {
            let a = make_random_cm(&backend, k, m)?;
            let b = make_random_cm(&backend, n, k)?;
            let mut c = make_random_cm(&backend, m, n)?;
            let cc = backend.clone_cm(&c)?;

            let alpha = Complex::new(rng.random(), rng.random());
            let beta = Complex::new(rng.random(), rng.random());
            backend.gemm_c(
                Trans::ConjTrans,
                Trans::ConjTrans,
                alpha,
                &a,
                &b,
                beta,
                &mut c,
            )?;

            let a = backend.to_host_cm(a)?;
            let b = backend.to_host_cm(b)?;
            let c = backend.to_host_cm(c)?;
            let cc = backend.to_host_cm(cc)?;
            let expected = a.adjoint() * b.adjoint() * alpha + cc * beta;
            c.iter().zip(expected.iter()).for_each(|(c, expected)| {
                assert_approx_eq::assert_approx_eq!(c.re, expected.re, EPS);
                assert_approx_eq::assert_approx_eq!(c.im, expected.im, EPS);
            });
        }
        Ok(())
    }

    #[rstest::rstest]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_solve_inplace(backend: CUDABackend) -> Result<(), HoloError> {
        {
            let tmp = make_random_m(&backend, N, N)?;
            let tmp = backend.to_host_m(tmp)?;

            let a = &tmp * tmp.adjoint();

            let mut rng = rand::rng();
            let x = VectorX::from_iterator(N, (0..N).map(|_| rng.random()));

            let b = &a * &x;

            let aa = backend.from_slice_m(N, N, a.as_slice())?;
            let mut bb = backend.from_slice_v(b.as_slice())?;

            backend.solve_inplace(&aa, &mut bb)?;

            let b2 = &a * backend.to_host_v(bb)?;
            assert!(approx::relative_eq!(b, b2, epsilon = 1e-3));
        }

        Ok(())
    }

    #[rstest::rstest]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_reduce_col(backend: CUDABackend) -> Result<(), HoloError> {
        let a = make_random_m(&backend, N, N)?;

        let mut b = backend.alloc_v(N)?;

        backend.reduce_col(&a, &mut b)?;

        let a = backend.to_host_m(a)?;
        let b = backend.to_host_v(b)?;

        (0..N).for_each(|row| {
            let sum = a.row(row).iter().sum::<f32>();
            assert_approx_eq::assert_approx_eq!(sum, b[row], EPS);
        });
        Ok(())
    }

    #[rstest::rstest]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_scaled_to_cv(backend: CUDABackend) -> Result<(), HoloError> {
        let a = make_random_cv(&backend, N)?;
        let b = make_random_cv(&backend, N)?;
        let mut c = backend.alloc_cv(N)?;

        backend.scaled_to_cv(&a, &b, &mut c)?;

        let a = backend.to_host_cv(a)?;
        let b = backend.to_host_cv(b)?;
        let c = backend.to_host_cv(c)?;
        c.iter()
            .zip(a.iter())
            .zip(b.iter())
            .for_each(|((&c, &a), &b)| {
                assert_approx_eq::assert_approx_eq!(c, a / a.abs() * b, EPS);
            });

        Ok(())
    }

    #[rstest::rstest]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_scaled_to_assign_cv(backend: CUDABackend) -> Result<(), HoloError> {
        let a = make_random_cv(&backend, N)?;
        let mut b = make_random_cv(&backend, N)?;
        let bc = backend.clone_cv(&b)?;

        backend.scaled_to_assign_cv(&a, &mut b)?;

        let a = backend.to_host_cv(a)?;
        let b = backend.to_host_cv(b)?;
        let bc = backend.to_host_cv(bc)?;
        b.iter()
            .zip(a.iter())
            .zip(bc.iter())
            .for_each(|((&b, &a), &bc)| {
                assert_approx_eq::assert_approx_eq!(b, bc / bc.abs() * a, EPS);
            });

        Ok(())
    }

    #[rstest::rstest]
    #[test]
    #[case(1, 2)]
    #[case(2, 1)]
    fn test_generate_propagation_matrix(
        #[case] dev_num: usize,
        #[case] foci_num: usize,
        backend: CUDABackend,
    ) -> Result<(), HoloError> {
        let reference = |geometry: Geometry, foci: Vec<Point3>| {
            let mut g = MatrixXc::zeros(
                foci.len(),
                geometry
                    .iter()
                    .map(|dev| dev.num_transducers())
                    .sum::<usize>(),
            );
            let transducers = geometry
                .iter()
                .flat_map(|dev| dev.iter().map(|tr| (dev.idx(), tr)))
                .collect::<Vec<_>>();
            (0..foci.len()).for_each(|i| {
                (0..transducers.len()).for_each(|j| {
                    g[(i, j)] = propagate::<Sphere>(
                        transducers[j].1,
                        geometry[transducers[j].0].wavenumber(),
                        geometry[transducers[j].0].axial_direction(),
                        &foci[i],
                    )
                })
            });
            g
        };

        let geometry = generate_geometry(dev_num);
        let foci = gen_foci(foci_num).map(|(p, _)| p).collect::<Vec<_>>();

        let g = backend.generate_propagation_matrix(&geometry, &foci, None)?;
        let g = backend.to_host_cm(g)?;
        reference(geometry, foci)
            .iter()
            .zip(g.iter())
            .for_each(|(r, g)| {
                assert_approx_eq::assert_approx_eq!(r.re, g.re, EPS);
                assert_approx_eq::assert_approx_eq!(r.im, g.im, EPS);
            });

        Ok(())
    }

    #[rstest::rstest]
    #[test]
    #[case(1, 2)]
    #[case(2, 1)]
    fn test_generate_propagation_matrix_with_filter(
        #[case] dev_num: usize,
        #[case] foci_num: usize,
        backend: CUDABackend,
    ) -> Result<(), HoloError> {
        use std::collections::HashMap;

        let filter = |geometry: &Geometry| {
            geometry
                .iter()
                .map(|dev| {
                    let mut filter = BitVec::new();
                    dev.iter().for_each(|tr| {
                        filter.push(tr.idx() > dev.num_transducers() / 2);
                    });
                    (dev.idx(), filter)
                })
                .collect::<HashMap<_, _>>()
        };

        let reference = |geometry, foci: Vec<Point3>| {
            let filter = filter(&geometry);
            let transducers = geometry
                .iter()
                .flat_map(|dev| {
                    dev.iter().filter_map(|tr| {
                        if filter[&dev.idx()][tr.idx()] {
                            Some((dev.idx(), tr))
                        } else {
                            None
                        }
                    })
                })
                .collect::<Vec<_>>();

            let mut g = MatrixXc::zeros(foci.len(), transducers.len());
            (0..foci.len()).for_each(|i| {
                (0..transducers.len()).for_each(|j| {
                    g[(i, j)] = propagate::<Sphere>(
                        transducers[j].1,
                        geometry[transducers[j].0].wavenumber(),
                        geometry[transducers[j].0].axial_direction(),
                        &foci[i],
                    )
                })
            });
            g
        };

        let geometry = generate_geometry(dev_num);
        let foci = gen_foci(foci_num).map(|(p, _)| p).collect::<Vec<_>>();
        let filter = filter(&geometry);

        let g = backend.generate_propagation_matrix(&geometry, &foci, Some(&filter))?;
        let g = backend.to_host_cm(g)?;
        assert_eq!(g.nrows(), foci.len());
        assert_eq!(
            g.ncols(),
            geometry
                .iter()
                .map(|dev| dev.num_transducers() / 2)
                .sum::<usize>()
        );
        reference(geometry, foci)
            .iter()
            .zip(g.iter())
            .for_each(|(r, g)| {
                assert_approx_eq::assert_approx_eq!(r.re, g.re, EPS);
                assert_approx_eq::assert_approx_eq!(r.im, g.im, EPS);
            });

        Ok(())
    }

    #[rstest::rstest]
    #[test]
    fn test_gen_back_prop(backend: CUDABackend) -> Result<(), HoloError> {
        let geometry = generate_geometry(1);
        let foci = gen_foci(1).map(|(p, _)| p).collect::<Vec<_>>();

        let m = geometry
            .iter()
            .map(|dev| dev.num_transducers())
            .sum::<usize>();
        let n = foci.len();

        let g = backend.generate_propagation_matrix(&geometry, &foci, None)?;

        let b = backend.gen_back_prop(m, n, &g)?;
        let g = backend.to_host_cm(g)?;
        let reference = {
            let mut b = MatrixXc::zeros(m, n);
            (0..n).for_each(|i| {
                let x = 1.0 / g.rows(i, 1).iter().map(|x| x.norm_sqr()).sum::<f32>();
                (0..m).for_each(|j| {
                    b[(j, i)] = g[(i, j)].conj() * x;
                })
            });
            b
        };

        let b = backend.to_host_cm(b)?;
        reference.iter().zip(b.iter()).for_each(|(r, b)| {
            assert_approx_eq::assert_approx_eq!(r.re, b.re, EPS);
            assert_approx_eq::assert_approx_eq!(r.im, b.im, EPS);
        });
        Ok(())
    }
}
