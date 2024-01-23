#[cfg(feature = "test-utilities")]
criterion::criterion_group!(
    benches,
    autd3_gain_holo::test_utilities::bench_utils::foci::<autd3_backend_cuda::CUDABackend, 4>,
    autd3_gain_holo::test_utilities::bench_utils::devices::<autd3_backend_cuda::CUDABackend, 2>
);
#[cfg(feature = "test-utilities")]
criterion::criterion_main!(benches);

#[cfg(not(feature = "test-utilities"))]
fn main() {}
