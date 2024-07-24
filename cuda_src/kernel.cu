#include <cuComplex.h>

#include <cstdint>

#define PI (3.141592653589793238462643)
#ifdef AUTD3_USE_METER
#define MILLIMETER (0.001)
#else
#define MILLIMETER (1.0)
#endif
#define T4010A1_AMP (275.574246625 * 200.0 / 4 / PI * MILLIMETER)

__device__ float absc2(const cuFloatComplex x) { return x.x * x.x + x.y * x.y; }
__device__ float absc(const cuFloatComplex x) { return sqrt(absc2(x)); }
__device__ cuFloatComplex conj(const cuFloatComplex a) {
  return make_cuFloatComplex(a.x, -a.y);
}
__device__ cuFloatComplex mulc(const cuFloatComplex a, const cuFloatComplex b) {
  return make_cuFloatComplex(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}
__device__ cuFloatComplex mulcr(const cuFloatComplex a, const float b) {
  return make_cuFloatComplex(a.x * b, a.y * b);
}

__device__ cuFloatComplex divcr(const cuFloatComplex x, const float y) {
  const float r = x.x / y;
  const float i = x.y / y;
  return make_cuFloatComplex(r, i);
}

__global__ void scaled_to_kernel(const cuFloatComplex *a,
                                 const cuFloatComplex *b, uint32_t row,
                                 uint32_t col, cuFloatComplex *c) {
  unsigned int xi = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int yi = blockIdx.y * blockDim.y + threadIdx.y;
  if (xi >= col || yi >= row) return;
  unsigned int i = yi + xi * row;
  c[i] = mulc(divcr(a[i], absc(a[i])), b[i]);
}

__global__ void get_diagonal_kernel(const cuFloatComplex *a, uint32_t row,
                                    uint32_t col, cuFloatComplex *b) {
  unsigned int xi = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int yi = blockIdx.y * blockDim.y + threadIdx.y;
  if (xi >= col || yi >= row) return;

  if (xi == yi) {
    unsigned int idx = yi + xi * row;
    b[xi] = a[idx];
  }
}

__global__ void get_diagonal_kernel(const float *a, uint32_t row, uint32_t col,
                                    float *b) {
  unsigned int xi = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int yi = blockIdx.y * blockDim.y + threadIdx.y;
  if (xi >= col || yi >= row) return;

  if (xi == yi) {
    unsigned int idx = yi + xi * row;
    b[xi] = a[idx];
  }
}

__global__ void set_diagonal_kernel_c(const cuFloatComplex *a, uint32_t n,
                                      cuFloatComplex *b) {
  unsigned int xi = blockIdx.x * blockDim.x + threadIdx.x;
  if (xi >= n) return;
  unsigned int idx = xi + xi * n;
  b[idx] = a[xi];
}

__global__ void set_diagonal_kernel(const float *a, uint32_t n, float *b) {
  unsigned int xi = blockIdx.x * blockDim.x + threadIdx.x;
  if (xi >= n) return;
  unsigned int idx = xi + xi * n;
  b[idx] = a[xi];
}

__global__ void reciprocal_kernel(const cuFloatComplex *a, const uint32_t row,
                                  const uint32_t col, cuFloatComplex *b) {
  unsigned int xi = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int yi = blockIdx.y * blockDim.y + threadIdx.y;
  if (xi >= col || yi >= row) return;

  unsigned int idx = yi + xi * row;
  float s = absc2(a[idx]);
  const float x = a[idx].x / s;
  const float y = -a[idx].y / s;
  b[idx] = make_cuFloatComplex(x, y);
}

__global__ void hadamard_product_kernel(const cuFloatComplex *a,
                                        const cuFloatComplex *b,
                                        const uint32_t row, const uint32_t col,
                                        cuFloatComplex *c) {
  unsigned int xi = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int yi = blockIdx.y * blockDim.y + threadIdx.y;
  if (xi >= col || yi >= row) return;

  unsigned int idx = yi + xi * row;
  c[idx] = mulc(a[idx], b[idx]);
}

__global__ void norm_squared_kernel(const cuFloatComplex *a, const uint32_t row,
                                    const uint32_t col, float *b) {
  unsigned int xi = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int yi = blockIdx.y * blockDim.y + threadIdx.y;
  if (xi >= col || yi >= row) return;

  unsigned int idx = yi + xi * row;
  b[idx] = absc2(a[idx]);
}

__global__ void make_complex_kernel(const float *re, const uint32_t row,
                                    const uint32_t col, cuFloatComplex *dst) {
  unsigned int xi = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int yi = blockIdx.y * blockDim.y + threadIdx.y;
  if (xi >= col || yi >= row) return;

  unsigned int idx = yi + xi * row;
  dst[idx] = make_cuFloatComplex(re[idx], 0);
}

__global__ void make_complex2_kernel(const float *re, const float *im,
                                     const uint32_t row, const uint32_t col,
                                     cuFloatComplex *dst) {
  unsigned int xi = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int yi = blockIdx.y * blockDim.y + threadIdx.y;
  if (xi >= col || yi >= row) return;

  unsigned int idx = yi + xi * row;
  dst[idx] = make_cuFloatComplex(re[idx], im[idx]);
}

__global__ void conj_kernel(const cuFloatComplex *a, const uint32_t row,
                            const uint32_t col, cuFloatComplex *b) {
  unsigned int xi = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int yi = blockIdx.y * blockDim.y + threadIdx.y;
  if (xi >= col || yi >= row) return;

  unsigned int idx = yi + xi * row;
  b[idx] = conj(a[idx]);
}

__device__ cuFloatComplex expc(const cuFloatComplex x) {
  const float s = exp(x.x);
  const float r = cos(x.y);
  const float i = sin(x.y);
  return make_cuFloatComplex(s * r, s * i);
}

__global__ void exp_kernel(const cuFloatComplex *a, const uint32_t row,
                           const uint32_t col, cuFloatComplex *b) {
  unsigned int xi = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int yi = blockIdx.y * blockDim.y + threadIdx.y;
  if (xi >= col || yi >= row) return;

  unsigned int idx = yi + xi * row;
  b[idx] = expc(a[idx]);
}

__global__ void real_kernel(const cuFloatComplex *src, const uint32_t row,
                            const uint32_t col, float *dst) {
  unsigned int xi = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int yi = blockIdx.y * blockDim.y + threadIdx.y;
  if (xi >= col || yi >= row) return;

  unsigned int idx = yi + xi * row;
  dst[idx] = src[idx].x;
}

__global__ void imag_kernel(const cuFloatComplex *src, const uint32_t row,
                            const uint32_t col, float *dst) {
  unsigned int xi = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int yi = blockIdx.y * blockDim.y + threadIdx.y;
  if (xi >= col || yi >= row) return;

  unsigned int idx = yi + xi * row;
  dst[idx] = src[idx].y;
}

__global__ void col_sum_kernel(const float *din, uint32_t m, uint32_t n,
                               float *dout) {
  uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= m) return;
  float sum = 0;
  for (uint32_t col = 0; col < n; col++) sum += din[col * m + row];
  dout[row] = sum;
}

__global__ void generate_propagation_matrix_kernel(
    const float *positions, const float *foci, const float *wavenums,
    const uint32_t row, const uint32_t col, cuFloatComplex *dst) {
  unsigned int xi = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int yi = blockIdx.y * blockDim.y + threadIdx.y;
  if (xi >= col || yi >= row) return;

  float xd = foci[3 * yi] - positions[3 * xi];
  float yd = foci[3 * yi + 1] - positions[3 * xi + 1];
  float zd = foci[3 * yi + 2] - positions[3 * xi + 2];
  float dist = sqrt(xd * xd + yd * yd + zd * zd);
  float r = T4010A1_AMP / dist;
  float phase = wavenums[xi] * dist;
  dst[yi + xi * row] = make_cuFloatComplex(r * cos(phase), r * sin(phase));
}

#ifdef __cplusplus
extern "C" {
#endif

#define BLOCK_SIZE (32)

void cu_scaled_to(const cuFloatComplex *a, const cuFloatComplex *b,
                  const uint32_t row, const uint32_t col, cuFloatComplex *c) {
  dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
  dim3 grid((col - 1) / BLOCK_SIZE + 1, (row - 1) / BLOCK_SIZE + 1, 1);
  scaled_to_kernel<<<grid, block>>>(a, b, row, col, c);
}

void cu_get_diagonal(const float *a, const uint32_t row, const uint32_t col,
                     float *b) {
  dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
  dim3 grid((col - 1) / BLOCK_SIZE + 1, (row - 1) / BLOCK_SIZE + 1, 1);
  get_diagonal_kernel<<<grid, block>>>(a, row, col, b);
}

void cu_get_diagonal_c(const cuFloatComplex *a, const uint32_t row,
                       const uint32_t col, cuFloatComplex *b) {
  dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
  dim3 grid((col - 1) / BLOCK_SIZE + 1, (row - 1) / BLOCK_SIZE + 1, 1);
  get_diagonal_kernel<<<grid, block>>>(a, row, col, b);
}

void cu_set_diagonal(const float *a, const uint32_t n, float *b) {
  dim3 block(BLOCK_SIZE * BLOCK_SIZE, 1, 1);
  dim3 grid((n - 1) / (BLOCK_SIZE * BLOCK_SIZE) + 1, 1, 1);
  set_diagonal_kernel<<<grid, block>>>(a, n, b);
}

void cu_set_diagonal_c(const cuFloatComplex *a, const uint32_t n,
                       cuFloatComplex *b) {
  dim3 block(BLOCK_SIZE * BLOCK_SIZE, 1, 1);
  dim3 grid((n - 1) / (BLOCK_SIZE * BLOCK_SIZE) + 1, 1, 1);
  set_diagonal_kernel_c<<<grid, block>>>(a, n, b);
}

void cu_reciprocal(const cuFloatComplex *a, const uint32_t row,
                   const uint32_t col, cuFloatComplex *b) {
  dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
  dim3 grid((col - 1) / BLOCK_SIZE + 1, (row - 1) / BLOCK_SIZE + 1, 1);
  reciprocal_kernel<<<grid, block>>>(a, row, col, b);
}

void cu_hadamard_product(const cuFloatComplex *a, const cuFloatComplex *b,
                         const uint32_t row, const uint32_t col,
                         cuFloatComplex *c) {
  dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
  dim3 grid((col - 1) / BLOCK_SIZE + 1, (row - 1) / BLOCK_SIZE + 1, 1);
  hadamard_product_kernel<<<grid, block>>>(a, b, row, col, c);
}

void cu_norm_squared(const cuFloatComplex *a, const uint32_t row,
                     const uint32_t col, float *b) {
  dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
  dim3 grid((col - 1) / BLOCK_SIZE + 1, (row - 1) / BLOCK_SIZE + 1, 1);
  norm_squared_kernel<<<grid, block>>>(a, row, col, b);
}

void cu_make_complex(const float *re, const uint32_t row, const uint32_t col,
                     cuFloatComplex *dst) {
  dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
  dim3 grid((col - 1) / BLOCK_SIZE + 1, (row - 1) / BLOCK_SIZE + 1, 1);
  make_complex_kernel<<<grid, block>>>(re, row, col, dst);
}

void cu_make_complex2(const float *re, const float *im, const uint32_t row,
                      const uint32_t col, cuFloatComplex *dst) {
  dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
  dim3 grid((col - 1) / BLOCK_SIZE + 1, (row - 1) / BLOCK_SIZE + 1, 1);
  make_complex2_kernel<<<grid, block>>>(re, im, row, col, dst);
}

void cu_conj(const cuFloatComplex *a, const uint32_t row, const uint32_t col,
             cuFloatComplex *b) {
  dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
  dim3 grid((col - 1) / BLOCK_SIZE + 1, (row - 1) / BLOCK_SIZE + 1, 1);
  conj_kernel<<<grid, block>>>(a, row, col, b);
}

void cu_exp(const cuFloatComplex *a, const uint32_t row, const uint32_t col,
            cuFloatComplex *b) {
  dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
  dim3 grid((col - 1) / BLOCK_SIZE + 1, (row - 1) / BLOCK_SIZE + 1, 1);
  exp_kernel<<<grid, block>>>(a, row, col, b);
}

void cu_real(const cuFloatComplex *src, const uint32_t row, const uint32_t col,
             float *dst) {
  dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
  dim3 grid((col - 1) / BLOCK_SIZE + 1, (row - 1) / BLOCK_SIZE + 1, 1);
  real_kernel<<<grid, block>>>(src, row, col, dst);
}

void cu_imag(const cuFloatComplex *src, const uint32_t row, const uint32_t col,
             float *dst) {
  dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
  dim3 grid((col - 1) / BLOCK_SIZE + 1, (row - 1) / BLOCK_SIZE + 1, 1);
  imag_kernel<<<grid, block>>>(src, row, col, dst);
}

void cu_reduce_col(const float *mat, const uint32_t m, const uint32_t n,
                   float *result) {
  dim3 block(1, BLOCK_SIZE * BLOCK_SIZE, 1);
  dim3 grid(1, (m - 1) / (BLOCK_SIZE * BLOCK_SIZE) + 1, 1);
  col_sum_kernel<<<grid, block>>>(mat, m, n, result);
}

void cu_generate_propagation_matrix(const float *positions, const float *foci,
                                    const float *wavenums, const uint32_t row,
                                    const uint32_t col, cuFloatComplex *dst) {
  dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
  dim3 grid((col - 1) / BLOCK_SIZE + 1, (row - 1) / BLOCK_SIZE + 1, 1);
  generate_propagation_matrix_kernel<<<grid, block>>>(positions, foci, wavenums,
                                                      row, col, dst);
}

#ifdef __cplusplus
}
#endif
