#include <cuComplex.h>

#include <cstdint>

#define PI (3.141592653589793238462643)
#ifdef AUTD3_USE_METER
#define MILLIMETER (0.001)
#else
#define MILLIMETER (1.0)
#endif
#define T4010A1_AMP (275.574246625 * 200.0 / 4 / PI * MILLIMETER)

__device__ double absc2(const cuDoubleComplex x) { return x.x * x.x + x.y * x.y; }
__device__ double absc(const cuDoubleComplex x) { return sqrt(absc2(x)); }
__device__ cuDoubleComplex conj(const cuDoubleComplex a) { return make_cuDoubleComplex(a.x, -a.y); }
__device__ cuDoubleComplex mulc(const cuDoubleComplex a, const cuDoubleComplex b) {
  return make_cuDoubleComplex(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}
__device__ cuDoubleComplex mulcr(const cuDoubleComplex a, const double b) { return make_cuDoubleComplex(a.x * b, a.y * b); }

__device__ cuDoubleComplex divcr(const cuDoubleComplex x, const double y) {
  const double r = x.x / y;
  const double i = x.y / y;
  return make_cuDoubleComplex(r, i);
}

__global__ void normalize_kernel(const cuDoubleComplex *x, uint32_t row, uint32_t col, cuDoubleComplex *y) {
  unsigned int xi = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int yi = blockIdx.y * blockDim.y + threadIdx.y;
  if (xi >= col || yi >= row) return;
  unsigned int i = yi + xi * row;
  y[i] = divcr(x[i], absc(x[i]));
}

__global__ void scaled_to_kernel(const cuDoubleComplex *a, const cuDoubleComplex *b, uint32_t row, uint32_t col, cuDoubleComplex *c) {
  unsigned int xi = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int yi = blockIdx.y * blockDim.y + threadIdx.y;
  if (xi >= col || yi >= row) return;
  unsigned int i = yi + xi * row;
  c[i] = mulc(divcr(a[i], absc(a[i])), b[i]);
}

__global__ void get_diagonal_kernel(const cuDoubleComplex *a, uint32_t row, uint32_t col, cuDoubleComplex *b) {
  unsigned int xi = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int yi = blockIdx.y * blockDim.y + threadIdx.y;
  if (xi >= col || yi >= row) return;

  if (xi == yi) {
    unsigned int idx = yi + xi * row;
    b[xi] = a[idx];
  }
}

__global__ void get_diagonal_kernel(const double *a, uint32_t row, uint32_t col, double *b) {
  unsigned int xi = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int yi = blockIdx.y * blockDim.y + threadIdx.y;
  if (xi >= col || yi >= row) return;

  if (xi == yi) {
    unsigned int idx = yi + xi * row;
    b[xi] = a[idx];
  }
}

__global__ void set_diagonal_kernel_c(const cuDoubleComplex *a, uint32_t n, cuDoubleComplex *b) {
  unsigned int xi = blockIdx.x * blockDim.x + threadIdx.x;
  if (xi >= n) return;
  unsigned int idx = xi + xi * n;
  b[idx] = a[xi];
}

__global__ void set_diagonal_kernel(const double *a, uint32_t n, double *b) {
  unsigned int xi = blockIdx.x * blockDim.x + threadIdx.x;
  if (xi >= n) return;
  unsigned int idx = xi + xi * n;
  b[idx] = a[xi];
}

__global__ void reciprocal_kernel(const cuDoubleComplex *a, const uint32_t row, const uint32_t col, cuDoubleComplex *b) {
  unsigned int xi = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int yi = blockIdx.y * blockDim.y + threadIdx.y;
  if (xi >= col || yi >= row) return;

  unsigned int idx = yi + xi * row;
  double s = absc2(a[idx]);
  const double x = a[idx].x / s;
  const double y = -a[idx].y / s;
  b[idx] = make_cuDoubleComplex(x, y);
}

__global__ void hadamard_product_kernel(const cuDoubleComplex *a, const cuDoubleComplex *b, const uint32_t row, const uint32_t col,
                                        cuDoubleComplex *c) {
  unsigned int xi = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int yi = blockIdx.y * blockDim.y + threadIdx.y;
  if (xi >= col || yi >= row) return;

  unsigned int idx = yi + xi * row;
  c[idx] = mulc(a[idx], b[idx]);
}

__global__ void abs_kernel(const cuDoubleComplex *a, const uint32_t row, const uint32_t col, double *b) {
  unsigned int xi = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int yi = blockIdx.y * blockDim.y + threadIdx.y;
  if (xi >= col || yi >= row) return;

  unsigned int idx = yi + xi * row;
  b[idx] = absc(a[idx]);
}

__global__ void sqrt_kernel(const double *a, const uint32_t row, const uint32_t col, double *b) {
  unsigned int xi = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int yi = blockIdx.y * blockDim.y + threadIdx.y;
  if (xi >= col || yi >= row) return;

  unsigned int idx = yi + xi * row;
  b[idx] = sqrt(a[idx]);
}

__global__ void make_complex_kernel(const double *re, const uint32_t row, const uint32_t col, cuDoubleComplex *dst) {
  unsigned int xi = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int yi = blockIdx.y * blockDim.y + threadIdx.y;
  if (xi >= col || yi >= row) return;

  unsigned int idx = yi + xi * row;
  dst[idx] = make_cuDoubleComplex(re[idx], 0);
}

__global__ void make_complex2_kernel(const double *re, const double *im, const uint32_t row, const uint32_t col, cuDoubleComplex *dst) {
  unsigned int xi = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int yi = blockIdx.y * blockDim.y + threadIdx.y;
  if (xi >= col || yi >= row) return;

  unsigned int idx = yi + xi * row;
  dst[idx] = make_cuDoubleComplex(re[idx], im[idx]);
}

__global__ void pow_kernel(const double *a, const double p, const uint32_t row, const uint32_t col, double *b) {
  unsigned int xi = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int yi = blockIdx.y * blockDim.y + threadIdx.y;
  if (xi >= col || yi >= row) return;

  unsigned int idx = yi + xi * row;
  b[idx] = pow(a[idx], p);
}

__global__ void conj_kernel(const cuDoubleComplex *a, const uint32_t row, const uint32_t col, cuDoubleComplex *b) {
  unsigned int xi = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int yi = blockIdx.y * blockDim.y + threadIdx.y;
  if (xi >= col || yi >= row) return;

  unsigned int idx = yi + xi * row;
  b[idx] = conj(a[idx]);
}

__global__ void calc_singular_inv_kernel(double *d_s, uint32_t row, uint32_t col, double alpha, cuDoubleComplex *p_singular_inv) {
  unsigned int xi = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int yi = blockIdx.y * blockDim.y + threadIdx.y;
  if (xi >= col || yi >= row) return;

  if (xi == yi)
    p_singular_inv[yi + xi * row] = make_cuDoubleComplex(d_s[xi] / (d_s[xi] * d_s[xi] + alpha), 0.0);
  else
    p_singular_inv[yi + xi * row] = make_cuDoubleComplex(0.0, 0.0);
}

__device__ cuDoubleComplex expc(const cuDoubleComplex x) {
  const double s = exp(x.x);
  const double r = cos(x.y);
  const double i = sin(x.y);
  return make_cuDoubleComplex(s * r, s * i);
}

__global__ void exp_kernel(const cuDoubleComplex *a, const uint32_t row, const uint32_t col, cuDoubleComplex *b) {
  unsigned int xi = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int yi = blockIdx.y * blockDim.y + threadIdx.y;
  if (xi >= col || yi >= row) return;

  unsigned int idx = yi + xi * row;
  b[idx] = expc(a[idx]);
}

__global__ void real_kernel(const cuDoubleComplex *src, const uint32_t row, const uint32_t col, double *dst) {
  unsigned int xi = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int yi = blockIdx.y * blockDim.y + threadIdx.y;
  if (xi >= col || yi >= row) return;

  unsigned int idx = yi + xi * row;
  dst[idx] = src[idx].x;
}

__global__ void imag_kernel(const cuDoubleComplex *src, const uint32_t row, const uint32_t col, double *dst) {
  unsigned int xi = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int yi = blockIdx.y * blockDim.y + threadIdx.y;
  if (xi >= col || yi >= row) return;

  unsigned int idx = yi + xi * row;
  dst[idx] = src[idx].y;
}

__global__ void col_sum_kernel(const double *din, uint32_t m, uint32_t n, double *dout) {
  uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= m) return;
  double sum = 0;
  for (uint32_t col = 0; col < n; col++) sum += din[col * m + row];
  dout[row] = sum;
}

__global__ void generate_propagation_matrix_kernel(const double *positions, const double *foci, const double *wavenums, const double *attens,
                                                   const uint32_t row, const uint32_t col, cuDoubleComplex *dst) {
  unsigned int xi = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int yi = blockIdx.y * blockDim.y + threadIdx.y;
  if (xi >= col || yi >= row) return;

  double xd = foci[3 * yi] - positions[3 * xi];
  double yd = foci[3 * yi + 1] - positions[3 * xi + 1];
  double zd = foci[3 * yi + 2] - positions[3 * xi + 2];
  double dist = sqrt(xd * xd + yd * yd + zd * zd);
  double r = T4010A1_AMP * exp(-dist * attens[xi]) / dist;
  double phase = -wavenums[xi] * dist;
  dst[yi + xi * row] = make_cuDoubleComplex(r * cos(phase), r * sin(phase));
}

#ifdef __cplusplus
extern "C" {
#endif

#define BLOCK_SIZE (32)

void cu_normalize(const cuDoubleComplex *x, const uint32_t row, const uint32_t col, cuDoubleComplex *y) {
  dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
  dim3 grid((col - 1) / BLOCK_SIZE + 1, (row - 1) / BLOCK_SIZE + 1, 1);
  normalize_kernel<<<grid, block>>>(x, row, col, y);
}

void cu_scaled_to(const cuDoubleComplex *a, const cuDoubleComplex *b, const uint32_t row, const uint32_t col, cuDoubleComplex *c) {
  dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
  dim3 grid((col - 1) / BLOCK_SIZE + 1, (row - 1) / BLOCK_SIZE + 1, 1);
  scaled_to_kernel<<<grid, block>>>(a, b, row, col, c);
}

void cu_get_diagonal(const double *a, const uint32_t row, const uint32_t col, double *b) {
  dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
  dim3 grid((col - 1) / BLOCK_SIZE + 1, (row - 1) / BLOCK_SIZE + 1, 1);
  get_diagonal_kernel<<<grid, block>>>(a, row, col, b);
}

void cu_get_diagonal_c(const cuDoubleComplex *a, const uint32_t row, const uint32_t col, cuDoubleComplex *b) {
  dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
  dim3 grid((col - 1) / BLOCK_SIZE + 1, (row - 1) / BLOCK_SIZE + 1, 1);
  get_diagonal_kernel<<<grid, block>>>(a, row, col, b);
}

void cu_set_diagonal(const double *a, const uint32_t n, double *b) {
  dim3 block(BLOCK_SIZE * BLOCK_SIZE, 1, 1);
  dim3 grid((n - 1) / (BLOCK_SIZE * BLOCK_SIZE) + 1, 1, 1);
  set_diagonal_kernel<<<grid, block>>>(a, n, b);
}

void cu_set_diagonal_c(const cuDoubleComplex *a, const uint32_t n, cuDoubleComplex *b) {
  dim3 block(BLOCK_SIZE * BLOCK_SIZE, 1, 1);
  dim3 grid((n - 1) / (BLOCK_SIZE * BLOCK_SIZE) + 1, 1, 1);
  set_diagonal_kernel_c<<<grid, block>>>(a, n, b);
}

void cu_reciprocal(const cuDoubleComplex *a, const uint32_t row, const uint32_t col, cuDoubleComplex *b) {
  dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
  dim3 grid((col - 1) / BLOCK_SIZE + 1, (row - 1) / BLOCK_SIZE + 1, 1);
  reciprocal_kernel<<<grid, block>>>(a, row, col, b);
}

void cu_hadamard_product(const cuDoubleComplex *a, const cuDoubleComplex *b, const uint32_t row, const uint32_t col, cuDoubleComplex *c) {
  dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
  dim3 grid((col - 1) / BLOCK_SIZE + 1, (row - 1) / BLOCK_SIZE + 1, 1);
  hadamard_product_kernel<<<grid, block>>>(a, b, row, col, c);
}

void cu_abs(const cuDoubleComplex *a, const uint32_t row, const uint32_t col, double *b) {
  dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
  dim3 grid((col - 1) / BLOCK_SIZE + 1, (row - 1) / BLOCK_SIZE + 1, 1);
  abs_kernel<<<grid, block>>>(a, row, col, b);
}

void cu_sqrt(const double *a, const uint32_t row, const uint32_t col, double *b) {
  dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
  dim3 grid((col - 1) / BLOCK_SIZE + 1, (row - 1) / BLOCK_SIZE + 1, 1);
  sqrt_kernel<<<grid, block>>>(a, row, col, b);
}

void cu_make_complex(const double *re, const uint32_t row, const uint32_t col, cuDoubleComplex *dst) {
  dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
  dim3 grid((col - 1) / BLOCK_SIZE + 1, (row - 1) / BLOCK_SIZE + 1, 1);
  make_complex_kernel<<<grid, block>>>(re, row, col, dst);
}

void cu_make_complex2(const double *re, const double *im, const uint32_t row, const uint32_t col, cuDoubleComplex *dst) {
  dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
  dim3 grid((col - 1) / BLOCK_SIZE + 1, (row - 1) / BLOCK_SIZE + 1, 1);
  make_complex2_kernel<<<grid, block>>>(re, im, row, col, dst);
}

void cu_pow(const double *a, const double p, const uint32_t row, const uint32_t col, double *b) {
  dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
  dim3 grid((col - 1) / BLOCK_SIZE + 1, (row - 1) / BLOCK_SIZE + 1, 1);
  pow_kernel<<<grid, block>>>(a, p, row, col, b);
}

void cu_conj(const cuDoubleComplex *a, const uint32_t row, const uint32_t col, cuDoubleComplex *b) {
  dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
  dim3 grid((col - 1) / BLOCK_SIZE + 1, (row - 1) / BLOCK_SIZE + 1, 1);
  conj_kernel<<<grid, block>>>(a, row, col, b);
}

void cu_calc_singular_inv(double *d_s, const uint32_t row, const uint32_t col, const double alpha, cuDoubleComplex *p_singular_inv) {
  dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
  dim3 grid((col - 1) / BLOCK_SIZE + 1, (row - 1) / BLOCK_SIZE + 1, 1);
  calc_singular_inv_kernel<<<grid, block>>>(d_s, row, col, alpha, p_singular_inv);
}

void cu_exp(const cuDoubleComplex *a, const uint32_t row, const uint32_t col, cuDoubleComplex *b) {
  dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
  dim3 grid((col - 1) / BLOCK_SIZE + 1, (row - 1) / BLOCK_SIZE + 1, 1);
  exp_kernel<<<grid, block>>>(a, row, col, b);
}

void cu_real(const cuDoubleComplex *src, const uint32_t row, const uint32_t col, double *dst) {
  dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
  dim3 grid((col - 1) / BLOCK_SIZE + 1, (row - 1) / BLOCK_SIZE + 1, 1);
  real_kernel<<<grid, block>>>(src, row, col, dst);
}

void cu_imag(const cuDoubleComplex *src, const uint32_t row, const uint32_t col, double *dst) {
  dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
  dim3 grid((col - 1) / BLOCK_SIZE + 1, (row - 1) / BLOCK_SIZE + 1, 1);
  imag_kernel<<<grid, block>>>(src, row, col, dst);
}

void cu_reduce_col(const double *mat, const uint32_t m, const uint32_t n, double *result) {
  dim3 block(1, BLOCK_SIZE * BLOCK_SIZE, 1);
  dim3 grid(1, (m - 1) / (BLOCK_SIZE * BLOCK_SIZE) + 1, 1);
  col_sum_kernel<<<grid, block>>>(mat, m, n, result);
}

void cu_generate_propagation_matrix(const double *positions, const double *foci, const double *wavenums, const double *attens, const uint32_t row,
                                    const uint32_t col, cuDoubleComplex *dst) {
  dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
  dim3 grid((col - 1) / BLOCK_SIZE + 1, (row - 1) / BLOCK_SIZE + 1, 1);
  generate_propagation_matrix_kernel<<<grid, block>>>(positions, foci, wavenums, attens, row, col, dst);
}

#ifdef __cplusplus
}
#endif
