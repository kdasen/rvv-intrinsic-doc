#include <riscv_vector.h>   // keep this

// accumulate and reduce
void reduce_golden(double *a, double *b, double *result_sum,
                     int *result_count, int n) {
  int count = 0;
  double s = 0;
  for (int i = 0; i < n; ++i) {
    if (a[i] != 0.0) {
      s += a[i] * b[i];
      count++;
    }
  }

  *result_sum = s;
  *result_count = count;
}

void reduce(double *a, double *b, double *result_sum, int *result_count,
              int n) {
  int count = 0;
  // set vlmax and initialize variables
  size_t vlmax = vsetvlmax_e64m1();
  vfloat64m1_t vec_zero = vfmv_v_f_f64m1(0, vlmax);
  vfloat64m1_t vec_s = vfmv_v_f_f64m1(0, vlmax);
  vfloat64m1_t vec_one = vfmv_v_f_f64m1(1, vlmax);
  for (size_t vl; n > 0; n -= vl, a += vl, b += vl) {
    vl = vsetvl_e64m1(n);

    vfloat64m1_t vec_a = vle64_v_f64m1(a, vl);
    vfloat64m1_t vec_b = vle64_v_f64m1(b, vl);

    vbool64_t mask = vmfne_vv_f64m1_b64(vec_a, vec_zero, vl);

    vec_s = vfmacc_vv_f64m1_m(mask, vec_s, vec_a, vec_b, vl);
    count = count + vpopc_m_b64(mask, vl);
  }
  vfloat64m1_t vec_sum;
  vec_sum = vfredusum_vs_f64m1_f64m1(vec_zero, vec_s, vec_zero, vlmax);
  double sum = vfmv_f_s_f64m1_f64(vec_sum);

  *result_sum = sum;
  *result_count = count;
}

int main() {
  const int N = 31;

  // Pseudorandom vectors:
  // 
  // A: 0    1    1    1    1   | 0    1 ...
  // B: 0.0, 0.1, 0.2, 0.3, 0.4 | 0.0, 0.1, ...
  // The 31st element is masked off
  // count = 6*4   = 24
  // sum   = 6*1.0 = 6
  double A[N], B[N];
  for (int i = 0; i<N; i++) {
    A[i] = (i%5 == 0) ? 0.0: 1.0;
    B[i] = (i%5) * 0.1;
  }


  // compute
  double golden_sum, actual_sum;
  int golden_count, actual_count;
  reduce_golden(A, B, &golden_sum, &golden_count, N);
  reduce(A, B, &actual_sum, &actual_count, N);

  // compare
  if(golden_sum - actual_sum < 1e-6 && golden_count == actual_count) {
    printf("pass");
  }
  else {
    printf("fail");
  }
}