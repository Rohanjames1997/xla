
#include <arm_sve.h>
#include <stdint.h>

#include "absl/base/attributes.h"
#include "absl/base/casts.h"
#include "absl/base/dynamic_annotations.h"

#include "xla/service/cpu/runtime_tanh.h"

#pragma GCC target("+sve")
#pragma clang attribute push(__attribute__((target("sve"))), apply_to = function)
void tanh_array_sve_x4(float *input, float *output, int size)
{
    // This is a manually scheduled implemtation of the x4 unrolled SVE version.
    // The statements in groups separated by line breaks should be able to go
    // through the execution pipelines in parallel.

    // Prepare the constants for calculations.
    const svfloat32_t plus_9 = svdup_f32(9.f);
    const svfloat32_t minus_9 = svdup_f32(-9.f);

    // The monomial coefficients of the numerator polynomial (odd).
    const svfloat32_t alpha_1 = svdup_f32(4.89352455891786e-03f);
    const svfloat32_t alpha_3 = svdup_f32(6.37261928875436e-04f);
    const svfloat32_t alpha_5 = svdup_f32(1.48572235717979e-05f);
    const svfloat32_t alpha_7 = svdup_f32(5.12229709037114e-08f);
    const svfloat32_t alpha_9 = svdup_f32(-8.60467152213735e-11f);
    const svfloat32_t alpha_11 = svdup_f32(2.00018790482477e-13f);
    const svfloat32_t alpha_13 = svdup_f32(-2.76076847742355e-16f);

    // The monomial coefficients of the denominator polynomial (even).
    const svfloat32_t beta_0 = svdup_f32(4.89352518554385e-03f);
    const svfloat32_t beta_2 = svdup_f32(2.26843463243900e-03f);
    const svfloat32_t beta_4 = svdup_f32(1.18534705686654e-04f);
    const svfloat32_t beta_6 = svdup_f32(1.19825839466702e-06f);

    int i = 0;
    int sve_len = svcntw();
    svbool_t predicate_0 = svwhilelt_b32_s32(i+sve_len*0, size);
    svbool_t predicate_1 = svwhilelt_b32_s32(i+sve_len*1, size);
    svbool_t predicate_2 = svwhilelt_b32_s32(i+sve_len*2, size);
    svbool_t predicate_3 = svwhilelt_b32_s32(i+sve_len*3, size);
    do {
        svfloat32_t x_0, x2_0, p_0, q_0, result_0;
        svfloat32_t x_1, x2_1, p_1, q_1, result_1;
        svfloat32_t x_2, x2_2, p_2, q_2, result_2;
        svfloat32_t x_3, x2_3, p_3, q_3, result_3;

        // load input lanes
        x_0 = svld1_f32(predicate_0, &input[i+sve_len*0]);
        x_1 = svld1_f32(predicate_1, &input[i+sve_len*1]);
        x_2 = svld1_f32(predicate_2, &input[i+sve_len*2]);
        x_3 = svld1_f32(predicate_3, &input[i+sve_len*3]);

        // Clamp the inputs to the range [-9, 9] since anything outside
        // this range is +/-1.0f in single-precision.
        x_0 = svmax_x(predicate_0, svmin_x(predicate_0, x_0, plus_9), minus_9);
        x_1 = svmax_x(predicate_1, svmin_x(predicate_1, x_1, plus_9), minus_9);
        x_2 = svmax_x(predicate_2, svmin_x(predicate_2, x_2, plus_9), minus_9);
        x_3 = svmax_x(predicate_3, svmin_x(predicate_3, x_3, plus_9), minus_9);

        // Since the polynomials are odd/even, we need x^2.
        x2_0 = svmul_x(predicate_0, x_0, x_0);
        x2_1 = svmul_x(predicate_1, x_1, x_1);
        x2_2 = svmul_x(predicate_1, x_2, x_2);
        x2_3 = svmul_x(predicate_1, x_3, x_3);

        // Evaluate the numerator polynomial p.
        // Evaluate the denominator polynomial q.
        p_0 = x2_0;
        p_1 = x2_1;
        q_0 = x2_0;
        q_1 = x2_1;
        p_2 = x2_2;
        p_3 = x2_3;
        q_2 = x2_2;
        q_3 = x2_3;

        p_0 = svmad_x(predicate_0, p_0,   alpha_13, alpha_11);
        p_1 = svmad_x(predicate_1, p_1,   alpha_13, alpha_11);
        q_0 = svmad_x(predicate_0, q_0,     beta_6, beta_4);
        q_1 = svmad_x(predicate_1, q_1,     beta_6, beta_4);

        p_2 = svmad_x(predicate_2, p_2,   alpha_13, alpha_11);
        p_3 = svmad_x(predicate_3, p_3,   alpha_13, alpha_11);
        q_2 = svmad_x(predicate_2, q_2,     beta_6, beta_4);
        q_3 = svmad_x(predicate_3, q_3,     beta_6, beta_4);

        p_0 = svmad_x(predicate_0, p_0,       x2_0, alpha_9);
        p_1 = svmad_x(predicate_1, p_1,       x2_1, alpha_9);
        q_0 = svmad_x(predicate_0, q_0,       x2_0, beta_2);
        q_1 = svmad_x(predicate_1, q_1,       x2_1, beta_2);

        p_2 = svmad_x(predicate_2, p_2,       x2_2, alpha_9);
        p_3 = svmad_x(predicate_3, p_3,       x2_3, alpha_9);
        q_2 = svmad_x(predicate_2, q_2,       x2_2, beta_2);
        q_3 = svmad_x(predicate_3, q_3,       x2_3, beta_2);

        p_0 = svmad_x(predicate_0, p_0,       x2_0, alpha_7);
        p_1 = svmad_x(predicate_1, p_1,       x2_1, alpha_7);
        q_0 = svmad_x(predicate_0, q_0,       x2_0, beta_0);
        q_1 = svmad_x(predicate_1, q_1,       x2_1, beta_0);

        p_2 = svmad_x(predicate_2, p_2,       x2_2, alpha_7);
        p_3 = svmad_x(predicate_3, p_3,       x2_3, alpha_7);
        q_2 = svmad_x(predicate_2, q_2,       x2_2, beta_0);
        q_3 = svmad_x(predicate_3, q_3,       x2_3, beta_0);

        p_0 = svmad_x(predicate_0, p_0,       x2_0, alpha_5);
        p_1 = svmad_x(predicate_1, p_1,       x2_1, alpha_5);
        p_2 = svmad_x(predicate_2, p_2,       x2_2, alpha_5);
        p_3 = svmad_x(predicate_3, p_3,       x2_3, alpha_5);

        p_0 = svmad_x(predicate_0, p_0,       x2_0, alpha_3);
        p_1 = svmad_x(predicate_1, p_1,       x2_1, alpha_3);
        p_2 = svmad_x(predicate_2, p_2,       x2_2, alpha_3);
        p_3 = svmad_x(predicate_3, p_3,       x2_3, alpha_3);

        p_0 = svmad_x(predicate_0, p_0,       x2_0, alpha_1);
        p_1 = svmad_x(predicate_1, p_1,       x2_1, alpha_1);
        p_2 = svmad_x(predicate_2, p_2,       x2_2, alpha_1);
        p_3 = svmad_x(predicate_3, p_3,       x2_3, alpha_1);

        p_0 = svmul_x(predicate_0, p_0, x_0);
        p_1 = svmul_x(predicate_1, p_1, x_1);
        p_2 = svmul_x(predicate_2, p_2, x_2);
        p_3 = svmul_x(predicate_3, p_3, x_3);

        // Divide the numerator by the denominator.
        result_0 = svdiv_x(predicate_0, p_0, q_0);
        result_1 = svdiv_x(predicate_1, p_1, q_1);
        result_2 = svdiv_x(predicate_2, p_2, q_2);
        result_3 = svdiv_x(predicate_3, p_3, q_3);

        // write results back to memory
        svst1_f32(predicate_0, &output[i+sve_len*0], result_0);
        svst1_f32(predicate_1, &output[i+sve_len*1], result_1);
        svst1_f32(predicate_2, &output[i+sve_len*2], result_2);
        svst1_f32(predicate_3, &output[i+sve_len*3], result_3);

        // loop accounting
        i += sve_len*4;
        predicate_0 = svwhilelt_b32_s32(i+sve_len*0, size);
        predicate_1 = svwhilelt_b32_s32(i+sve_len*1, size);
        predicate_2 = svwhilelt_b32_s32(i+sve_len*2, size);
        predicate_3 = svwhilelt_b32_s32(i+sve_len*3, size);
    } while (svptest_any(svptrue_b8(), predicate_0));
}
#pragma clang attribute pop


ABSL_ATTRIBUTE_NO_SANITIZE_MEMORY float __xla_cpu_runtime_Aarch64SveHyperbolicTangent(float input)
{
    float output;
    tanh_array_sve_x4(&input, &output, 1);
    return output;
}
