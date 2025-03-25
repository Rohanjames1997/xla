

#ifndef XLA_SERVICE_CPU_RUNTIME_TANH_H_
#define XLA_SERVICE_CPU_RUNTIME_TANH_H_

#include <stdint.h>

extern "C" {

extern void __xla_cpu_runtime_Aarch64SveHyperbolicTangent(float *input, float *output, int size);
}

#endif  // XLA_SERVICE_CPU_RUNTIME_TOPK_H_
