#ifndef _COMMON_H_
#define _COMMON_H_

#include <cuda_runtime.h>

// __device__ __forceinline__ uint32_t ld_gbl_ca(const __restrict__ uint32_t *addr) {
//     uint32_t return_value;
//     asm("ld.global.ca.u32 %0, [%1];" : "=r"(return_value) : "l"(addr));
//     return return_value;
// }

// __device__ __forceinline__ uint32_t ld_gbl_cs(const __restrict__ uint32_t *addr) {
//     uint32_t return_value;
//     asm("ld.global.cs.u32 %0, [%1];" : "=r"(return_value) : "l"(addr));
//     return return_value;
// }

// __device__ __forceinline__ void st_gbl_wt(const __restrict__ uint32_t *addr, const uint32_t value) {
//     asm("st.global.wt.u32 [%0], %1;" :: "l"(addr), "r"(value));
// }

// __device__ __forceinline__ void st_gbl_cs(const __restrict__ uint32_t *addr, const uint32_t value) {
//     asm("st.global.cs.f32 [%0], %1;" :: "l"(addr), "r"(value));
// }


//load global instructions
__device__ __inline__ float ld_gbl_cs(const float *p) {
    float x;
    asm volatile ("ld.global.cs.f32 %0, [%1];" : "=f"(x) : "l"(p));
    return x;
}

__device__ __inline__ float4 ld_gbl_cs(const float4 *p) {
    float4 v;
    asm volatile ("ld.global.cs.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(p));
    return v;
}

__device__ __inline__ float2 ld_gbl_cs(const float2 *p) {
    float2 v;
    asm volatile ("ld.global.cs.v2.f32 {%0,%1}, [%2];" : "=f"(v.x), "=f"(v.y) : "l"(p));
    return v;
}

__device__ __inline__ float ld_gbl_ca(const float *p) {
    float x;
    asm volatile ("ld.global.ca.f32 %0, [%1];" : "=f"(x) : "l"(p));
    return x;
}

__device__ __inline__ float4 ld_gbl_ca(const float4 *p) {
    float4 v;
    asm volatile ("ld.global.ca.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(p));
    return v;
}

__device__ __inline__ float2 ld_gbl_ca(const float2 *p) {
    float2 v;
    asm volatile ("ld.global.ca.v2.f32 {%0,%1}, [%2];" : "=f"(v.x), "=f"(v.y) : "l"(p));
    return v;
}

//store global instructions
__device__ __inline__ void st_gbl_wt(const float *addr, const float value) {
    asm("st.global.wt.f32 [%0], %1;" :: "l"(addr), "f"(value));
}

__device__ __inline__ void st_gbl_wt(const float4 *addr, const float x, const float y, const float z, const float w) {
    asm("st.global.wt.v4.f32 [%0], {%1,%2,%3,%4};" :: "l"(addr), "f"(x), "f"(y), "f"(z), "f"(w));
}

__device__ __inline__ void st_gbl_cs(const float *addr, const float value) {
    asm("st.global.cs.f32 [%0], %1;" :: "l"(addr), "f"(value));
}

__device__ __inline__ void st_gbl_cs(const float4 *addr, const float x, const float y, const float z, const float w) {
    asm("st.global.cs.v4.f32 [%0], {%1,%2,%3,%4};" :: "l"(addr), "f"(x), "f"(y), "f"(z), "f"(w));
}

__device__ __inline__ void st_gbl_cs(const float4 *addr, const float4 v) {
    asm("st.global.cs.v4.f32 [%0], {%1,%2,%3,%4};" :: "l"(addr), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w));
}

__device__ __inline__ void st_gbl_cs(const float2 *addr, const float x, const float y) {
    asm("st.global.cs.v2.f32 [%0], {%1,%2};" :: "l"(addr), "f"(x), "f"(y));
}

//load shared instructions
//__device__ __inline__ float ld_shared(float *p) {
//    float x;
//    asm volatile ("ld.shared.f32 %0, [%1];" : "=f"(x) : "l"(p));
//    return x;
//}

__device__ __inline__ float4 ld_shared(float4 *p) {
    float4 v;
    asm volatile ("ld.shared.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(p));
    return v;
}

//__device__ __inline__ void st_shared(float *addr, const float value) {
//    asm("st.shared.f32 [%0], %1;" :: "l"(addr), "f"(value));
//}

// __device__ inline void st_shared(const __restrict__ float4 *addr, const float x, const float y, const float z, const float w) {
//     asm("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "l"(addr), "f"(x), "f"(y), "f"(z), "f"(w));
// }

__device__ __inline__ void st_shared(float4 *addr, const float value_x, const float value_y, const float value_z, const float value_w) {
    asm("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "l"(addr), "f"(value_x), "f"(value_y), "f"(value_z), "f"(value_w));
}

__device__ __inline__ void st_shared(float4 *addr, const float4 value) {
    asm("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "l"(addr), "f"(value.x), "f"(value.y), "f"(value.z), "f"(value.w));
}

#endif 
