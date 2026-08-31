// SPDX-License-Identifier: BSD-3-Clause
//
// The two UMMA descriptors needed to drive SM100 `tcgen05.mma.kind::f8f6f4`
// from inline PTX, vendored from CUTLASS `cute/arch/mma_sm100_desc.hpp`
// (<https://github.com/NVIDIA/cutlass>, BSD-3-Clause) with the bitfields
// verbatim.
//
// Vendored rather than included so the solution stays self-contained per the
// contest rules and independent of the CUTLASS include path in the evaluation
// container. Kept out of the `cute`/`cutlass` namespaces to avoid colliding
// with the real headers, and depends on nothing but <cstdint>.

#pragma once

#include <cstdint>

namespace dsa_umma {

enum class Major : uint8_t {
    K  = 0,
    MN = 1,
};

enum class ScaleIn : uint8_t {
    One = 0,
    Neg = 1,
};

enum class Saturate : uint8_t {
    False = 0,
    True  = 1,
};

// Swizzle encoding for SmemDescriptor.layout_type_, per the SM100 UMMA ISA.
enum class LayoutType : uint8_t {
    SWIZZLE_NONE           = 0,
    SWIZZLE_128B_BASE32B   = 1,
    SWIZZLE_128B           = 2,
    SWIZZLE_64B            = 4,
    SWIZZLE_32B            = 6,
};

// Operand data type for InstrDescriptor.a_format_ / b_format_.
enum class F8F6F4Format : uint8_t {
    E4M3 = 0,
    E5M2 = 1,
    E2M3 = 3,
    E3M2 = 4,
    E2M1 = 5,
};

// Accumulator format for InstrDescriptor.c_format_.
enum class CFormat : uint8_t {
    F16 = 0,
    F32 = 1,
    S32 = 2,
};

// SmemDescriptor, 64-bit. Ranges below are [from, to) with LSB == bit 0:
//   [ 0,14)  start_address       (14 bits, SMEM byte addr >> 4)
//   [14,16)  unused
//   [16,30)  leading_byte_offset (14 bits, byte stride >> 4)
//   [30,32)  unused
//   [32,46)  stride_byte_offset  (14 bits, byte stride >> 4)
//   [46,48)  version             (2 bits; SM100 UMMA uses value 1)
//   [48,49)  unused
//   [49,52)  base_offset         (3 bits)
//   [52,53)  lbo_mode            (1 bit; SM100 UMMA "legacy mode" uses 0)
//   [53,61)  unused
//   [61,64)  layout_type         (3 bits, LayoutType enum)

union SmemDescriptor {
    uint64_t desc_;

    // Uniform uint64_t bitfield storage, as CUTLASS does: mixing storage types
    // lets the compiler insert padding, which silently shifts later fields and
    // makes the MMA read the wrong SMEM rows.
    struct {
        uint64_t start_address_       : 14;
        uint64_t /* pad */            : 2;
        uint64_t leading_byte_offset_ : 14;
        uint64_t /* pad */            : 2;
        uint64_t stride_byte_offset_  : 14;
        uint64_t version_             : 2;
        uint64_t /* pad */            : 1;
        uint64_t base_offset_         : 3;
        uint64_t lbo_mode_            : 1;
        uint64_t /* pad */            : 8;
        uint64_t layout_type_         : 3;
    };

    __host__ __device__ constexpr
    operator uint64_t() const noexcept { return desc_; }
};

static_assert(sizeof(SmemDescriptor) == 8, "SmemDescriptor must be 64-bit");

// InstrDescriptor, 32-bit, for dense F8F6F4 UMMA:
//   [ 0, 2) sparse_id2    (0 for dense)
//   [ 2, 3) sparse_flag   (0 = dense)
//   [ 3, 4) saturate
//   [ 4, 6) c_format      (CFormat enum)
//   [ 6, 7) unused
//   [ 7,10) a_format      (F8F6F4Format enum)
//   [10,13) b_format      (F8F6F4Format enum)
//   [13,14) a_negate
//   [14,15) b_negate
//   [15,16) a_major       (Major enum; 0 = K-major)
//   [16,17) b_major       (Major enum; 0 = K-major)
//   [17,23) n_dim         (N >> 3; range 1..32 -> N=8..256)
//   [23,24) unused
//   [24,29) m_dim         (M >> 4; 4 = M64, 8 = M128, 16 = M256)
//   [29,30) unused
//   [30,32) max_shift     (0 = no shift)

union InstrDescriptor {
    uint32_t desc_;

    // Uniform storage, for the reason given on SmemDescriptor.
    struct {
        uint32_t sparse_id2_  : 2;
        uint32_t sparse_flag_ : 1;
        uint32_t saturate_    : 1;
        uint32_t c_format_    : 2;
        uint32_t /* pad */    : 1;
        uint32_t a_format_    : 3;
        uint32_t b_format_    : 3;
        uint32_t a_negate_    : 1;
        uint32_t b_negate_    : 1;
        uint32_t a_major_     : 1;
        uint32_t b_major_     : 1;
        uint32_t n_dim_       : 6;
        uint32_t /* pad */    : 1;
        uint32_t m_dim_       : 5;
        uint32_t /* pad */    : 1;
        uint32_t max_shift_   : 2;
    };

    __host__ __device__ constexpr explicit
    operator uint32_t() const noexcept { return desc_; }
};

static_assert(sizeof(InstrDescriptor) == 4, "InstrDescriptor must be 32-bit");

// `m` must be in {64, 128, 256}, `n` a multiple of 8 in [8, 256]. Not constexpr
// because C++17 forbids switching the active union member in a constant
// expression; nvcc inlines it regardless.
__host__ __device__ inline InstrDescriptor
make_instr_desc_f8f6f4(F8F6F4Format a_fmt,
                       F8F6F4Format b_fmt,
                       CFormat      c_fmt,
                       Major        a_major,
                       Major        b_major,
                       uint32_t     m,
                       uint32_t     n,
                       ScaleIn      a_neg = ScaleIn::One,
                       ScaleIn      b_neg = ScaleIn::One,
                       Saturate     sat   = Saturate::False) {
    InstrDescriptor d{};
    d.desc_         = 0;
    d.sparse_id2_   = 0;
    d.sparse_flag_  = 0;
    d.saturate_     = static_cast<uint8_t>(sat);
    d.c_format_     = static_cast<uint8_t>(c_fmt);
    d.a_format_     = static_cast<uint8_t>(a_fmt);
    d.b_format_     = static_cast<uint8_t>(b_fmt);
    d.a_negate_     = static_cast<uint8_t>(a_neg);
    d.b_negate_     = static_cast<uint8_t>(b_neg);
    d.a_major_      = static_cast<uint8_t>(a_major);
    d.b_major_      = static_cast<uint8_t>(b_major);
    d.n_dim_        = (n >> 3);
    d.m_dim_        = (m >> 4);
    d.max_shift_    = 0;
    return d;
}

} // namespace dsa_umma
