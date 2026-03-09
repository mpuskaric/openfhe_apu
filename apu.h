// Helper functions for porting the ApproxSwitchCRTBasis function to the APU
// Note: AI-assisted code
// Reviewed and adapted by https://github.com/mpuskaric  

#include "config_core.h"

#include "lattice/hal/default/poly-impl.h"
#include "lattice/hal/default/dcrtpoly.h"

#include "utils/exception.h"
#include "utils/inttypes.h"
#include "utils/parallel.h"
#include "utils/utilities.h"
#include "utils/utilities-int.h"

#include <algorithm>
#include <ostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <chrono>

#include <typeinfo>

namespace lbcrypto {

#pragma omp declare target
// Data type for m_vectors object 
struct m_vectors_struct {
    uint64_t* data;
    uint64_t modulus;
};


//mul64hi implementation to use on APU

inline uint64_t mul64_hi(uint64_t a, uint64_t b) {
    __uint128_t prod = static_cast<__uint128_t>(a) * b;
    return static_cast<uint64_t>(prod >> 64);
}


// Modular multiplication using a precomputation for the multiplicand.

inline __attribute__((always_inline))
uint64_t ModMulFastConst_64(uint64_t a, uint64_t b, uint64_t modulus, uint64_t bInv) {
	uint64_t q = mul64_hi(a, bInv);
	uint64_t y = a * b - q * modulus;
	uint64_t mask = -(y >= modulus);
	return y - (modulus & mask);
}


// a + modulus might go out of uint_64 limit 
inline __attribute__((always_inline))
uint64_t ModSubFast(uint64_t a, uint64_t b, uint64_t modulus) {
	//uint64_t max = 18446744073709551615ULL;
	if (a < b) {
		//if (a > (max - modulus)) { std::cout << "Overflow detected " << std::endl; }
		//return a + modulus - b;
		return modulus - (b - a);
	}
	else {
		return a - b;
	}
}


inline __attribute__((always_inline))
void ModMulFastConst_64_flat(uint64_t a, uint64_t b, uint64_t modulus, uint64_t bInv, uint64_t &res) {
	__uint128_t prod = static_cast<__uint128_t>(a) * bInv;
    	uint64_t q =  static_cast<uint64_t>(prod >> 64);
        uint64_t y = a * b - q * modulus;
        uint64_t mask = -(y >= modulus);
        res = y - (modulus & mask);
}

// mul64 FLAT
inline __attribute__((always_inline))
std::array<uint64_t,2> mul64_flat(uint64_t a, uint64_t b) {
        std::array<uint64_t,2> result;
        __uint128_t prod = (__uint128_t)a * b;
        result[0] = (uint64_t)prod;        // lo
        result[1] = (uint64_t)(prod >> 64); // hi
        return result;
}

inline __attribute__((always_inline))
void mul64_flat_into(uint64_t a, uint64_t b, uint64_t out[2]) {
    unsigned __int128 prod = (unsigned __int128)a * (unsigned __int128)b;
    out[0] = (uint64_t)prod;           // lo
    out[1] = (uint64_t)(prod >> 64);   // hi
}
inline __attribute__((always_inline))
void mul64_flat_into2(uint64_t a, uint64_t b, uint64_t out[2]) {
    uint64_t a_lo = (uint32_t)a;
    uint64_t a_hi = a >> 32;
    uint64_t b_lo = (uint32_t)b;
    uint64_t b_hi = b >> 32;

    uint64_t p0 = a_lo * b_lo;  // 32×32 → 64
    uint64_t p1 = a_lo * b_hi;  // 32×32 → 64
    uint64_t p2 = a_hi * b_lo;  // 32×32 → 64
    uint64_t p3 = a_hi * b_hi;  // 32×32 → 64

    // assemble low part
    uint64_t middle = (p0 >> 32) + (uint32_t)p1 + (uint32_t)p2;
    out[0] = (middle << 32) | (uint32_t)p0;

    // assemble high part
    out[1] = p3 + (p1 >> 32) + (p2 >> 32) + (middle >> 32);
}

//Subtract FLAT
inline __attribute__((always_inline))
std::array<uint64_t,2> sub128_flat(const uint64_t a[2], const uint64_t b[2]) {
        std::array<uint64_t,2> result;
        uint64_t borrow = (a[0] < b[0]);
        result[0] = a[0] - b[0];
        result[1] = a[1] - b[1] - borrow;
        return result;
}

//Add FLAT
inline __attribute__((always_inline))
std::array<uint64_t,2> add128_flat(const uint64_t a[2], const uint64_t b[2]) {
        std::array<uint64_t,2> result;
        result[0] = a[0] + b[0];
        uint64_t carry = (result[0] < a[0]);
        result[1] = a[1] + b[1] + carry;
        return result;
}

//Add in place FLAT
inline __attribute__((always_inline))
void add128_flat_inplace(uint64_t a[2], const uint64_t b[2]) {
    auto tmp = add128_flat(a, b);
    a[0] = tmp[0];
    a[1] = tmp[1];
}
inline __attribute__((always_inline))
void add128_flat_inplace2(uint64_t a, const uint64_t b, uint64_t c[2]) {
	unsigned __int128 prod = (unsigned __int128)a * (unsigned __int128)b;
	uint64_t out[2];
	out[0] = (uint64_t)prod;
	out[1] = (uint64_t)(prod >> 64);
	auto tmp = add128_flat(c, out);
	c[0] = tmp[0];
	c[1] = tmp[1];
}
inline __attribute__((always_inline))
void add128_flat_inplace3(uint64_t a, const uint64_t b, uint64_t c[2]) {
	unsigned __int128 prod = (unsigned __int128)a * (unsigned __int128)b;
	uint64_t lo = (uint64_t)prod;
	uint64_t hi = (uint64_t)(prod >> 64);

	uint64_t tmp = c[0] + lo;
	uint64_t carry = (tmp < lo);
	c[0] = tmp;
	c[1] = c[1] + hi + carry;
}



//mul128_hi FLAT
inline __attribute__((always_inline))
uint64_t mul128_hi_flat(const uint64_t a[2], const uint64_t b[2]) {
        __uint128_t lo_lo = (__uint128_t)a[0] * b[0]; // a.lo * b.lo
        __uint128_t lo_hi = (__uint128_t)a[0] * b[1]; // a.lo * b.hi
        __uint128_t hi_lo = (__uint128_t)a[1] * b[0]; // a.hi * b.lo
        __uint128_t hi_hi = (__uint128_t)a[1] * b[1]; // a.hi * b.hi

        __uint128_t cross = (lo_hi + hi_lo) + (lo_lo >> 64);
        __uint128_t high128 = (hi_hi << 64) + cross;

        return (uint64_t)(high128 >> 64);
}

// Barrett Reduction FLAT
inline __attribute__((always_inline))
uint64_t BarrettUint128ModUint64_apu_flat(
        const uint64_t a[2],
        uint64_t modulus,
        const uint64_t mu[2]) {

        uint64_t q = mul128_hi_flat(a, mu);
        std::array<uint64_t,2> qmul = mul64_flat(q, modulus);
        std::array<uint64_t,2> r = sub128_flat(a, qmul.data());

        r[0] -= modulus & -(r[0] >= modulus);

        return r[0];
}
inline __attribute__((always_inline))
void BarrettUint128ModUint64_apu_flat2(
        const uint64_t a[2],
        uint64_t modulus,
        const uint64_t mu[2],
        uint64_t &res) {

        uint64_t q = mul128_hi_flat(a, mu); 
	    
}


//Comparison FLAT
inline __attribute__((always_inline))
bool lessThan128_flat(const uint64_t a[2], const uint64_t b[2]) {
    return (a[1] < b[1]) || (a[1] == b[1] && a[0] < b[0]);
}

// Modular subtraction FLAT
inline __attribute__((always_inline))
std::array<uint64_t,2> ModSubFast128_flat(const uint64_t a[2], const uint64_t b[2], const uint64_t modulus2[2]) {
        if (lessThan128_flat(a, b)) {
                return add128_flat(sub128_flat(modulus2,b).data(),a);
        }
        else {
                return sub128_flat(a,b);
        }
}


#pragma omp end declare target
}

