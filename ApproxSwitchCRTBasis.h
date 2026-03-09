
//==================================================================================
// BSD 2-Clause License
//
// Copyright (c) 2014-2024, NJIT, Duality Technologies Inc. and other contributors
//
// All rights reserved.
//
// Author TPOC: contact@openfhe.org
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//==================================================================================


// ApproxSwitchCRTBasis function adapted for the APU
template <typename VecType>
DCRTPolyImpl<VecType> DCRTPolyImpl<VecType>::ApproxSwitchCRTBasis(
    const std::shared_ptr<Params>& paramsQ, const std::shared_ptr<Params>& paramsP,
    const std::vector<NativeInteger>& QHatInvModq, const std::vector<NativeInteger>& QHatInvModqPrecon,
    const std::vector<std::vector<NativeInteger>>& QHatModp, const std::vector<DoubleNativeInt>& modpBarrettMu) const {

DCRTPolyImpl<VecType> ans(paramsP, m_format, true);

uint32_t ringDim = m_params->GetRingDimension();
uint32_t sizeQ = (m_vectors.size() > paramsQ->GetParams().size()) ? paramsQ->GetParams().size() : m_vectors.size();
uint32_t sizeP = ans.m_vectors.size();

#if defined(HAVE_INT128) && NATIVEINT == 64
m_vectors_struct* flat_m_vectors  = (m_vectors_struct*) malloc(sizeQ * sizeof(m_vectors_struct));
for(uint32_t i=0; i<sizeQ; i++) {
        flat_m_vectors[i].data = (uint64_t*) malloc(ringDim * sizeof(uint64_t));
}

uint64_t* flat_QHatInvModq              =       (uint64_t*) malloc(sizeQ * sizeof(uint64_t));
uint64_t* flat_QHatInvModqPrecon        =       (uint64_t*) malloc(sizeQ * sizeof(uint64_t));
uint64_t* flat_QHatModp                 =       (uint64_t*) malloc((sizeQ * sizeP + sizeP) * sizeof(uint64_t));
uint64_t* flat2_modpBarrettMu           =       (uint64_t*) malloc(2*sizeP * sizeof(uint64_t));
m_vectors_struct* flat_ans_m_vectors =  (m_vectors_struct*) malloc(sizeP * sizeof(m_vectors_struct));

uint64_t s_m_vectors_data2[8*32768];
uint64_t modulus0[8];

for (uint32_t i=0; i<sizeP; i++) {
        flat_ans_m_vectors[i].data = (uint64_t*) malloc(ringDim * sizeof(uint64_t));
}

// prep data
auto start = high_resolution_clock::now();
for (uint32_t i = 0; i < sizeQ; i++) {
        flat_m_vectors[i].modulus       = m_vectors[i].GetModulus().ConvertToInt();
        flat_QHatInvModq[i]             = QHatInvModq[i].ConvertToInt();
        flat_QHatInvModqPrecon[i]       = QHatInvModqPrecon[i].ConvertToInt<uint64_t>();
        for (uint32_t k = 0; k < sizeP; k++) {
          flat_QHatModp[i * sizeP + k] = QHatModp[i][k].ConvertToInt<uint64_t>();
        }
}

for (uint32_t ri = 0; ri < ringDim; ri++) {
        for (uint32_t i = 0; i < sizeQ; i++) {
                 s_m_vectors_data2[(ri*8)+i] = m_vectors[i][ri].template ConvertToInt<uint64_t>();
        }
}

DoubleNativeInt val=0;
for (uint32_t i = 0; i < sizeP; i++) {
        val =modpBarrettMu[i];
        modulus0[i] = ans.m_vectors[i].GetModulus().ConvertToInt();

        flat2_modpBarrettMu[2*i]   = static_cast<uint64_t>(val);
        flat2_modpBarrettMu[2*i+1] = static_cast<uint64_t>(val >> 64);
}

uint64_t flat_join[3*8];
for (int i=0; i<24; i+=3) {
        flat_join[i]   = flat_QHatInvModq[i/3];
        flat_join[i+1] = flat_m_vectors[i/3].modulus;
        flat_join[i+2] = flat_QHatInvModqPrecon[i/3];
}

auto stop = high_resolution_clock::now();
auto duration = duration_cast<microseconds>( stop - start );
std::cout << "\n\t Data Preparation - ApproxSwitchCRTBasis: " << duration.count() << " [us]" << std::endl;

uint64_t xQ;
uint64_t data0[262144];

double start1 = omp_get_wtime();

#pragma omp target teams distribute parallel for private(xQ) thread_limit(32)
for (int ri = 0; ri < ringDim; ri++) {
        uint64_t sum0[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

        #pragma unroll
        for (int k = 0; k < 8; k++) { // sizeQ
            xQ = ModMulFastConst_64(s_m_vectors_data2[ri*8+k], flat_join[k*3], flat_join[k*3+1], flat_join[k*3+2]);

            add128_flat_inplace3(xQ, flat_QHatModp[k*8    ], &sum0[0] );
            add128_flat_inplace3(xQ, flat_QHatModp[k*8 + 1], &sum0[2] );
            add128_flat_inplace3(xQ, flat_QHatModp[k*8 + 2], &sum0[4] );
            add128_flat_inplace3(xQ, flat_QHatModp[k*8 + 3], &sum0[6] );
            add128_flat_inplace3(xQ, flat_QHatModp[k*8 + 4], &sum0[8] );
            add128_flat_inplace3(xQ, flat_QHatModp[k*8 + 5], &sum0[10]);
            add128_flat_inplace3(xQ, flat_QHatModp[k*8 + 6], &sum0[12]);
            add128_flat_inplace3(xQ, flat_QHatModp[k*8 + 7], &sum0[14]);
        }

        #pragma omp simd
        for (int n = 0; n<8; n++) // sizeP
                BarrettUint128ModUint64_apu_flat2(&sum0[n*2],  modulus0[n], &flat2_modpBarrettMu[n*2],  data0[ri*8+n]);
        }

double stop1 = omp_get_wtime();
printf("ApproxSwitchCRTBasis - Pragma Time: %.3f [us]\n", (stop1-start1)*1e6);


for (uint32_t ri = 0; ri < ringDim; ri++) {
        ans.m_vectors[0][ri] = data0[ri*8];
        ans.m_vectors[1][ri] = data0[ri*8+1];
        ans.m_vectors[2][ri] = data0[ri*8+2];
        ans.m_vectors[3][ri] = data0[ri*8+3];
        ans.m_vectors[4][ri] = data0[ri*8+4];
        ans.m_vectors[5][ri] = data0[ri*8+5];
        ans.m_vectors[6][ri] = data0[ri*8+6];
        ans.m_vectors[7][ri] = data0[ri*8+7];
}

#endif
return ans;
}

