/*
	Optimized keccak256_gpu_hash_ZP CUDA implementation for CCMINER written by (c) sp in january 2021
	For the ZenProtocol coin
*/

#include "miner.h"
#ifdef __cplusplus
#include <cstdint>
#else
#include <stdint.h>
#endif
#include <memory.h>
#include "cuda_helper.h"


#define UINT2(x,y) make_uint2(x,y)

static uint32_t *d_KNonce[MAX_GPUS];

__constant__ uint32_t pTarget[8];
__constant__ uint64_t keccak_round_constants[24] = {
	0x0000000000000001ull, 0x0000000000008082ull,
	0x800000000000808aull, 0x8000000080008000ull,
	0x000000000000808bull, 0x0000000080000001ull,
	0x8000000080008081ull, 0x8000000000008009ull,
	0x000000000000008aull, 0x0000000000000088ull,
	0x0000000080008009ull, 0x000000008000000aull,
	0x000000008000808bull, 0x800000000000008bull,
	0x8000000000008089ull, 0x8000000000008003ull,
	0x8000000000008002ull, 0x8000000000000080ull,
	0x000000000000800aull, 0x800000008000000aull,
	0x8000000080008081ull, 0x8000000000008080ull,
	0x0000000080000001ull, 0x8000000080008008ull
};

__constant__ uint2 keccak_round_constants35[24] = {
		{ 0x00000001ul, 0x00000000 }, { 0x00008082ul, 0x00000000 },
		{ 0x0000808aul, 0x80000000 }, { 0x80008000ul, 0x80000000 },
		{ 0x0000808bul, 0x00000000 }, { 0x80000001ul, 0x00000000 },
		{ 0x80008081ul, 0x80000000 }, { 0x00008009ul, 0x80000000 },
		{ 0x0000008aul, 0x00000000 }, { 0x00000088ul, 0x00000000 },
		{ 0x80008009ul, 0x00000000 }, { 0x8000000aul, 0x00000000 },
		{ 0x8000808bul, 0x00000000 }, { 0x0000008bul, 0x80000000 },
		{ 0x00008089ul, 0x80000000 }, { 0x00008003ul, 0x80000000 },
		{ 0x00008002ul, 0x80000000 }, { 0x00000080ul, 0x80000000 },
		{ 0x0000800aul, 0x00000000 }, { 0x8000000aul, 0x80000000 },
		{ 0x80008081ul, 0x80000000 }, { 0x00008080ul, 0x80000000 },
		{ 0x80000001ul, 0x00000000 }, { 0x80008008ul, 0x80000000 }
};


__constant__ uint2 __align__(16) c_PaddedMessageZP_PRE[23];
__constant__ uint2 __align__(16) c_PaddedMessage80[10]; // padded message (80 bytes + padding?)
#define bitselect(a, b, c) ((a) ^ ((c) & ((b) ^ (a))))

static void __forceinline__ __device__ keccak_block(uint2 *s)
{
	uint2 bc[5], tmpxor[5], tmp1, tmp2;
//	uint2 s[25];

#pragma unroll 1
	for (int i= 0; i < 24; i++) 
	{
#pragma unroll
		for (uint32_t x = 0; x < 5; x++)
			tmpxor[x] = s[x] ^ s[x + 5] ^ s[x + 10] ^ s[x + 15] ^ s[x + 20];

		bc[0] = tmpxor[0] ^ ROL2(tmpxor[2], 1);
		bc[1] = tmpxor[1] ^ ROL2(tmpxor[3], 1);
		bc[2] = tmpxor[2] ^ ROL2(tmpxor[4], 1);
		bc[3] = tmpxor[3] ^ ROL2(tmpxor[0], 1);
		bc[4] = tmpxor[4] ^ ROL2(tmpxor[1], 1);

		tmp1 = s[1] ^ bc[0];

		s[0] ^= bc[4];
		s[1] = ROL2(s[6] ^ bc[0], 44);
		s[6] = ROL2(s[9] ^ bc[3], 20);
		s[9] = ROL2(s[22] ^ bc[1], 61);
		s[22] = ROL2(s[14] ^ bc[3], 39);
		s[14] = ROL2(s[20] ^ bc[4], 18);
		s[20] = ROL2(s[2] ^ bc[1], 62);
		s[2] = ROL2(s[12] ^ bc[1], 43);
		s[12] = ROL2(s[13] ^ bc[2], 25);
		s[13] = ROL8(s[19] ^ bc[3]);
		s[19] = ROR8(s[23] ^ bc[2]);
		s[23] = ROL2(s[15] ^ bc[4], 41);
		s[15] = ROL2(s[4] ^ bc[3], 27);
		s[4] = ROL2(s[24] ^ bc[3], 14);
		s[24] = ROL2(s[21] ^ bc[0], 2);
		s[21] = ROL2(s[8] ^ bc[2], 55);
		s[8] = ROL2(s[16] ^ bc[0], 45);
		s[16] = ROL2(s[5] ^ bc[4], 36);
		s[5] = ROL2(s[3] ^ bc[2], 28);
		s[3] = ROL2(s[18] ^ bc[2], 21);
		s[18] = ROL2(s[17] ^ bc[1], 15);
		s[17] = ROL2(s[11] ^ bc[0], 10);
		s[11] = ROL2(s[7] ^ bc[1], 6);
		s[7] = ROL2(s[10] ^ bc[4], 3);
		s[10] = ROL2(tmp1, 1);

		tmp1 = s[0]; tmp2 = s[1]; s[0] = bitselect(s[0] ^ s[2], s[0], s[1]); s[1] = bitselect(s[1] ^ s[3], s[1], s[2]); s[2] = bitselect(s[2] ^ s[4], s[2], s[3]); s[3] = bitselect(s[3] ^ tmp1, s[3], s[4]); s[4] = bitselect(s[4] ^ tmp2, s[4], tmp1);
		tmp1 = s[5]; tmp2 = s[6]; s[5] = bitselect(s[5] ^ s[7], s[5], s[6]); s[6] = bitselect(s[6] ^ s[8], s[6], s[7]); s[7] = bitselect(s[7] ^ s[9], s[7], s[8]); s[8] = bitselect(s[8] ^ tmp1, s[8], s[9]); s[9] = bitselect(s[9] ^ tmp2, s[9], tmp1);
		tmp1 = s[10]; tmp2 = s[11]; s[10] = bitselect(s[10] ^ s[12], s[10], s[11]); s[11] = bitselect(s[11] ^ s[13], s[11], s[12]); s[12] = bitselect(s[12] ^ s[14], s[12], s[13]); s[13] = bitselect(s[13] ^ tmp1, s[13], s[14]); s[14] = bitselect(s[14] ^ tmp2, s[14], tmp1);
		tmp1 = s[15]; tmp2 = s[16]; s[15] = bitselect(s[15] ^ s[17], s[15], s[16]); s[16] = bitselect(s[16] ^ s[18], s[16], s[17]); s[17] = bitselect(s[17] ^ s[19], s[17], s[18]); s[18] = bitselect(s[18] ^ tmp1, s[18], s[19]); s[19] = bitselect(s[19] ^ tmp2, s[19], tmp1);
		tmp1 = s[20]; tmp2 = s[21]; s[20] = bitselect(s[20] ^ s[22], s[20], s[21]); s[21] = bitselect(s[21] ^ s[23], s[21], s[22]); s[22] = bitselect(s[22] ^ s[24], s[22], s[23]); s[23] = bitselect(s[23] ^ tmp1, s[23], s[24]); s[24] = bitselect(s[24] ^ tmp2, s[24], tmp1);
		s[0] ^= keccak_round_constants35[i];
	}
}

__global__	__launch_bounds__(512)
void keccak256_gpu_hash_80(uint32_t threads, uint32_t startNounce,  uint32_t *const __restrict__ resNounce)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		const uint32_t nounce = startNounce + thread;
		uint2 bc[5], tmpxor[5], tmp1, tmp2;
		uint2 s[25];
		
		s[9] = make_uint2(c_PaddedMessage80[9].x, cuda_swab32(nounce));
		s[10] = make_uint2( 0x06, 0);
		s[16] = make_uint2(0, 0x80000000);

		tmpxor[0] = c_PaddedMessage80[0] ^ c_PaddedMessage80[5] ^ s[10];
		tmpxor[1] = c_PaddedMessage80[1] ^ c_PaddedMessage80[6] ^ s[16];
		tmpxor[2] = c_PaddedMessage80[2] ^ c_PaddedMessage80[7];
		tmpxor[3] = c_PaddedMessage80[3] ^ c_PaddedMessage80[8];
		tmpxor[4] = c_PaddedMessage80[4] ^ s[9];

		bc[0] = tmpxor[0] ^ ROL2(tmpxor[2], 1);
		bc[1] = tmpxor[1] ^ ROL2(tmpxor[3], 1);
		bc[2] = tmpxor[2] ^ ROL2(tmpxor[4], 1);
		bc[3] = tmpxor[3] ^ ROL2(tmpxor[0], 1);
		bc[4] = tmpxor[4] ^ ROL2(tmpxor[1], 1);

		tmp1 = c_PaddedMessage80[1] ^ bc[0];

		s[0] = c_PaddedMessage80[0] ^ bc[4];
		s[1] = ROL2(c_PaddedMessage80[6] ^ bc[0], 44);
		s[6] = ROL2(s[9] ^ bc[3], 20);
		s[9] = ROL2(bc[1], 61);
		s[22] = ROL2(bc[3], 39);
		s[14] = ROL2(bc[4], 18);
		s[20] = ROL2(c_PaddedMessage80[2] ^ bc[1], 62);
		s[2] = ROL2(bc[1], 43);
		s[12] = ROL2(bc[2], 25);
		s[13] = ROL8(bc[3]);
		s[19] = ROR8(bc[2]);
		s[23] = ROL2(bc[4], 41);
		s[15] = ROL2(c_PaddedMessage80[4] ^ bc[3], 27);
		s[4] = ROL2(bc[3], 14);
		s[24] = ROL2(bc[0], 2);
		s[21] = ROL2(c_PaddedMessage80[8] ^ bc[2], 55);
		s[8] = ROL2(s[16] ^ bc[0], 45);
		s[16] = ROL2(c_PaddedMessage80[5] ^ bc[4], 36);
		s[5] = ROL2(c_PaddedMessage80[3] ^ bc[2], 28);
		s[3] = ROL2( bc[2], 21);
		s[18] = ROL2(bc[1], 15);
		s[17] = ROL2(bc[0], 10);
		s[11] = ROL2(c_PaddedMessage80[7] ^ bc[1], 6);
		s[7] = ROL2(s[10] ^ bc[4], 3);
		s[10] = ROL2(tmp1, 1);

		tmp1 = s[0]; tmp2 = s[1]; s[0] = bitselect(s[0] ^ s[2], s[0], s[1]); s[1] = bitselect(s[1] ^ s[3], s[1], s[2]); s[2] = bitselect(s[2] ^ s[4], s[2], s[3]); s[3] = bitselect(s[3] ^ tmp1, s[3], s[4]); s[4] = bitselect(s[4] ^ tmp2, s[4], tmp1);
		tmp1 = s[5]; tmp2 = s[6]; s[5] = bitselect(s[5] ^ s[7], s[5], s[6]); s[6] = bitselect(s[6] ^ s[8], s[6], s[7]); s[7] = bitselect(s[7] ^ s[9], s[7], s[8]); s[8] = bitselect(s[8] ^ tmp1, s[8], s[9]); s[9] = bitselect(s[9] ^ tmp2, s[9], tmp1);
		tmp1 = s[10]; tmp2 = s[11]; s[10] = bitselect(s[10] ^ s[12], s[10], s[11]); s[11] = bitselect(s[11] ^ s[13], s[11], s[12]); s[12] = bitselect(s[12] ^ s[14], s[12], s[13]); s[13] = bitselect(s[13] ^ tmp1, s[13], s[14]); s[14] = bitselect(s[14] ^ tmp2, s[14], tmp1);
		tmp1 = s[15]; tmp2 = s[16]; s[15] = bitselect(s[15] ^ s[17], s[15], s[16]); s[16] = bitselect(s[16] ^ s[18], s[16], s[17]); s[17] = bitselect(s[17] ^ s[19], s[17], s[18]); s[18] = bitselect(s[18] ^ tmp1, s[18], s[19]); s[19] = bitselect(s[19] ^ tmp2, s[19], tmp1);
		tmp1 = s[20]; tmp2 = s[21]; s[20] = bitselect(s[20] ^ s[22], s[20], s[21]); s[21] = bitselect(s[21] ^ s[23], s[21], s[22]); s[22] = bitselect(s[22] ^ s[24], s[22], s[23]); s[23] = bitselect(s[23] ^ tmp1, s[23], s[24]); s[24] = bitselect(s[24] ^ tmp2, s[24], tmp1);
		s[0].x ^= 1;

#pragma unroll 2
		for (int i = 1; i < 23; i++) 
		{

#pragma unroll
			for (uint32_t x = 0; x < 5; x++)
				tmpxor[x] = s[x] ^ s[x + 5] ^ s[x + 10] ^ s[x + 15] ^ s[x + 20];

			bc[0] = tmpxor[0] ^ ROL2(tmpxor[2], 1);
			bc[1] = tmpxor[1] ^ ROL2(tmpxor[3], 1);
			bc[2] = tmpxor[2] ^ ROL2(tmpxor[4], 1);
			bc[3] = tmpxor[3] ^ ROL2(tmpxor[0], 1);
			bc[4] = tmpxor[4] ^ ROL2(tmpxor[1], 1);

			tmp1 = s[1] ^ bc[0];

			s[0] ^= bc[4];
			s[1] = ROL2(s[6] ^ bc[0], 44);
			s[6] = ROL2(s[9] ^ bc[3], 20);
			s[9] = ROL2(s[22] ^ bc[1], 61);
			s[22] = ROL2(s[14] ^ bc[3], 39);
			s[14] = ROL2(s[20] ^ bc[4], 18);
			s[20] = ROL2(s[2] ^ bc[1], 62);
			s[2] = ROL2(s[12] ^ bc[1], 43);
			s[12] = ROL2(s[13] ^ bc[2], 25);
			s[13] = ROL8(s[19] ^ bc[3]);
			s[19] = ROR8(s[23] ^ bc[2]);
			s[23] = ROL2(s[15] ^ bc[4], 41);
			s[15] = ROL2(s[4] ^ bc[3], 27);
			s[4] = ROL2(s[24] ^ bc[3], 14);
			s[24] = ROL2(s[21] ^ bc[0], 2);
			s[21] = ROL2(s[8] ^ bc[2], 55);
			s[8] = ROL2(s[16] ^ bc[0], 45);
			s[16] = ROL2(s[5] ^ bc[4], 36);
			s[5] = ROL2(s[3] ^ bc[2], 28);
			s[3] = ROL2(s[18] ^ bc[2], 21);
			s[18] = ROL2(s[17] ^ bc[1], 15);
			s[17] = ROL2(s[11] ^ bc[0], 10);
			s[11] = ROL2(s[7] ^ bc[1], 6);
			s[7] = ROL2(s[10] ^ bc[4], 3);
			s[10] = ROL2(tmp1, 1);

			tmp1 = s[0]; tmp2 = s[1]; s[0] = bitselect(s[0] ^ s[2], s[0], s[1]); s[1] = bitselect(s[1] ^ s[3], s[1], s[2]); s[2] = bitselect(s[2] ^ s[4], s[2], s[3]); s[3] = bitselect(s[3] ^ tmp1, s[3], s[4]); s[4] = bitselect(s[4] ^ tmp2, s[4], tmp1);
			tmp1 = s[5]; tmp2 = s[6]; s[5] = bitselect(s[5] ^ s[7], s[5], s[6]); s[6] = bitselect(s[6] ^ s[8], s[6], s[7]); s[7] = bitselect(s[7] ^ s[9], s[7], s[8]); s[8] = bitselect(s[8] ^ tmp1, s[8], s[9]); s[9] = bitselect(s[9] ^ tmp2, s[9], tmp1);
			tmp1 = s[10]; tmp2 = s[11]; s[10] = bitselect(s[10] ^ s[12], s[10], s[11]); s[11] = bitselect(s[11] ^ s[13], s[11], s[12]); s[12] = bitselect(s[12] ^ s[14], s[12], s[13]); s[13] = bitselect(s[13] ^ tmp1, s[13], s[14]); s[14] = bitselect(s[14] ^ tmp2, s[14], tmp1);
			tmp1 = s[15]; tmp2 = s[16]; s[15] = bitselect(s[15] ^ s[17], s[15], s[16]); s[16] = bitselect(s[16] ^ s[18], s[16], s[17]); s[17] = bitselect(s[17] ^ s[19], s[17], s[18]); s[18] = bitselect(s[18] ^ tmp1, s[18], s[19]); s[19] = bitselect(s[19] ^ tmp2, s[19], tmp1);
			tmp1 = s[20]; tmp2 = s[21]; s[20] = bitselect(s[20] ^ s[22], s[20], s[21]); s[21] = bitselect(s[21] ^ s[23], s[21], s[22]); s[22] = bitselect(s[22] ^ s[24], s[22], s[23]); s[23] = bitselect(s[23] ^ tmp1, s[23], s[24]); s[24] = bitselect(s[24] ^ tmp2, s[24], tmp1);
			s[0] ^= keccak_round_constants35[i];
		}
		uint2 t[5];
		t[0] = s[0] ^ s[5] ^ s[10] ^ s[15] ^ s[20];
		t[1] = s[1] ^ s[6] ^ s[11] ^ s[16] ^ s[21];
		t[2] = s[2] ^ s[7] ^ s[12] ^ s[17] ^ s[22];
		t[3] = s[3] ^ s[8] ^ s[13] ^ s[18] ^ s[23];
		t[4] = s[4] ^ s[9] ^ s[14] ^ s[19] ^ s[24];

		s[0] ^= t[4] ^ ROL2(t[1], 1);
		s[18] ^= t[2] ^ ROL2(t[4], 1);
		s[24] ^= t[3] ^ ROL2(t[0], 1);

		s[3] = ROL2(s[18], 21) ^ ((~ROL2(s[24], 14)) & s[0]);


		if (devectorize(s[3]) <= ((uint64_t*)pTarget)[3])
		{
			uint32_t tmp = atomicCAS(resNounce, 0xffffffff, nounce);
			if (tmp != 0xffffffff)
				resNounce[1] = nounce;
		}
	}
}

__device__ __forceinline__
uint64_t xor5(uint64_t a, uint64_t b, uint64_t c, uint64_t d, uint64_t e)
{
	uint64_t result;
	asm("xor.b64 %0, %1, %2;" : "=l"(result) : "l"(d), "l"(e));
	asm("xor.b64 %0, %0, %1;" : "+l"(result) : "l"(c));
	asm("xor.b64 %0, %0, %1;" : "+l"(result) : "l"(b));
	asm("xor.b64 %0, %0, %1;" : "+l"(result) : "l"(a));
	return result;
}

__device__ __forceinline__
uint2 xor3x2(const uint2 a, const uint2 b, const uint2 c)
{
	uint2 result; // = result = a^b^c;
	asm("lop3.b32 %0, %1, %2, %3, 0x96;" : "=r"(result.x) : "r"(a.x), "r"(b.x), "r"(c.x)); //0x96 = 0xF0 ^ 0xCC ^ 0xAA
	asm("lop3.b32 %0, %1, %2, %3, 0x96;" : "=r"(result.y) : "r"(a.y), "r"(b.y), "r"(c.y)); //0x96 = 0xF0 ^ 0xCC ^ 0xAA
	return result;
}

__device__ __forceinline__
uint2 chi2(const uint2 a, const uint2 b, const uint2 c)
{ //keccak - chi
//	uint2 result = a ^ (~b) & c;
//	0xD2 ^ ((~0xCC) & 0xAA)
	uint2 result;
	asm("lop3.b32 %0, %1, %2, %3, 0xD2;" : "=r"(result.x) : "r"(a.x), "r"(b.x), "r"(c.x)); //0x96 = 0xF0 ^ 0xCC ^ 0xAA
	asm("lop3.b32 %0, %1, %2, %3, 0xD2;" : "=r"(result.y) : "r"(a.y), "r"(b.y), "r"(c.y)); //0x96 = 0xF0 ^ 0xCC ^ 0xAA

	return result;
}


__global__	__launch_bounds__(1024,1)
void keccak256_gpu_hash_ZP(uint32_t threads, uint32_t *const __restrict__ resNounce, const uint2 highTarget)
{
	const uint2 keccak_round_constants35[24] = {
		{ 0x00000001ul, 0x00000000 }, { 0x00008082ul, 0x00000000 },
		{ 0x0000808aul, 0x80000000 }, { 0x80008000ul, 0x80000000 },
		{ 0x0000808bul, 0x00000000 }, { 0x80000001ul, 0x00000000 },
		{ 0x80008081ul, 0x80000000 }, { 0x00008009ul, 0x80000000 },
		{ 0x0000008aul, 0x00000000 }, { 0x00000088ul, 0x00000000 },
		{ 0x80008009ul, 0x00000000 }, { 0x8000000aul, 0x00000000 },
		{ 0x8000808bul, 0x00000000 }, { 0x0000008bul, 0x80000000 },
		{ 0x00008089ul, 0x80000000 }, { 0x00008003ul, 0x80000000 },
		{ 0x00008002ul, 0x80000000 }, { 0x00000080ul, 0x80000000 },
		{ 0x0000800aul, 0x00000000 }, { 0x8000000aul, 0x80000000 },
		{ 0x80008081ul, 0x80000000 }, { 0x00008080ul, 0x80000000 },
		{ 0x80000001ul, 0x00000000 }, { 0x80008008ul, 0x80000000 }
	};

	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		const uint32_t nounce = thread;
		uint2 bc[5], tmpxor[5], tmp1, tmp2;
		uint2 s[25];

		s[11] = c_PaddedMessageZP_PRE[5];
		uint2 k;
		k.x = cuda_swab32(nounce);
		k.y = 6;
		s[16].x = 0;
		s[16].y = 0x80000000;


		uint2 i = c_PaddedMessageZP_PRE[12] ^ k;
		bc[0] = c_PaddedMessageZP_PRE[2] ^ ROL2(i, 1);
		bc[1] = c_PaddedMessageZP_PRE[1];
		bc[2] = i ^ ROL2(c_PaddedMessageZP_PRE[3], 1);

		tmp1 = c_PaddedMessageZP_PRE[8] ^ bc[0];

		s[1] = ROL2(c_PaddedMessageZP_PRE[10] ^ bc[0], 44);
		s[2] = ROL2(k ^ bc[1], 43);
		s[12] = ROL2(bc[2], 25);
		s[19] = ROR8(bc[2]);
		s[24] = ROL2(bc[0], 2);
		s[21] = ROL2(c_PaddedMessageZP_PRE[19] ^ bc[2], 55);
		s[8] = ROL2(s[16] ^ bc[0], 45);
		s[5] = ROL2(c_PaddedMessageZP_PRE[17] ^ bc[2], 28);
		s[3] = ROL2( bc[2], 21);
		s[17] = ROL2(s[11] ^ bc[0], 10);
		s[10] = ROL2(tmp1, 1);


		s[0] = c_PaddedMessageZP_PRE[0];
		s[4] = c_PaddedMessageZP_PRE[4];
		s[6] = c_PaddedMessageZP_PRE[6];
		s[7] = c_PaddedMessageZP_PRE[7];
		s[9] = c_PaddedMessageZP_PRE[9];
		s[11] = c_PaddedMessageZP_PRE[11];
		s[13] = c_PaddedMessageZP_PRE[13];
		s[14] = c_PaddedMessageZP_PRE[14];
		s[15] = c_PaddedMessageZP_PRE[15];
		s[16] = c_PaddedMessageZP_PRE[16];
		s[18] = c_PaddedMessageZP_PRE[18];
		s[20] = c_PaddedMessageZP_PRE[20];
		s[23] = c_PaddedMessageZP_PRE[21];
		s[22] = c_PaddedMessageZP_PRE[22];


		tmp1 = s[0];
		tmp2 = s[1];
		s[0] = chi2(s[0], s[1], s[2]);
		s[1] = chi2(s[1], s[2], s[3]);
		s[2] = chi2(s[2], s[3], s[4]);
		s[3] = chi2(s[3], s[4], tmp1);
		s[4] = chi2(s[4], tmp1, tmp2);


#pragma unroll
		for (int j = 5; j < 25; j += 5)
		{
			tmp1 = s[j];
			tmp2 = s[j + 1];
			s[j] = chi2(s[j], s[j + 1], s[j + 2]);
			s[j + 1] = chi2(s[j + 1], s[j + 2], s[j + 3]);
			s[j + 2] = chi2(s[j + 2], s[j + 3], s[j + 4]);
			s[j + 3] = chi2(s[j + 3], s[j + 4], tmp1);
			s[j + 4] = chi2(s[j + 4], tmp1, tmp2);
		}

		s[0].x ^= 1;

#pragma unroll
		for (int i = 1; i < 23; i++) 
		{

#pragma unroll
			for (int j = 0; j < 5; j++) {
				tmpxor[j] = vectorize(xor5(devectorize(s[j]), devectorize(s[j + 5]), devectorize(s[j + 10]), devectorize(s[j + 15]), devectorize(s[j + 20])));
			}

			for (int j = 0; j < 5; j++) {
				bc[j] = ROL2(tmpxor[j], 1);
			}
			s[4] = xor3x2(s[4], tmpxor[3], bc[0]); s[9] = xor3x2(s[9], tmpxor[3], bc[0]); s[14] = xor3x2(s[14], tmpxor[3], bc[0]); s[19] = xor3x2(s[19], tmpxor[3], bc[0]); s[24] = xor3x2(s[24], tmpxor[3], bc[0]);
			s[0] = xor3x2(s[0], tmpxor[4], bc[1]); s[5] = xor3x2(s[5], tmpxor[4], bc[1]); s[10] = xor3x2(s[10], tmpxor[4], bc[1]); s[15] = xor3x2(s[15], tmpxor[4], bc[1]); s[20] = xor3x2(s[20], tmpxor[4], bc[1]);
			s[1] = xor3x2(s[1], tmpxor[0], bc[2]); s[6] = xor3x2(s[6], tmpxor[0], bc[2]); s[11] = xor3x2(s[11], tmpxor[0], bc[2]); s[16] = xor3x2(s[16], tmpxor[0], bc[2]); s[21] = xor3x2(s[21], tmpxor[0], bc[2]);
			s[2] = xor3x2(s[2], tmpxor[1], bc[3]); s[7] = xor3x2(s[7], tmpxor[1], bc[3]); s[12] = xor3x2(s[12], tmpxor[1], bc[3]); s[17] = xor3x2(s[17], tmpxor[1], bc[3]); s[22] = xor3x2(s[22], tmpxor[1], bc[3]);
			s[3] = xor3x2(s[3], tmpxor[2], bc[4]); s[8] = xor3x2(s[8], tmpxor[2], bc[4]); s[13] = xor3x2(s[13], tmpxor[2], bc[4]); s[18] = xor3x2(s[18], tmpxor[2], bc[4]); s[23] = xor3x2(s[23], tmpxor[2], bc[4]);

			tmp1 = s[1];
			s[1] = ROL2(s[6], 44);	s[6] = ROL2(s[9], 20);	s[9] = ROL2(s[22], 61);	s[22] = ROL2(s[14], 39);
			s[14] = ROL2(s[20], 18);	s[20] = ROL2(s[2], 62);	s[2] = ROL2(s[12], 43);	s[12] = ROL2(s[13], 25);
			s[13] = ROL8(s[19]);	s[19] = ROR8(s[23]);	s[23] = ROL2(s[15], 41);	s[15] = ROL2(s[4], 27);
			s[4] = ROL2(s[24], 14);	s[24] = ROL2(s[21], 2);	s[21] = ROL2(s[8], 55);	s[8] = ROL2(s[16], 45);
			s[16] = ROL2(s[5], 36);	s[5] = ROL2(s[3], 28);	s[3] = ROL2(s[18], 21);	s[18] = ROL2(s[17], 15);
			s[17] = ROL2(s[11], 10);	s[11] = ROL2(s[7], 6);	s[7] = ROL2(s[10], 3);	s[10] = ROL2(tmp1, 1);

			#pragma unroll
			for (int j = 0; j < 25; j += 5)
			{
				tmp1 = s[j];
				tmp2 = s[j + 1];
				s[j] = chi2(s[j], s[j + 1], s[j + 2]);
				s[j + 1] = chi2(s[j + 1], s[j + 2], s[j + 3]);
				s[j + 2] = chi2(s[j + 2], s[j + 3], s[j + 4]);
				s[j + 3] = chi2(s[j + 3], s[j + 4], tmp1);
				s[j + 4] = chi2(s[j + 4], tmp1, tmp2);
			}
			s[0].x ^= keccak_round_constants35[i].x;
			s[0].y ^= keccak_round_constants35[i].y;
		}

#pragma unroll
		for (int j = 0; j < 5; j++) {
			tmpxor[j] = vectorize(xor5(devectorize(s[j]), devectorize(s[j + 5]), devectorize(s[j + 10]), devectorize(s[j + 15]), devectorize(s[j + 20])));
		}

		for (int j = 0; j < 5; j++) {
			bc[j] = ROL2(tmpxor[j], 1);
		}
		s[4] = xor3x2(s[4], tmpxor[3], bc[0]); s[9] = xor3x2(s[9], tmpxor[3], bc[0]); s[14] = xor3x2(s[14], tmpxor[3], bc[0]); s[19] = xor3x2(s[19], tmpxor[3], bc[0]); s[24] = xor3x2(s[24], tmpxor[3], bc[0]);
		s[0] = xor3x2(s[0], tmpxor[4], bc[1]); s[5] = xor3x2(s[5], tmpxor[4], bc[1]); s[10] = xor3x2(s[10], tmpxor[4], bc[1]); s[15] = xor3x2(s[15], tmpxor[4], bc[1]); s[20] = xor3x2(s[20], tmpxor[4], bc[1]);
		s[1] = xor3x2(s[1], tmpxor[0], bc[2]); s[6] = xor3x2(s[6], tmpxor[0], bc[2]); s[11] = xor3x2(s[11], tmpxor[0], bc[2]); s[16] = xor3x2(s[16], tmpxor[0], bc[2]); s[21] = xor3x2(s[21], tmpxor[0], bc[2]);
		s[2] = xor3x2(s[2], tmpxor[1], bc[3]); s[7] = xor3x2(s[7], tmpxor[1], bc[3]); s[12] = xor3x2(s[12], tmpxor[1], bc[3]); s[17] = xor3x2(s[17], tmpxor[1], bc[3]); s[22] = xor3x2(s[22], tmpxor[1], bc[3]);
		s[3] = xor3x2(s[3], tmpxor[2], bc[4]); s[8] = xor3x2(s[8], tmpxor[2], bc[4]); s[13] = xor3x2(s[13], tmpxor[2], bc[4]); s[18] = xor3x2(s[18], tmpxor[2], bc[4]); s[23] = xor3x2(s[23], tmpxor[2], bc[4]);

		tmp1 = s[1];
		s[1] = ROL2(s[6], 44);	s[6] = ROL2(s[9], 20);	s[9] = ROL2(s[22], 61);	s[22] = ROL2(s[14], 39);
		s[14] = ROL2(s[20], 18);	s[20] = ROL2(s[2], 62);	s[2] = ROL2(s[12], 43);	s[12] = ROL2(s[13], 25);
		s[13] = ROL8(s[19]);	s[19] = ROR8(s[23]);	s[23] = ROL2(s[15], 41);	s[15] = ROL2(s[4], 27);
		s[4] = ROL2(s[24], 14);	s[24] = ROL2(s[21], 2);	s[21] = ROL2(s[8], 55);	s[8] = ROL2(s[16], 45);
		s[16] = ROL2(s[5], 36);	s[5] = ROL2(s[3], 28);	s[3] = ROL2(s[18], 21);	s[18] = ROL2(s[17], 15);
		s[17] = ROL2(s[11], 10);	s[11] = ROL2(s[7], 6);	s[7] = ROL2(s[10], 3);	s[10] = ROL2(tmp1, 1);

#pragma unroll
		for (int j = 0; j < 25; j += 5)
		{
			tmp1 = s[j];
			tmp2 = s[j + 1];
			s[j] = chi2(s[j], s[j + 1], s[j + 2]);
			s[j + 1] = chi2(s[j + 1], s[j + 2], s[j + 3]);
			s[j + 2] = chi2(s[j + 2], s[j + 3], s[j + 4]);
			s[j + 3] = chi2(s[j + 3], s[j + 4], tmp1);
			s[j + 4] = chi2(s[j + 4], tmp1, tmp2);
		}
		s[0].x ^= keccak_round_constants35[23].x;
		s[0].y ^= keccak_round_constants35[23].y;

		if (cuda_swab32(s[0].x) <= (highTarget.x) && ( cuda_swab32(s[0].y) <= (highTarget.y)) )
		{
			const uint32_t tmp = atomicExch(&resNounce[0], nounce);
			if (tmp != UINT32_MAX)
				resNounce[1] = tmp;
		}
	}
}

__host__
void keccak256_cpu_hash_ZP(int thr_id, uint32_t threads, uint32_t *h_nounce, uint2 hightarget)
{
	CUDA_SAFE_CALL(cudaMemsetAsync(d_KNonce[thr_id], 0xff, 2 * sizeof(uint32_t), gpustream[thr_id]));
	const uint32_t threadsperblock = 1024;

	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);
	keccak256_gpu_hash_ZP << <grid, block, 0, gpustream[thr_id] >> >(threads, d_KNonce[thr_id], hightarget);
//	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	CUDA_SAFE_CALL(cudaMemcpy(h_nounce, d_KNonce[thr_id], 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
}

__host__
void keccak256_setBlock_ZP(int thr_id, void *pdata)
{
	unsigned char PaddedMessage[104];
	memcpy(PaddedMessage, pdata, 104);
	

	uint64_t* PaddedMessageZP = (uint64_t*)PaddedMessage;
	uint64_t bc[5], tmpxor[5];
	uint64_t s[24];

	s[11] = PaddedMessageZP[11];
	s[12] = PaddedMessageZP[2] ^ PaddedMessageZP[7];			//make_uint2(cuda_swab32(nounce), 0x06);
	s[16] = (uint64_t)1 << 63;//(uint64_t)make_uint2(0, 0x80000000);

	s[2] = tmpxor[0] = PaddedMessageZP[0] ^ PaddedMessageZP[5] ^ PaddedMessageZP[10];
	tmpxor[1] = PaddedMessageZP[1] ^ PaddedMessageZP[6] ^ s[11] ^ s[16];
	tmpxor[3] = PaddedMessageZP[3] ^ PaddedMessageZP[8];
	s[3] = tmpxor[4] = PaddedMessageZP[4] ^ PaddedMessageZP[9];

	bc[1] = tmpxor[1] ^ ROTL64(tmpxor[3], 1);
	bc[3] = tmpxor[3] ^ ROTL64(tmpxor[0], 1);
	bc[4] = tmpxor[4] ^ ROTL64(tmpxor[1], 1);

//	tmp1 = c_PaddedMessageZP[1] ^ bc[0];
	s[1] = bc[1];


	s[0] = PaddedMessageZP[0] ^ bc[4];
//	s[1] = ROL2(c_PaddedMessageZP[6] ^ bc[0], 44);
	s[6] = ROTL64(PaddedMessageZP[9] ^ bc[3], 20);
	s[9] = ROTL64(bc[1], 61);
	s[22] = ROTL64(bc[3], 39);
	s[14] = ROTL64(bc[4], 18);
	s[20] = ROTL64(PaddedMessageZP[2] ^ bc[1], 62);
//	s[2] = ROTL64(s[12] ^ bc[1], 43);
//	s[12] = ROTL64(bc[2], 25);
	s[13] = ROTL64(bc[3], 8);
//	s[19] = ROR8(bc[2]);
	s[21] = ROTL64(bc[4], 41);
	s[15] = ROTL64(PaddedMessageZP[4] ^ bc[3], 27);
	s[4] = ROTL64(bc[3], 14);
//	s[24] = ROL2(bc[0], 2);
//	s[21] = ROL2(PaddedMessageZP[8] ^ bc[2], 55);
//	s[8] = ROL2(s[16] ^ bc[0], 45);
	s[16] = ROTL64(PaddedMessageZP[5] ^ bc[4], 36);
//	s[5] = ROL2(c_PaddedMessageZP[3] ^ bc[2], 28);
//	s[3] = ROL2(bc[2], 21);
	s[18] = ROTL64(bc[1], 15);
//	s[17] = ROL2(s[11] ^ bc[0], 10);
	s[11] = ROTL64(PaddedMessageZP[7] ^ bc[1], 6);
	s[7] = ROTL64(PaddedMessageZP[10] ^ bc[4], 3);
//	s[10] = ROL2(tmp1, 1);

	s[5] = PaddedMessageZP[11];
	s[8] = PaddedMessageZP[1];
	s[10] = PaddedMessageZP[6];
	s[19] = PaddedMessageZP[8];
	s[17] = PaddedMessageZP[3];
	CUDA_SAFE_CALL(cudaMemcpyToSymbolAsync(c_PaddedMessageZP_PRE, &s[0], 23 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice, gpustream[thr_id]));

//	if (opt_debug)
//		CUDA_SAFE_CALL(cudaDeviceSynchronize());
}


__host__
void keccak256_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *h_nounce)
{
	CUDA_SAFE_CALL(cudaMemsetAsync(d_KNonce[thr_id], 0xff, 2 * sizeof(uint32_t), gpustream[thr_id]));
	const uint32_t threadsperblock = 512;

	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);
	keccak256_gpu_hash_80<<<grid, block, 0, gpustream[thr_id]>>>(threads, startNounce, d_KNonce[thr_id]);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	CUDA_SAFE_CALL(cudaMemcpy(h_nounce, d_KNonce[thr_id], 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
}



__global__ __launch_bounds__(256,3)
void keccak256_gpu_hash_32(uint32_t threads, uint32_t startNounce, uint64_t *outputHash)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
//	if (thread < threads)
	{
		uint2 keccak_gpu_state[25];
		#pragma unroll 25
		for (int i = 0; i<25; i++) {
			if (i<4) keccak_gpu_state[i] = vectorize(outputHash[i*threads+thread]);
			else     keccak_gpu_state[i] = UINT2(0, 0);
		}
		keccak_gpu_state[4]  = UINT2(0x06, 0);
		keccak_gpu_state[16] = UINT2(0, 0x80000000);
		keccak_block(keccak_gpu_state);

		#pragma unroll 4
		for (int i=0; i<4; i++)
			outputHash[i*threads+thread] = devectorize(keccak_gpu_state[i]);
	}
}

__host__
void keccak256_cpu_hash_32(int thr_id, uint32_t threads, uint32_t startNounce, uint64_t *d_outputHash)
{
	const uint32_t threadsperblock = 256;

	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);

	keccak256_gpu_hash_32 <<<grid, block, 0, gpustream[thr_id]>>> (threads, startNounce, d_outputHash);
	CUDA_SAFE_CALL(cudaGetLastError());
}

__host__
void keccak256_setBlock_80(int thr_id, void *pdata,const void *pTargetIn)
{
	unsigned char PaddedMessage[80];
	memcpy(PaddedMessage, pdata, 80);
	CUDA_SAFE_CALL(cudaMemcpyToSymbolAsync(pTarget, pTargetIn, 8 * sizeof(uint32_t), 0, cudaMemcpyHostToDevice, gpustream[thr_id]));
	CUDA_SAFE_CALL(cudaMemcpyToSymbolAsync(c_PaddedMessage80, PaddedMessage, 10 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice, gpustream[thr_id]));
	if(opt_debug)
		CUDA_SAFE_CALL(cudaDeviceSynchronize());
}

__host__
void keccak256_cpu_init(int thr_id, uint32_t threads)
{
	CUDA_SAFE_CALL(cudaMalloc(&d_KNonce[thr_id], 2*sizeof(uint32_t)));
}