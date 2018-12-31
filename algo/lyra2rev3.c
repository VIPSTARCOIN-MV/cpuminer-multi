#include <memory.h>
#ifndef SSE
#include <immintrin.h>
#else
#ifdef SSE3
#include <tmmintrin.h>
#else
#include <emmintrin.h> //SSE2
#endif
#endif

#include "sha3/sph_blake.h"
#include "sha3/sph_cubehash.h"
#include "sha3/sph_bmw.h"

#include "lyra2/Lyra2.h"

#include "miner.h"

#ifndef SSE

void lyra2rev3_hash_AVX(void *state, const void *input, __m256i *wholeMatrix, int32_t *flag, __m256i *wholeMatrix2)
{
	__m256i hashA[8], hashB[8];

	if (flag) {
		sph_blake256_80_init(wholeMatrix2, input, 80);
		flag = 0;
	}

	sph_blake256_80_AVX(hashA, input, 80, wholeMatrix2);

	hashB[0] = _mm256_unpacklo_epi32(hashA[0], hashA[1]); // 00 01 08 09 20 21 28 29
	hashB[1] = _mm256_unpacklo_epi32(hashA[2], hashA[3]); // 02 03 0A 0B 22 23 2A 2B
	hashB[2] = _mm256_unpacklo_epi32(hashA[4], hashA[5]); // 04 05 0C 0D 24 25 2C 2D
	hashB[3] = _mm256_unpacklo_epi32(hashA[6], hashA[7]); // 06 07 0E 0F 26 27 2E 2F
	hashB[4] = _mm256_unpackhi_epi32(hashA[0], hashA[1]); // 10 11 18 19 30 31 38 39
	hashB[5] = _mm256_unpackhi_epi32(hashA[2], hashA[3]); // 12 13 1A 1B 32 33 3A 3B
	hashB[6] = _mm256_unpackhi_epi32(hashA[4], hashA[5]); // 14 15 1C 1D 34 35 3C 3D
	hashB[7] = _mm256_unpackhi_epi32(hashA[6], hashA[7]); // 16 17 1E 1F 36 37 3E 3F

  LYRA2v2(hashA + i, hashB + i, wholeMatrix);

	for (int i = 0; i < 8; i++)
	{
		sph_cubehash256(hashB + i, hashA + i, 32);
	}

	hashB[0] = _mm256_unpacklo_epi64(hashA[0], hashA[2]); // 00 01 10 11 04 05 14 15
	hashB[1] = _mm256_unpackhi_epi64(hashA[0], hashA[2]); // 02 03 12 13 06 07 16 17
	hashB[2] = _mm256_unpacklo_epi64(hashA[1], hashA[3]); // 08 09 18 19 0C 0D 1C 1D
	hashB[3] = _mm256_unpackhi_epi64(hashA[1], hashA[3]); // 0A 0B 1A 1B 0E 0F 1E 1F
	hashB[4] = _mm256_unpacklo_epi64(hashA[4], hashA[6]); // 20 21 30 31 24 25 34 35
	hashB[5] = _mm256_unpackhi_epi64(hashA[4], hashA[6]); // 22 23 32 33 26 27 36 37
	hashB[6] = _mm256_unpacklo_epi64(hashA[5], hashA[7]); // 28 29 38 39 2C 2D 3C 3D
	hashB[7] = _mm256_unpackhi_epi64(hashA[5], hashA[7]); // 2A 2B 3A 3B 2E 2F 3E 3F

	hashA[0] = _mm256_permute2x128_si256(hashB[0], hashB[2], 0x20); // 00 01 10 11 08 09 18 19
	hashA[1] = _mm256_permute2x128_si256(hashB[1], hashB[3], 0x20); // 02 03 12 13 0A 0B 1A 1B
	hashA[2] = _mm256_permute2x128_si256(hashB[0], hashB[2], 0x31); // 04 05 14 15 0C 0D 1C 1D
	hashA[3] = _mm256_permute2x128_si256(hashB[1], hashB[3], 0x31); // 06 07 16 17 0E 0F 1E 1F
	hashA[4] = _mm256_permute2x128_si256(hashB[4], hashB[6], 0x20); // 20 21 30 31 28 29 38 39
	hashA[5] = _mm256_permute2x128_si256(hashB[5], hashB[7], 0x20); // 22 23 32 33 2A 2B 3A 3B
	hashA[6] = _mm256_permute2x128_si256(hashB[4], hashB[6], 0x31); // 24 25 34 35 2C 2D 3C 3D
	hashA[7] = _mm256_permute2x128_si256(hashB[5], hashB[7], 0x31); // 26 27 36 37 2E 2F 3E 3F

	hashB[0] = _mm256_permute2x128_si256(hashA[0], hashA[4], 0x20); // 00 01 10 11 20 21 30 31
	hashB[1] = _mm256_permute2x128_si256(hashA[1], hashA[5], 0x20); // 02 03 12 13 22 23 32 33
	hashB[2] = _mm256_permute2x128_si256(hashA[2], hashA[6], 0x20); // 04 05 14 15 24 25 34 35
	hashB[3] = _mm256_permute2x128_si256(hashA[3], hashA[7], 0x20); // 06 07 16 17 26 27 36 37
	hashB[4] = _mm256_permute2x128_si256(hashA[0], hashA[4], 0x31); // 08 09 18 19 28 29 38 39
	hashB[5] = _mm256_permute2x128_si256(hashA[1], hashA[5], 0x31); // 0A 0B 1A 1B 2A 2B 3A 3B
	hashB[6] = _mm256_permute2x128_si256(hashA[2], hashA[6], 0x31); // 0C 0D 1C 1D 2C 2D 3C 3D
	hashB[7] = _mm256_permute2x128_si256(hashA[3], hashA[7], 0x31); // 0E 0F 1E 1F 2E 2F 3D 3F

	LYRA2v2(hashA + i, hashB + i, wholeMatrix);

	sph_bmw256_AVX(hashB, hashA, 32);

	hashA[0] = _mm256_unpacklo_epi32(hashB[0], hashB[1]); // 00 01 10 11 40 41 50 51
	hashA[1] = _mm256_unpackhi_epi32(hashB[0], hashB[1]); // 20 21 30 31 60 61 70 71
	hashA[2] = _mm256_unpacklo_epi32(hashB[2], hashB[3]); // 02 03 12 13 42 43 52 53
	hashA[3] = _mm256_unpackhi_epi32(hashB[2], hashB[3]); // 22 23 32 33 62 63 72 73
	hashA[4] = _mm256_unpacklo_epi32(hashB[4], hashB[5]); // 04 05 14 15 44 45 54 55
	hashA[5] = _mm256_unpackhi_epi32(hashB[4], hashB[5]); // 24 25 34 35 64 65 74 75
	hashA[6] = _mm256_unpacklo_epi32(hashB[6], hashB[7]); // 06 07 16 17 46 47 56 57
	hashA[7] = _mm256_unpackhi_epi32(hashB[6], hashB[7]); // 26 27 36 37 66 67 76 77

	hashB[0] = _mm256_unpacklo_epi64(hashA[0], hashA[2]); // 00 01 02 03 40 41 42 43
	hashB[1] = _mm256_unpackhi_epi64(hashA[0], hashA[2]); // 10 11 12 13 50 51 52 53
	hashB[2] = _mm256_unpacklo_epi64(hashA[1], hashA[3]); // 20 21 22 23 60 61 62 63
	hashB[3] = _mm256_unpackhi_epi64(hashA[1], hashA[3]); // 30 31 32 33 70 71 72 73
	hashB[4] = _mm256_unpacklo_epi64(hashA[4], hashA[6]); // 04 05 06 07 44 45 46 47
	hashB[5] = _mm256_unpackhi_epi64(hashA[4], hashA[6]); // 14 15 16 17 54 55 56 57
	hashB[6] = _mm256_unpacklo_epi64(hashA[5], hashA[7]); // 24 25 26 27 64 65 66 67
	hashB[7] = _mm256_unpackhi_epi64(hashA[5], hashA[7]); // 34 35 36 37 74 75 76 77

	hashA[0] = _mm256_permute2x128_si256(hashB[0], hashB[4], 0x20); // 00 01 02 03 04 05 06 07
	hashA[1] = _mm256_permute2x128_si256(hashB[1], hashB[5], 0x20); // 10 11 12 13 14 15 16 17
	hashA[2] = _mm256_permute2x128_si256(hashB[2], hashB[6], 0x20); // 20 21 22 23 24 25 26 27
	hashA[3] = _mm256_permute2x128_si256(hashB[3], hashB[7], 0x20); // 30 31 32 33 34 35 36 37
	hashA[4] = _mm256_permute2x128_si256(hashB[0], hashB[4], 0x31); // 40 41 42 43 44 45 46 47
	hashA[5] = _mm256_permute2x128_si256(hashB[1], hashB[5], 0x31); // 50 51 52 53 54 55 56 57
	hashA[6] = _mm256_permute2x128_si256(hashB[2], hashB[6], 0x31); // 60 61 62 63 64 65 66 67
	hashA[7] = _mm256_permute2x128_si256(hashB[3], hashB[7], 0x31); // 70 71 72 73 74 75 76 77

	for (int i = 0; i < 8; i++)
		_mm256_storeu_si256(((__m256i*)state) + i, hashA[i]);
	_mm256_zeroupper();
}
#else

void lyra2rev3_hash_SSE(void *state, const void *input, __m128i *wholeMatrix, int32_t *flag, __m128i *wholeMatrix2)
{
	__m128i hash[32];
	__m128i *hashA = hash;
	__m128i *hashB = hash + 16;

	if (flag) {
		sph_blake256_80_init(wholeMatrix2, input, 80);
		flag = 0;
	}

	for (int j = 0; j < 2; j++, hashA += 8, hashB += 8)
	{
		sph_blake256_80_SSE2(hashA, input, 80, wholeMatrix2);

		hashB[0] = _mm_unpacklo_epi32(hashA[0], hashA[1]); // 00 01 08 09
		hashB[1] = _mm_unpacklo_epi32(hashA[2], hashA[3]); // 02 03 0A 0B
		hashB[2] = _mm_unpacklo_epi32(hashA[4], hashA[5]); // 04 05 0C 0D
		hashB[3] = _mm_unpacklo_epi32(hashA[6], hashA[7]); // 06 07 0E 0F
		hashB[4] = _mm_unpackhi_epi32(hashA[0], hashA[1]); // 10 11 18 19
		hashB[5] = _mm_unpackhi_epi32(hashA[2], hashA[3]); // 12 13 1A 1B
		hashB[6] = _mm_unpackhi_epi32(hashA[4], hashA[5]); // 14 15 1C 1D
		hashB[7] = _mm_unpackhi_epi32(hashA[6], hashA[7]); // 16 17 1E 1F

#ifdef SSE3
		LYRA2v2_SSSE3(hashB + i, hashA + i, wholeMatrix);
#else
		LYRA2v2_SSE2(hashB + i, hashA + i, wholeMatrix);
#endif

		for (int i = 0; i < 8; i += 2)
		{
			sph_cubehash256_SSE2(hashA + i, hashB + i, 32);
		}

		hashA[0] = _mm_unpacklo_epi64(hashB[0], hashB[2]); // 00 01 08 09
		hashA[1] = _mm_unpackhi_epi64(hashB[0], hashB[2]); // 02 03 0A 0B
		hashA[2] = _mm_unpacklo_epi64(hashB[1], hashB[3]); // 04 05 0C 0D
		hashA[3] = _mm_unpackhi_epi64(hashB[1], hashB[3]); // 06 07 0E 0F
		hashA[4] = _mm_unpacklo_epi64(hashB[4], hashB[6]); // 10 11 18 19
		hashA[5] = _mm_unpackhi_epi64(hashB[4], hashB[6]); // 12 13 1A 1B
		hashA[6] = _mm_unpacklo_epi64(hashB[5], hashB[7]); // 14 15 1C 1D
		hashA[7] = _mm_unpackhi_epi64(hashB[5], hashB[7]); // 16 17 1E 1F

#ifdef SSE3
		LYRA2v2_SSSE3(hashB + i, hashA + i, wholeMatrix);
#else
		LYRA2v2_SSE2(hashB + i, hashA + i, wholeMatrix);
#endif

		sph_bmw256_SSE2(hashA, hashB, 32);

		hashB[0] = _mm_unpacklo_epi32(hashA[0], hashA[1]); // 00 01 08 09
		hashB[1] = _mm_unpacklo_epi32(hashA[2], hashA[3]); // 02 03 0A 0B
		hashB[2] = _mm_unpacklo_epi32(hashA[4], hashA[5]); // 04 05 0C 0D
		hashB[3] = _mm_unpacklo_epi32(hashA[6], hashA[7]); // 06 07 0E 0F
		hashB[4] = _mm_unpackhi_epi32(hashA[0], hashA[1]); // 10 11 18 19
		hashB[5] = _mm_unpackhi_epi32(hashA[2], hashA[3]); // 12 13 1A 1B
		hashB[6] = _mm_unpackhi_epi32(hashA[4], hashA[5]); // 14 15 1C 1D
		hashB[7] = _mm_unpackhi_epi32(hashA[6], hashA[7]); // 16 17 1E 1F

		hashA[0] = _mm_unpacklo_epi64(hashB[0], hashB[1]); // 00 01 02 03
		hashA[1] = _mm_unpacklo_epi64(hashB[2], hashB[3]); // 04 05 06 07
		hashA[2] = _mm_unpackhi_epi64(hashB[0], hashB[1]); // 08 09 0A 0B
		hashA[3] = _mm_unpackhi_epi64(hashB[2], hashB[3]); // 0C 0D 0E 0F
		hashA[4] = _mm_unpacklo_epi64(hashB[4], hashB[5]); // 10 11 12 13
		hashA[5] = _mm_unpacklo_epi64(hashB[6], hashB[7]); // 14 15 16 17
		hashA[6] = _mm_unpackhi_epi64(hashB[4], hashB[5]); // 18 19 1A 1B
		hashA[7] = _mm_unpackhi_epi64(hashB[6], hashB[7]); // 1C 1D 1E 1F

		for (int i = 0; i < 8; i++)
			_mm_storeu_si128(((__m128i*)state) + i + j * 8, hashA[i]);
	}
}
#endif

int scanhash_lyra2rev3(int thr_id, struct work *work, uint32_t max_nonce, uint64_t *hashes_done)
{
	uint32_t _ALIGN(128) hash[64];
	uint32_t _ALIGN(128) endiandata[20];
	uint32_t _ALIGN(128) wholeMatrix[384];
	uint32_t _ALIGN(128) wholeMatrix2[8];

	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;

	const uint32_t Htarg = ptarget[7];
	const uint32_t first_nonce = pdata[19];
	uint32_t nonce = first_nonce;
	int32_t flag = -1;

	if (opt_benchmark)
		ptarget[7] = 0x0000ff;

	for (int i=0; i < 19; i++) {
		be32enc(&endiandata[i], pdata[i]);
	}

	do {
		be32enc(&endiandata[19], nonce);
#ifndef SSE
		lyra2rev3_hash_AVX(hash, endiandata, wholeMatrix, &flag, wholeMatrix2);
#else
		lyra2rev3_hash_SSE(hash, endiandata, wholeMatrix, &flag, wholeMatrix2);
#endif
		for (int i = 0; i < 8; i++) {
			if (hash[7 + i * 8] <= Htarg && fulltest(hash + i * 8, ptarget)) {
				work_set_target_ratio(work, hash + i * 8);
				pdata[19] = nonce;
				*hashes_done = pdata[19] - first_nonce;
				return 1;
			}
			nonce ++;
		}

	} while (nonce < max_nonce && !work_restart[thr_id].restart);

	pdata[19] = nonce;
	*hashes_done = pdata[19] - first_nonce + 1;
	return 0;
}
