/**************************************************************************************************
*                                                                                                 *
* This file is part of BLASFEO.                                                                   *
*                                                                                                 *
* BLASFEO -- BLAS for embedded optimization.                                                      *
* Copyright (C) 2019 by Gianluca Frison.                                                          *
* Developed at IMTEK (University of Freiburg) under the supervision of Moritz Diehl.              *
* All rights reserved.                                                                            *
*                                                                                                 *
* The 2-Clause BSD License                                                                        *
*                                                                                                 *
* Redistribution and use in source and binary forms, with or without                              *
* modification, are permitted provided that the following conditions are met:                     *
*                                                                                                 *
* 1. Redistributions of source code must retain the above copyright notice, this                  *
*    list of conditions and the following disclaimer.                                             *
* 2. Redistributions in binary form must reproduce the above copyright notice,                    *
*    this list of conditions and the following disclaimer in the documentation                    *
*    and/or other materials provided with the distribution.                                       *
*                                                                                                 *
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND                 *
* ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED                   *
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE                          *
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR                 *
* ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES                  *
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;                    *
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND                     *
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT                      *
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS                   *
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                                    *
*                                                                                                 *
* Author: Gianluca Frison, gianluca.frison (at) imtek.uni-freiburg.de                             *
*                                                                                                 *
**************************************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

#include <blasfeo_common.h>
#include <blasfeo_s_aux.h>
#include <blasfeo_s_kernel.h>
#include <blasfeo_s_blasfeo_api.h>
#if defined(BLASFEO_REF_API)
#include <blasfeo_s_blasfeo_ref_api.h>
#endif



// dpotrf
void blasfeo_hp_spotrf_l(int m, struct blasfeo_smat *sC, int ci, int cj, struct blasfeo_smat *sD, int di, int dj)
	{

	if(m<=0)
		return;

	if(ci!=0 | di!=0)
		{
#if defined(BLASFEO_REF_API)
		blasfeo_ref_spotrf_l(m, sC, ci, cj, sD, di, dj);
		return;
#else
#ifdef EXT_DEP
		printf("\nblasfeo_spotrf_l: feature not implemented yet: ci=%d, di=%d\n", ci, di);
#endif	
		exit(1);
#endif
		}

	const int ps = 4;

	float d_1 = 1.0;

	int sdc = sC->cn;
	int sdd = sD->cn;
	float *pC = sC->pA + cj*ps;
	float *pD = sD->pA + dj*ps;
	float *dD = sD->dA;
	if(di==0 && dj==0) // XXX what to do if di and dj are not zero
		sD->use_dA = 1;
	else
		sD->use_dA = 0;

	int i, j, l;

	i = 0;
#if defined(TARGET_ARMV8A_ARM_CORTEX_A57) || defined(TARGET_ARMV8A_ARM_CORTEX_A53)
#if 1
	for(; i<m-15; i+=16)
		{
		j = 0;
		for(; j<i; j+=4)
			{
			kernel_strsm_nt_rl_inv_16x4_lib4(j, &pD[i*sdd], sdd, &pD[j*sdd], &d_1, &pC[j*ps+i*sdc], sdc, &pD[j*ps+i*sdd], sdd, &pD[j*ps+j*sdd], &dD[j]);
			}
		kernel_spotrf_nt_l_16x4_lib4(j, &pD[i*sdd], sdd, &pD[j*sdd], &pC[j*ps+j*sdc], sdc, &pD[j*ps+j*sdd], sdd, &dD[j]);
		kernel_spotrf_nt_l_12x4_lib4(j+4, &pD[(i+4)*sdd], sdd, &pD[(j+4)*sdd], &pC[(j+4)*ps+(j+4)*sdc], sdc, &pD[(j+4)*ps+(j+4)*sdd], sdd, &dD[j+4]);
		kernel_spotrf_nt_l_8x4_lib4(j+8, &pD[(i+8)*sdd], sdd, &pD[(j+8)*sdd], &pC[(j+8)*ps+(j+8)*sdc], sdc, &pD[(j+8)*ps+(j+8)*sdd], sdd, &dD[j+8]);
		kernel_spotrf_nt_l_4x4_lib4(j+12, &pD[(i+12)*sdd], &pD[(j+12)*sdd], &pC[(j+12)*ps+(j+12)*sdc], &pD[(j+12)*ps+(j+12)*sdd], &dD[j+12]);
		}
	if(m>i)
		{
		if(m-i<=4)
			{
			goto left_4;
			}
		else if(m-i<=8)
			{
			goto left_8;
			}
		else if(m-i<=12)
			{
			goto left_12;
			}
		else
			{
			goto left_16;
			}
		}
#elif 0
	for(; i<m-11; i+=12)
		{
		j = 0;
		for(; j<i; j+=4)
			{
			kernel_strsm_nt_rl_inv_12x4_lib4(j, &pD[i*sdd], sdd, &pD[j*sdd], &pC[j*ps+i*sdc], sdc, &pD[j*ps+i*sdd], sdd, &pD[j*ps+j*sdd], &dD[j]);
			}
		kernel_spotrf_nt_l_12x4_lib4(j, &pD[i*sdd], sdd, &pD[j*sdd], &pC[j*ps+j*sdc], sdc, &pD[j*ps+j*sdd], sdd, &dD[j]);
		kernel_spotrf_nt_l_8x4_lib4(j+4, &pD[(i+4)*sdd], sdd, &pD[(j+4)*sdd], &pC[(j+4)*ps+(j+4)*sdc], sdc, &pD[(j+4)*ps+(j+4)*sdd], sdd, &dD[j+4]);
		kernel_spotrf_nt_l_4x4_lib4(j+8, &pD[(i+8)*sdd], &pD[(j+8)*sdd], &pC[(j+8)*ps+(j+8)*sdc], &pD[(j+8)*ps+(j+8)*sdd], &dD[j+8]);
		}
	if(m>i)
		{
		if(m-i<=4)
			{
			goto left_4;
			}
		else if(m-i<=8)
			{
			goto left_8;
			}
		else
			{
			goto left_12;
			}
		}
#else
	for(; i<m-7; i+=8)
		{
		j = 0;
		for(; j<i; j+=4)
			{
			kernel_strsm_nt_rl_inv_8x4_lib4(j, &pD[i*sdd], sdd, &pD[j*sdd], &d_1, &pC[j*ps+i*sdc], sdc, &pD[j*ps+i*sdd], sdd, &pD[j*ps+j*sdd], &dD[j]);
			}
		kernel_spotrf_nt_l_8x4_lib4(j, &pD[i*sdd], sdd, &pD[j*sdd], &pC[j*ps+j*sdc], sdc, &pD[j*ps+j*sdd], sdd, &dD[j]);
		kernel_spotrf_nt_l_4x4_lib4(j+4, &pD[(i+4)*sdd], &pD[(j+4)*sdd], &pC[(j+4)*ps+(j+4)*sdc], &pD[(j+4)*ps+(j+4)*sdd], &dD[j+4]);
		}
	if(m>i)
		{
		if(m-i<=4)
			{
			goto left_4;
			}
		else
			{
			goto left_8;
			}
		}
#endif
#elif defined(TARGET_ARMV7A_ARM_CORTEX_A15)
	for(; i<m-11; i+=12)
		{
		j = 0;
		for(; j<i; j+=4)
			{
			kernel_strsm_nt_rl_inv_12x4_lib4(j, &pD[i*sdd], sdd, &pD[j*sdd], &d_1, &pC[j*ps+i*sdc], sdc, &pD[j*ps+i*sdd], sdd, &pD[j*ps+j*sdd], &dD[j]);
			}
		kernel_spotrf_nt_l_12x4_lib4(j, &pD[i*sdd], sdd, &pD[j*sdd], &pC[j*ps+j*sdc], sdc, &pD[j*ps+j*sdd], sdd, &dD[j]);
		kernel_spotrf_nt_l_8x4_lib4(j+4, &pD[(i+4)*sdd], sdd, &pD[(j+4)*sdd], &pC[(j+4)*ps+(j+4)*sdc], sdc, &pD[(j+4)*ps+(j+4)*sdd], sdd, &dD[j+4]);
		kernel_spotrf_nt_l_4x4_lib4(j+8, &pD[(i+8)*sdd], &pD[(j+8)*sdd], &pC[(j+8)*ps+(j+8)*sdc], &pD[(j+8)*ps+(j+8)*sdd], &dD[j+8]);
		}
	if(m>i)
		{
		if(m-i<=4)
			{
			goto left_4;
			}
		else if(m-i<=8)
			{
			goto left_8;
			}
		else
			{
			goto left_12;
			}
		}
#elif defined(TARGET_ARMV7A_ARM_CORTEX_A9) | defined(TARGET_ARMV7A_ARM_CORTEX_A7)
	for(; i<m-7; i+=8)
		{
		j = 0;
		for(; j<i; j+=4)
			{
			kernel_strsm_nt_rl_inv_8x4_lib4(j, &pD[i*sdd], sdd, &pD[j*sdd], &d_1, &pC[j*ps+i*sdc], sdc, &pD[j*ps+i*sdd], sdd, &pD[j*ps+j*sdd], &dD[j]);
			}
		kernel_spotrf_nt_l_8x4_lib4(j, &pD[i*sdd], sdd, &pD[j*sdd], &pC[j*ps+j*sdc], sdc, &pD[j*ps+j*sdd], sdd, &dD[j]);
		kernel_spotrf_nt_l_4x4_lib4(j+4, &pD[(i+4)*sdd], &pD[(j+4)*sdd], &pC[(j+4)*ps+(j+4)*sdc], &pD[(j+4)*ps+(j+4)*sdd], &dD[j+4]);
		}
	if(m>i)
		{
		if(m-i<=4)
			{
			goto left_4;
			}
		else
			{
			goto left_8;
			}
		}
#else
	for(; i<m-3; i+=4)
		{
		j = 0;
		for(; j<i; j+=4)
			{
			kernel_strsm_nt_rl_inv_4x4_lib4(j, &pD[i*sdd], &pD[j*sdd], &d_1, &pC[j*ps+i*sdc], &pD[j*ps+i*sdd], &pD[j*ps+j*sdd], &dD[j]);
			}
		kernel_spotrf_nt_l_4x4_lib4(j, &pD[i*sdd], &pD[j*sdd], &pC[j*ps+j*sdc], &pD[j*ps+j*sdd], &dD[j]);
		}
	if(m>i)
		{
		goto left_4;
		}
#endif

	// common return if i==m
	return;

	// clean up loops definitions

#if defined(TARGET_ARMV8A_ARM_CORTEX_A57) || defined(TARGET_ARMV8A_ARM_CORTEX_A53)
	left_16:
	j = 0;
	for(; j<i; j+=4)
		{
		kernel_strsm_nt_rl_inv_16x4_vs_lib4(j, &pD[i*sdd], sdd, &pD[j*sdd], &d_1, &pC[j*ps+i*sdc], sdc, &pD[j*ps+i*sdd], sdd, &pD[j*ps+j*sdd], &dD[j], m-i, m-j);
		}
	kernel_spotrf_nt_l_16x4_vs_lib4(j, &pD[i*sdd], sdd, &pD[j*sdd], &pC[j*ps+j*sdc], sdc, &pD[j*ps+j*sdd], sdd, &dD[j], m-i, m-j);
	kernel_spotrf_nt_l_12x4_vs_lib4(j+4, &pD[(i+4)*sdd], sdd, &pD[(j+4)*sdd], &pC[(j+4)*ps+(j+4)*sdc], sdc, &pD[(j+4)*ps+(j+4)*sdd], sdd, &dD[j+4], m-i-4, m-j-4);
	kernel_spotrf_nt_l_8x4_vs_lib4(j+8, &pD[(i+8)*sdd], sdd, &pD[(j+8)*sdd], &pC[(j+8)*ps+(j+8)*sdc], sdc, &pD[(j+8)*ps+(j+8)*sdd], sdd, &dD[j+8], m-i-8, m-j-8);
	kernel_spotrf_nt_l_4x4_vs_lib4(j+12, &pD[(i+12)*sdd], &pD[(j+12)*sdd], &pC[(j+12)*ps+(i+12)*sdc], &pD[(j+12)*ps+(i+12)*sdd], &dD[j+12], m-i-12, m-j-12);
	return;
#endif

#if defined(TARGET_ARMV8A_ARM_CORTEX_A57) || defined(TARGET_ARMV8A_ARM_CORTEX_A53)
	left_12:
	if(m-i==12)
		{
		j = 0;
		for(; j<i; j+=4)
			{
			kernel_strsm_nt_rl_inv_12x4_lib4(j, &pD[i*sdd], sdd, &pD[j*sdd], &d_1, &pC[j*ps+i*sdc], sdc, &pD[j*ps+i*sdd], sdd, &pD[j*ps+j*sdd], &dD[j]);
			}
		kernel_spotrf_nt_l_12x4_lib4(j, &pD[i*sdd], sdd, &pD[j*sdd], &pC[j*ps+j*sdc], sdc, &pD[j*ps+j*sdd], sdd, &dD[j]);
		kernel_spotrf_nt_l_8x4_lib4(j+4, &pD[(i+4)*sdd], sdd, &pD[(j+4)*sdd], &pC[(j+4)*ps+(j+4)*sdc], sdc, &pD[(j+4)*ps+(j+4)*sdd], sdd, &dD[j+4]);
		kernel_spotrf_nt_l_4x4_lib4(j+8, &pD[(i+8)*sdd], &pD[(j+8)*sdd], &pC[(j+8)*ps+(i+8)*sdc], &pD[(j+8)*ps+(i+8)*sdd], &dD[j+8]);
		}
	else
		{
		j = 0;
		for(; j<i; j+=4)
			{
			kernel_strsm_nt_rl_inv_12x4_vs_lib4(j, &pD[i*sdd], sdd, &pD[j*sdd], &d_1, &pC[j*ps+i*sdc], sdc, &pD[j*ps+i*sdd], sdd, &pD[j*ps+j*sdd], &dD[j], m-i, m-j);
			}
		kernel_spotrf_nt_l_12x4_vs_lib4(j, &pD[i*sdd], sdd, &pD[j*sdd], &pC[j*ps+j*sdc], sdc, &pD[j*ps+j*sdd], sdd, &dD[j], m-i, m-j);
		kernel_spotrf_nt_l_8x4_vs_lib4(j+4, &pD[(i+4)*sdd], sdd, &pD[(j+4)*sdd], &pC[(j+4)*ps+(j+4)*sdc], sdc, &pD[(j+4)*ps+(j+4)*sdd], sdd, &dD[j+4], m-i-4, m-j-4);
		kernel_spotrf_nt_l_4x4_vs_lib4(j+8, &pD[(i+8)*sdd], &pD[(j+8)*sdd], &pC[(j+8)*ps+(i+8)*sdc], &pD[(j+8)*ps+(i+8)*sdd], &dD[j+8], m-i-8, m-j-8);
		}
	return;
#endif
#if defined(TARGET_ARMV7A_ARM_CORTEX_A15)
	left_12:
	if(m-i==12)
		{
		j = 0;
		for(; j<i; j+=4)
			{
			kernel_strsm_nt_rl_inv_12x4_lib4(j, &pD[i*sdd], sdd, &pD[j*sdd], &d_1, &pC[j*ps+i*sdc], sdc, &pD[j*ps+i*sdd], sdd, &pD[j*ps+j*sdd], &dD[j]);
			}
		kernel_spotrf_nt_l_12x4_lib4(j, &pD[i*sdd], sdd, &pD[j*sdd], &pC[j*ps+j*sdc], sdc, &pD[j*ps+j*sdd], sdd, &dD[j]);
		kernel_spotrf_nt_l_8x4_lib4(j+4, &pD[(i+4)*sdd], sdd, &pD[(j+4)*sdd], &pC[(j+4)*ps+(j+4)*sdc], sdc, &pD[(j+4)*ps+(j+4)*sdd], sdd, &dD[j+4]);
		kernel_spotrf_nt_l_4x4_lib4(j+8, &pD[(i+8)*sdd], &pD[(j+8)*sdd], &pC[(j+8)*ps+(i+8)*sdc], &pD[(j+8)*ps+(i+8)*sdd], &dD[j+8]);
		}
	else
		{
		j = 0;
		for(; j<i; j+=4)
			{
	//		kernel_dtrsm_nt_rl_inv_12x4_vs_lib4(j, &pD[i*sdd], sdd, &pD[j*sdd], &pC[j*ps+i*sdc], sdc, &pD[j*ps+i*sdd], sdd, &pD[j*ps+j*sdd], &dD[j], m-i, m-j);
			kernel_strsm_nt_rl_inv_4x4_vs_lib4(j, &pD[i*sdd], &pD[j*sdd], &d_1, &pC[j*ps+i*sdc], &pD[j*ps+i*sdd], &pD[j*ps+j*sdd], &dD[j], m-i, m-j);
			kernel_strsm_nt_rl_inv_4x4_vs_lib4(j, &pD[(i+4)*sdd], &pD[j*sdd], &d_1, &pC[j*ps+(i+4)*sdc], &pD[j*ps+(i+4)*sdd], &pD[j*ps+j*sdd], &dD[j], m-i-4, m-j);
			kernel_strsm_nt_rl_inv_4x4_vs_lib4(j, &pD[(i+8)*sdd], &pD[j*sdd], &d_1, &pC[j*ps+(i+8)*sdc], &pD[j*ps+(i+8)*sdd], &pD[j*ps+j*sdd], &dD[j], m-i-8, m-j);
			}
	//	kernel_spotrf_nt_l_12x4_vs_lib4(j, &pD[i*sdd], sdd, &pD[j*sdd], &pC[j*ps+j*sdc], sdc, &pD[j*ps+j*sdd], sdd, &dD[j], m-i, m-j);
		kernel_spotrf_nt_l_4x4_vs_lib4(j, &pD[i*sdd], &pD[j*sdd], &pC[j*ps+j*sdc], &pD[j*ps+j*sdd], &dD[j], m-i, m-j);
		kernel_strsm_nt_rl_inv_4x4_vs_lib4(j, &pD[(i+4)*sdd], &pD[j*sdd], &d_1, &pC[j*ps+(i+4)*sdc], &pD[j*ps+(i+4)*sdd], &pD[j*ps+j*sdd], &dD[j], m-i-4, m-j);
		kernel_strsm_nt_rl_inv_4x4_vs_lib4(j, &pD[(i+8)*sdd], &pD[j*sdd], &d_1, &pC[j*ps+(i+8)*sdc], &pD[j*ps+(i+8)*sdd], &pD[j*ps+j*sdd], &dD[j], m-i-8, m-j);
//		kernel_spotrf_nt_l_8x4_vs_lib4(j+4, &pD[(i+4)*sdd], sdd, &pD[(j+4)*sdd], &pC[(j+4)*ps+(j+4)*sdc], sdc, &pD[(j+4)*ps+(j+4)*sdd], sdd, &dD[j+4], m-i-4, m-j-4);
		kernel_spotrf_nt_l_4x4_vs_lib4(j+4, &pD[(i+4)*sdd], &pD[(j+4)*sdd], &pC[(j+4)*ps+(j+4)*sdc], &pD[(j+4)*ps+(j+4)*sdd], &dD[j+4], m-i-4, m-j-4);
		kernel_strsm_nt_rl_inv_4x4_vs_lib4(j+4, &pD[(i+8)*sdd], &pD[(j+4)*sdd], &d_1, &pC[(j+4)*ps+(i+8)*sdc], &pD[(j+4)*ps+(i+8)*sdd], &pD[(j+4)*ps+(j+4)*sdd], &dD[j+4], m-i-8, m-j-4);
		kernel_spotrf_nt_l_4x4_vs_lib4(j+8, &pD[(i+8)*sdd], &pD[(j+8)*sdd], &pC[(j+8)*ps+(i+8)*sdc], &pD[(j+8)*ps+(i+8)*sdd], &dD[j+8], m-i-8, m-j-8);
		}
	return;
#endif

#if defined(TARGET_ARMV8A_ARM_CORTEX_A57) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
	left_8:
	if(m-i==8)
		{
		j = 0;
		for(; j<i; j+=4)
			{
			kernel_strsm_nt_rl_inv_8x4_lib4(j, &pD[i*sdd], sdd, &pD[j*sdd], &d_1, &pC[j*ps+i*sdc], sdc, &pD[j*ps+i*sdd], sdd, &pD[j*ps+j*sdd], &dD[j]);
			}
		kernel_spotrf_nt_l_8x4_lib4(j, &pD[i*sdd], sdd, &pD[j*sdd], &pC[j*ps+j*sdc], sdc, &pD[j*ps+j*sdd], sdd, &dD[j]);
		kernel_spotrf_nt_l_4x4_lib4(j+4, &pD[(i+4)*sdd], &pD[(j+4)*sdd], &pC[(j+4)*ps+(i+4)*sdc], &pD[(j+4)*ps+(i+4)*sdd], &dD[j+4]);
		}
	else
		{
		j = 0;
		for(; j<i; j+=4)
			{
			kernel_strsm_nt_rl_inv_8x4_vs_lib4(j, &pD[i*sdd], sdd, &pD[j*sdd], &d_1, &pC[j*ps+i*sdc], sdc, &pD[j*ps+i*sdd], sdd, &pD[j*ps+j*sdd], &dD[j], m-i, m-j);
			}
		kernel_spotrf_nt_l_8x4_vs_lib4(j, &pD[i*sdd], sdd, &pD[j*sdd], &pC[j*ps+j*sdc], sdc, &pD[j*ps+j*sdd], sdd, &dD[j], m-i, m-j);
		kernel_spotrf_nt_l_4x4_vs_lib4(j+4, &pD[(i+4)*sdd], &pD[(j+4)*sdd], &pC[(j+4)*ps+(i+4)*sdc], &pD[(j+4)*ps+(i+4)*sdd], &dD[j+4], m-i-4, m-j-4);
		}
	return;
#endif
#if defined(TARGET_ARMV7A_ARM_CORTEX_A15) | defined(TARGET_ARMV7A_ARM_CORTEX_A9) | defined(TARGET_ARMV7A_ARM_CORTEX_A7)
	left_8:
	if(m-i==8)
		{
		j = 0;
		for(; j<i; j+=4)
			{
			kernel_strsm_nt_rl_inv_8x4_lib4(j, &pD[i*sdd], sdd, &pD[j*sdd], &d_1, &pC[j*ps+i*sdc], sdc, &pD[j*ps+i*sdd], sdd, &pD[j*ps+j*sdd], &dD[j]);
			}
		kernel_spotrf_nt_l_8x4_lib4(j, &pD[i*sdd], sdd, &pD[j*sdd], &pC[j*ps+j*sdc], sdc, &pD[j*ps+j*sdd], sdd, &dD[j]);
		kernel_spotrf_nt_l_4x4_lib4(j+4, &pD[(i+4)*sdd], &pD[(j+4)*sdd], &pC[(j+4)*ps+(i+4)*sdc], &pD[(j+4)*ps+(i+4)*sdd], &dD[j+4]);
		}
	else
		{
		j = 0;
		for(; j<i; j+=4)
			{
	//		kernel_dtrsm_nt_rl_inv_8x4_vs_lib4(j, &pD[i*sdd], sdd, &pD[j*sdd], &pC[j*ps+i*sdc], sdc, &pD[j*ps+i*sdd], sdd, &pD[j*ps+j*sdd], &dD[j], m-i, m-j);
			kernel_strsm_nt_rl_inv_4x4_vs_lib4(j, &pD[i*sdd], &pD[j*sdd], &d_1, &pC[j*ps+i*sdc], &pD[j*ps+i*sdd], &pD[j*ps+j*sdd], &dD[j], m-i, m-j);
			kernel_strsm_nt_rl_inv_4x4_vs_lib4(j, &pD[(i+4)*sdd], &pD[j*sdd], &d_1, &pC[j*ps+(i+4)*sdc], &pD[j*ps+(i+4)*sdd], &pD[j*ps+j*sdd], &dD[j], m-i-4, m-j);
			}
	//	kernel_spotrf_nt_l_8x4_vs_lib4(j, &pD[i*sdd], sdd, &pD[j*sdd], &pC[j*ps+j*sdc], sdc, &pD[j*ps+j*sdd], sdd, &dD[j], m-i, m-j);
		kernel_spotrf_nt_l_4x4_vs_lib4(j, &pD[i*sdd], &pD[j*sdd], &pC[j*ps+j*sdc], &pD[j*ps+j*sdd], &dD[j], m-i, m-j);
		kernel_strsm_nt_rl_inv_4x4_vs_lib4(j, &pD[(i+4)*sdd], &pD[j*sdd], &d_1, &pC[j*ps+(i+4)*sdc], &pD[j*ps+(i+4)*sdd], &pD[j*ps+j*sdd], &dD[j], m-i-4, m-j);
		kernel_spotrf_nt_l_4x4_vs_lib4(j+4, &pD[(i+4)*sdd], &pD[(j+4)*sdd], &pC[(j+4)*ps+(i+4)*sdc], &pD[(j+4)*ps+(i+4)*sdd], &dD[j+4], m-i-4, m-j-4);
		}
	return;
#endif

	left_4:
	if(m-i==4)
		{
		j = 0;
		for(; j<i; j+=4)
			{
			kernel_strsm_nt_rl_inv_4x4_lib4(j, &pD[i*sdd], &pD[j*sdd], &d_1, &pC[j*ps+i*sdc], &pD[j*ps+i*sdd], &pD[j*ps+j*sdd], &dD[j]);
			}
		kernel_spotrf_nt_l_4x4_lib4(j, &pD[i*sdd], &pD[j*sdd], &pC[j*ps+j*sdc], &pD[j*ps+j*sdd], &dD[j]);
		}
	else
		{
		j = 0;
		for(; j<i; j+=4)
			{
			kernel_strsm_nt_rl_inv_4x4_vs_lib4(j, &pD[i*sdd], &pD[j*sdd], &d_1, &pC[j*ps+i*sdc], &pD[j*ps+i*sdd], &pD[j*ps+j*sdd], &dD[j], m-i, m-j);
			}
		kernel_spotrf_nt_l_4x4_vs_lib4(j, &pD[i*sdd], &pD[j*sdd], &pC[j*ps+j*sdc], &pD[j*ps+j*sdd], &dD[j], m-i, m-j);
		}
	return;

	}



// dpotrf
void blasfeo_hp_spotrf_l_mn(int m, int n, struct blasfeo_smat *sC, int ci, int cj, struct blasfeo_smat *sD, int di, int dj)
	{

	if(m<=0 || n<=0)
		return;

	if(ci!=0 | di!=0)
		{
#if defined(BLASFEO_REF_API)
		blasfeo_ref_spotrf_l_mn(m, n, sC, ci, cj, sD, di, dj);
		return;
#else
#ifdef EXT_DEP
		printf("\nblasfeo_spotrf_l: feature not implemented yet: ci=%d, di=%d\n", ci, di);
#endif	
		exit(1);
#endif
		}

	const int ps = 4;

	float d_1 = 1.0;

	int sdc = sC->cn;
	int sdd = sD->cn;
	float *pC = sC->pA + cj*ps;
	float *pD = sD->pA + dj*ps;
	float *dD = sD->dA;
	if(di==0 && dj==0) // XXX what to do if di and dj are not zero
		sD->use_dA = 1;
	else
		sD->use_dA = 0;

	int i, j, l;

	i = 0;
#if defined(TARGET_ARMV8A_ARM_CORTEX_A57) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
	for(; i<m-15; i+=16)
		{
		j = 0;
		for(; j<i & j<n-3; j+=4)
			{
			kernel_strsm_nt_rl_inv_16x4_lib4(j, &pD[i*sdd], sdd, &pD[j*sdd], &d_1, &pC[j*ps+i*sdc], sdc, &pD[j*ps+i*sdd], sdd, &pD[j*ps+j*sdd], &dD[j]);
			}
		if(j<n)
			{
			if(j<i) // dtrsm
				{
				kernel_strsm_nt_rl_inv_16x4_vs_lib4(j, &pD[i*sdd], sdd, &pD[j*sdd], &d_1, &pC[j*ps+i*sdc], sdc, &pD[j*ps+i*sdd], sdd, &pD[j*ps+j*sdd], &dD[j], m-i, n-j);
				}
			else // dpotrf
				{
				if(j<n-15)
					{
					kernel_spotrf_nt_l_16x4_lib4(j, &pD[i*sdd], sdd, &pD[j*sdd], &pC[j*ps+j*sdc], sdc, &pD[j*ps+j*sdd], sdd, &dD[j]);
					kernel_spotrf_nt_l_12x4_lib4(j+4, &pD[(i+4)*sdd], sdd, &pD[(j+4)*sdd], &pC[(j+4)*ps+(j+4)*sdc], sdc, &pD[(j+4)*ps+(j+4)*sdd], sdd, &dD[j+4]);
//					kernel_spotrf_nt_l_8x8_lib4(j+8, &pD[(i+8)*sdd], sdd, &pD[(j+8)*sdd], sdd, &pC[(j+8)*ps+(i+8)*sdc], sdc, &pD[(j+8)*ps+(i+8)*sdd], sdd, &dD[j+8]);
					kernel_spotrf_nt_l_8x4_lib4(j+8, &pD[(i+8)*sdd], sdd, &pD[(j+8)*sdd], &pC[(j+8)*ps+(i+8)*sdc], sdc, &pD[(j+8)*ps+(i+8)*sdd], sdd, &dD[j+8]);
					kernel_spotrf_nt_l_4x4_lib4(j+12, &pD[(i+12)*sdd], &pD[(j+12)*sdd], &pC[(j+12)*ps+(i+12)*sdc], &pD[(j+12)*ps+(i+12)*sdd], &dD[j+12]);
					}
				else
					{
					kernel_spotrf_nt_l_16x4_vs_lib4(j, &pD[i*sdd], sdd, &pD[j*sdd], &pC[j*ps+j*sdc], sdc, &pD[j*ps+j*sdd], sdd, &dD[j], m-i, n-j);
					if(j<n-4)
						{
						kernel_spotrf_nt_l_12x4_vs_lib4(j+4, &pD[(i+4)*sdd], sdd, &pD[(j+8)*sdd], &pC[(j+8)*ps+(j+8)*sdc], sdc, &pD[(j+8)*ps+(j+8)*sdd], sdd, &dD[j+4], m-i-4, n-j-4);
						if(j<n-8)
							{
							kernel_spotrf_nt_l_8x4_vs_lib4(j+8, &pD[(i+8)*sdd], sdd, &pD[(j+8)*sdd], &pC[(j+8)*ps+(i+8)*sdc], sdc, &pD[(j+8)*ps+(i+8)*sdd], sdd, &dD[j+8], m-i-8, n-j-8);
							if(j<n-12)
								{
								kernel_spotrf_nt_l_4x4_vs_lib4(j+12, &pD[(i+12)*sdd], &pD[(j+12)*sdd], &pC[(j+12)*ps+(i+12)*sdc], &pD[(j+12)*ps+(i+12)*sdd], &dD[j+12], m-i-12, n-j-12);
								}
							}
						}
					}
				}
			}
		}
	if(m>i)
		{
		if(m-i<=4)
			{
			goto left_4;
			}
		else if(m-i<=8)
			{
			goto left_8;
			}
		else if(m-i<=12)
			{
			goto left_12;
			}
		else
			{
			goto left_16;
			}
		}
#elif 0 //defined(TARGET_ARMV8A_ARM_CORTEX_A57) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
	for(; i<m-11; i+=12)
		{
		j = 0;
		for(; j<i & j<n-3; j+=4)
			{
			kernel_strsm_nt_rl_inv_12x4_lib4(j, &pD[i*sdd], sdd, &pD[j*sdd], &d_1, &pC[j*ps+i*sdc], sdc, &pD[j*ps+i*sdd], sdd, &pD[j*ps+j*sdd], &dD[j]);
			}
		if(j<n)
			{
			if(j<i) // dtrsm
				{
				kernel_strsm_nt_rl_inv_12x4_vs_lib4(j, &pD[i*sdd], sdd, &pD[j*sdd], &d_1, &pC[j*ps+i*sdc], sdc, &pD[j*ps+i*sdd], sdd, &pD[j*ps+j*sdd], &dD[j], m-i, n-j);
				}
			else // dpotrf
				{
				if(j<n-11)
					{
					kernel_spotrf_nt_l_12x4_lib4(j, &pD[i*sdd], sdd, &pD[j*sdd], &pC[j*ps+j*sdc], sdc, &pD[j*ps+j*sdd], sdd, &dD[j]);
//					kernel_spotrf_nt_l_8x8_lib4(j+4, &pD[(i+4)*sdd], sdd, &pD[(j+4)*sdd], sdd, &pC[(j+4)*ps+(i+4)*sdc], sdc, &pD[(j+4)*ps+(i+4)*sdd], sdd, &dD[j+4]);
					kernel_spotrf_nt_l_8x4_lib4(j+4, &pD[(i+4)*sdd], sdd, &pD[(j+4)*sdd], &pC[(j+4)*ps+(i+4)*sdc], sdc, &pD[(j+4)*ps+(i+4)*sdd], sdd, &dD[j+4]);
					kernel_spotrf_nt_l_4x4_lib4(j+8, &pD[(i+8)*sdd], &pD[(j+8)*sdd], &pC[(j+8)*ps+(i+8)*sdc], &pD[(j+8)*ps+(i+8)*sdd], &dD[j+8]);
					}
				else
					{
					kernel_spotrf_nt_l_12x4_vs_lib4(j, &pD[i*sdd], sdd, &pD[j*sdd], &pC[j*ps+j*sdc], sdc, &pD[j*ps+j*sdd], sdd, &dD[j], m-i, n-j);
					if(j<n-4)
						{
						kernel_spotrf_nt_l_8x4_vs_lib4(j+4, &pD[(i+4)*sdd], sdd, &pD[(j+4)*sdd], &pC[(j+4)*ps+(i+4)*sdc], sdc, &pD[(j+4)*ps+(i+4)*sdd], sdd, &dD[j+4], m-i-4, n-j-4);
						if(j<n-8)
							{
							kernel_spotrf_nt_l_4x4_vs_lib4(j+8, &pD[(i+8)*sdd], &pD[(j+8)*sdd], &pC[(j+8)*ps+(i+8)*sdc], &pD[(j+8)*ps+(i+8)*sdd], &dD[j+8], m-i-8, n-j-8);
							}
						}
					}
				}
			}
		}
	if(m>i)
		{
		if(m-i<=4)
			{
			goto left_4;
			}
		else if(m-i<=8)
			{
			goto left_8;
			}
		else
			{
			goto left_12;
			}
		}
#elif 0 //defined(TARGET_ARMV8A_ARM_CORTEX_A57) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
	for(; i<m-7; i+=8)
		{
		j = 0;
		for(; j<i & j<n-3; j+=4)
			{
			kernel_strsm_nt_rl_inv_8x4_lib4(j, &pD[i*sdd], sdd, &pD[j*sdd], &d_1, &pC[j*ps+i*sdc], sdc, &pD[j*ps+i*sdd], sdd, &pD[j*ps+j*sdd], &dD[j]);
			}
		if(j<n)
			{
			if(j<i) // dtrsm
				{
				kernel_strsm_nt_rl_inv_8x4_vs_lib4(j, &pD[i*sdd], sdd, &pD[j*sdd], &d_1, &pC[j*ps+i*sdc], sdc, &pD[j*ps+i*sdd], sdd, &pD[j*ps+j*sdd], &dD[j], m-i, n-j);
				}
			else // dpotrf
				{
				if(j<n-7)
//				if(0)
					{
					kernel_spotrf_nt_l_8x4_lib4(j, &pD[i*sdd], sdd, &pD[j*sdd], &pC[j*ps+j*sdc], sdc, &pD[j*ps+j*sdd], sdd, &dD[j]);
					kernel_spotrf_nt_l_4x4_lib4(j+4, &pD[(i+4)*sdd], &pD[(j+4)*sdd], &pC[(j+4)*ps+(i+4)*sdc], &pD[(j+4)*ps+(i+4)*sdd], &dD[j+4]);
					}
				else
					{
					kernel_spotrf_nt_l_8x4_vs_lib4(j, &pD[i*sdd], sdd, &pD[j*sdd], &pC[j*ps+j*sdc], sdc, &pD[j*ps+j*sdd], sdd, &dD[j], m-i, n-j);
					if(j<n-4)
						{
						kernel_spotrf_nt_l_4x4_vs_lib4(j+4, &pD[(i+4)*sdd], &pD[(j+4)*sdd], &pC[(j+4)*ps+(i+4)*sdc], &pD[(j+4)*ps+(i+4)*sdd], &dD[j+4], m-i-4, n-j-4);
						}
					}
				}
			}
		}
	if(m>i)
		{
		if(m-i<=4)
			{
			goto left_4;
			}
		else
			{
			goto left_8;
			}
		}
#else
	for(; i<m-3; i+=4)
		{
		j = 0;
		for(; j<i && j<n-3; j+=4)
			{
			kernel_strsm_nt_rl_inv_4x4_lib4(j, &pD[i*sdd], &pD[j*sdd], &d_1, &pC[j*ps+i*sdc], &pD[j*ps+i*sdd], &pD[j*ps+j*sdd], &dD[j]);
			}
		if(j<n)
			{
			if(j<i) // dtrsm
				{
				kernel_strsm_nt_rl_inv_4x4_vs_lib4(j, &pD[i*sdd], &pD[j*sdd], &d_1, &pC[j*ps+i*sdc], &pD[j*ps+i*sdd], &pD[j*ps+j*sdd], &dD[j], m-i, n-j);
				}
			else // dpotrf
				{
				if(j<n-3)
					{
					kernel_spotrf_nt_l_4x4_lib4(j, &pD[i*sdd], &pD[j*sdd], &pC[j*ps+j*sdc], &pD[j*ps+j*sdd], &dD[j]);
					}
				else
					{
					kernel_spotrf_nt_l_4x4_vs_lib4(j, &pD[i*sdd], &pD[j*sdd], &pC[j*ps+j*sdc], &pD[j*ps+j*sdd], &dD[j], m-i, n-j);
					}
				}
			}
		}
	if(m>i)
		{
		goto left_4;
		}
#endif

	// common return if i==m
	return;

	// clean up loops definitions

#if defined(TARGET_ARMV8A_ARM_CORTEX_A57) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
	left_16:
	j = 0;
	for(; j<i & j<n; j+=4)
		{
		kernel_strsm_nt_rl_inv_16x4_vs_lib4(j, &pD[i*sdd], sdd, &pD[j*sdd], &d_1, &pC[j*ps+i*sdc], sdc, &pD[j*ps+i*sdd], sdd, &pD[j*ps+j*sdd], &dD[j], m-i, n-j);
		}
	if(j<n)
		{
		kernel_spotrf_nt_l_16x4_vs_lib4(j, &pD[i*sdd], sdd, &pD[j*sdd], &pC[j*ps+j*sdc], sdc, &pD[j*ps+j*sdd], sdd, &dD[j], m-i, n-j);
		if(j<n-4)
			{
			kernel_spotrf_nt_l_12x4_vs_lib4(j+4, &pD[(i+4)*sdd], sdd, &pD[(j+4)*sdd], &pC[(j+4)*ps+(j+4)*sdc], sdc, &pD[(j+4)*ps+(j+4)*sdd], sdd, &dD[j+4], m-i-4, n-j-4);
			if(j<n-8)
				{
				kernel_spotrf_nt_l_8x4_vs_lib4(j+8, &pD[(i+8)*sdd], sdd, &pD[(j+8)*sdd], &pC[(j+8)*ps+(i+8)*sdc], sdc, &pD[(j+8)*ps+(i+8)*sdd], sdd, &dD[j+8], m-i-8, n-j-8);
				if(j<n-12)
					{
					kernel_spotrf_nt_l_4x4_vs_lib4(j+12, &pD[(i+12)*sdd], &pD[(j+12)*sdd], &pC[(j+12)*ps+(i+12)*sdc], &pD[(j+12)*ps+(i+12)*sdd], &dD[j+12], m-i-12, n-j-12);
					}
				}
			}
		}
	return;
#endif

#if defined(TARGET_ARMV8A_ARM_CORTEX_A57) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
	left_12:
	j = 0;
	for(; j<i & j<n; j+=4)
		{
		kernel_strsm_nt_rl_inv_12x4_vs_lib4(j, &pD[i*sdd], sdd, &pD[j*sdd], &d_1, &pC[j*ps+i*sdc], sdc, &pD[j*ps+i*sdd], sdd, &pD[j*ps+j*sdd], &dD[j], m-i, n-j);
		}
	if(j<n)
		{
		kernel_spotrf_nt_l_12x4_vs_lib4(j, &pD[i*sdd], sdd, &pD[j*sdd], &pC[j*ps+j*sdc], sdc, &pD[j*ps+j*sdd], sdd, &dD[j], m-i, n-j);
		if(j<n-4)
			{
			kernel_spotrf_nt_l_8x4_vs_lib4(j+4, &pD[(i+4)*sdd], sdd, &pD[(j+4)*sdd], &pC[(j+4)*ps+(i+4)*sdc], sdc, &pD[(j+4)*ps+(i+4)*sdd], sdd, &dD[j+4], m-i-4, n-j-4);
			if(j<n-8)
				{
				kernel_spotrf_nt_l_4x4_vs_lib4(j+8, &pD[(i+8)*sdd], &pD[(j+8)*sdd], &pC[(j+8)*ps+(i+8)*sdc], &pD[(j+8)*ps+(i+8)*sdd], &dD[j+8], m-i-8, n-j-8);
				}
			}
		}
	return;
#endif

#if defined(TARGET_ARMV8A_ARM_CORTEX_A57) | defined(TARGET_ARMV8A_ARM_CORTEX_A53)
	left_8:
	j = 0;
	for(; j<i & j<n; j+=4)
		{
		kernel_strsm_nt_rl_inv_8x4_vs_lib4(j, &pD[i*sdd], sdd, &pD[j*sdd], &d_1, &pC[j*ps+i*sdc], sdc, &pD[j*ps+i*sdd], sdd, &pD[j*ps+j*sdd], &dD[j], m-i, n-j);
		}
	if(j<n)
		{
		kernel_spotrf_nt_l_8x4_vs_lib4(j, &pD[i*sdd], sdd, &pD[j*sdd], &pC[j*ps+j*sdc], sdc, &pD[j*ps+j*sdd], sdd, &dD[j], m-i, n-j);
		if(j<n-4)
			{
			kernel_spotrf_nt_l_4x4_vs_lib4(j+4, &pD[(i+4)*sdd], &pD[(j+4)*sdd], &pC[(j+4)*ps+(i+4)*sdc], &pD[(j+4)*ps+(i+4)*sdd], &dD[j+4], m-i-4, n-j-4);
			}
		}
	return;
#endif

	left_4:
	j = 0;
	for(; j<i && j<n-3; j+=4)
		{
		kernel_strsm_nt_rl_inv_4x4_vs_lib4(j, &pD[i*sdd], &pD[j*sdd], &d_1, &pC[j*ps+i*sdc], &pD[j*ps+i*sdd], &pD[j*ps+j*sdd], &dD[j], m-i, n-j);
		}
	if(j<n)
		{
		if(j<i) // dtrsm
			{
			kernel_strsm_nt_rl_inv_4x4_vs_lib4(j, &pD[i*sdd], &pD[j*sdd], &d_1, &pC[j*ps+i*sdc], &pD[j*ps+i*sdd], &pD[j*ps+j*sdd], &dD[j], m-i, n-j);
			}
		else // dpotrf
			{
			kernel_spotrf_nt_l_4x4_vs_lib4(j, &pD[i*sdd], &pD[j*sdd], &pC[j*ps+j*sdc], &pD[j*ps+j*sdd], &dD[j], m-i, n-j);
			}
		}
	return;

	}



// spotrf
void blasfeo_hp_spotrf_u(int m, struct blasfeo_smat *sC, int ci, int cj, struct blasfeo_smat *sD, int di, int dj)
	{

	if(m<=0)
		return;

#if defined(BLASFEO_REF_API)
	blasfeo_ref_spotrf_u(m, sC, ci, cj, sD, di, dj);
	return;
#else
#ifdef EXT_DEP
	printf("\nblasfeo_spotrf_u: feature not implemented yet\n");
#endif	
	exit(1);
#endif
	
	}



// dsyrk dpotrf
void blasfeo_hp_ssyrk_spotrf_ln(int m, int k, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_smat *sB, int bi, int bj, struct blasfeo_smat *sC, int ci, int cj, struct blasfeo_smat *sD, int di, int dj)
	{

	if(m<=0)
		return;

	if(ai!=0 | bi!=0 | ci!=0 | di!=0)
		{
#if defined(BLASFEO_REF_API)
		blasfeo_ref_ssyrk_spotrf_ln(m, k, sA, ai, aj, sB, bi, bj, sC, ci, cj, sD, di, dj);
		return;
#else
#ifdef EXT_DEP
		printf("\nblasfeo_ssyrk_spotrf_ln: feature not implemented yet: ai=%d, bi=%d, ci=%d, di=%d\n", ai, bi, ci, di);
#endif	
		exit(1);
#endif
		}

	const int ps = 4;

	int sda = sA->cn;
	int sdb = sB->cn;
	int sdc = sC->cn;
	int sdd = sD->cn;
	float *pA = sA->pA + aj*ps;
	float *pB = sB->pA + bj*ps;
	float *pC = sC->pA + cj*ps;
	float *pD = sD->pA + dj*ps;
	float *dD = sD->dA; // XXX what to do if di and dj are not zero

	if(di==0 && dj==0)
		sD->use_dA = 1;
	else
		sD->use_dA = 0;

	int i, j, l;

	i = 0;
	for(; i<m-3; i+=4)
		{
		j = 0;
		for(; j<i; j+=4)
			{
			kernel_sgemm_strsm_nt_rl_inv_4x4_lib4(k, &pA[i*sda], &pB[j*sdb], j, &pD[i*sdd], &pD[j*sdd], &pC[j*ps+i*sdc], &pD[j*ps+i*sdd], &pD[j*ps+j*sdd], &dD[j]);
			}
		kernel_ssyrk_spotrf_nt_l_4x4_lib4(k, &pA[i*sda], &pB[j*sdb], j, &pD[i*sdd], &pD[j*sdd], &pC[j*ps+j*sdc], &pD[j*ps+j*sdd], &dD[j]);
		}
	if(m>i)
		{
		goto left_4;
		}

	// common return if i==m
	return;

	// clean up loops definitions

	left_4:
	j = 0;
	for(; j<i; j+=4)
		{
		kernel_sgemm_strsm_nt_rl_inv_4x4_vs_lib4(k, &pA[i*sda], &pB[j*sdb], j, &pD[i*sdd], &pD[j*sdd], &pC[j*ps+i*sdc], &pD[j*ps+i*sdd], &pD[j*ps+j*sdd], &dD[j], m-i, m-j);
		}
	kernel_ssyrk_spotrf_nt_l_4x4_vs_lib4(k, &pA[i*sda], &pB[j*sdb], j, &pD[i*sdd], &pD[j*sdd], &pC[j*ps+j*sdc], &pD[j*ps+j*sdd], &dD[j], m-i, m-j);

	return;
	}



void blasfeo_hp_ssyrk_spotrf_ln_mn(int m, int n, int k, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_smat *sB, int bi, int bj, struct blasfeo_smat *sC, int ci, int cj, struct blasfeo_smat *sD, int di, int dj)
	{

	if(m<=0 || n<=0)
		return;

	if(ai!=0 | bi!=0 | ci!=0 | di!=0)
		{
#if defined(BLASFEO_REF_API)
		blasfeo_ref_ssyrk_spotrf_ln_mn(m, n, k, sA, ai, aj, sB, bi, bj, sC, ci, cj, sD, di, dj);
		return;
#else
#ifdef EXT_DEP
		printf("\nblasfeo_ssyrk_spotrf_ln: feature not implemented yet: ai=%d, bi=%d, ci=%d, di=%d\n", ai, bi, ci, di);
#endif	
		exit(1);
#endif
		}

	const int ps = 4;

	int sda = sA->cn;
	int sdb = sB->cn;
	int sdc = sC->cn;
	int sdd = sD->cn;
	float *pA = sA->pA + aj*ps;
	float *pB = sB->pA + bj*ps;
	float *pC = sC->pA + cj*ps;
	float *pD = sD->pA + dj*ps;
	float *dD = sD->dA; // XXX what to do if di and dj are not zero

	if(di==0 && dj==0)
		sD->use_dA = 1;
	else
		sD->use_dA = 0;

	int i, j, l;

	i = 0;

	for(; i<m-3; i+=4)
		{
		j = 0;
		for(; j<i & j<n-3; j+=4)
			{
			kernel_sgemm_strsm_nt_rl_inv_4x4_lib4(k, &pA[i*sda], &pB[j*sdb], j, &pD[i*sdd], &pD[j*sdd], &pC[j*ps+i*sdc], &pD[j*ps+i*sdd], &pD[j*ps+j*sdd], &dD[j]);
			}
		if(j<n)
			{
			if(j<i) // dgemm
				{
				kernel_sgemm_strsm_nt_rl_inv_4x4_vs_lib4(k, &pA[i*sda], &pB[j*sdb], j, &pD[i*sdd], &pD[j*sdd], &pC[j*ps+i*sdc], &pD[j*ps+i*sdd], &pD[j*ps+j*sdd], &dD[j], m-i, n-j);
				}
			else // dsyrk
				{
				if(j<n-3)
					{
					kernel_ssyrk_spotrf_nt_l_4x4_lib4(k, &pA[i*sda], &pB[j*sdb], j, &pD[i*sdd], &pD[j*sdd], &pC[j*ps+j*sdc], &pD[j*ps+j*sdd], &dD[j]);
					}
				else
					{
					kernel_ssyrk_spotrf_nt_l_4x4_vs_lib4(k, &pA[i*sda], &pB[j*sdb], j, &pD[i*sdd], &pD[j*sdd], &pC[j*ps+j*sdc], &pD[j*ps+j*sdd], &dD[j], m-i, n-j);
					}
				}
			}
		}
	if(m>i)
		{
		goto left_4;
		}

	// common return if i==m
	return;

	// clean up loops definitions

	left_4:
	j = 0;
	for(; j<i & j<n; j+=4)
		{
		kernel_sgemm_strsm_nt_rl_inv_4x4_vs_lib4(k, &pA[i*sda], &pB[j*sdb], j, &pD[i*sdd], &pD[j*sdd], &pC[j*ps+i*sdc], &pD[j*ps+i*sdd], &pD[j*ps+j*sdd], &dD[j], m-i, n-j);
		}
	if(j<n)
		{
		kernel_ssyrk_spotrf_nt_l_4x4_vs_lib4(k, &pA[i*sda], &pB[j*sdb], j, &pD[i*sdd], &pD[j*sdd], &pC[j*ps+j*sdc], &pD[j*ps+j*sdd], &dD[j], m-i, n-j);
		}

	return;
	}



// dgetrf no pivoting
void blasfeo_hp_sgetrf_np(int m, int n, struct blasfeo_smat *sC, int ci, int cj, struct blasfeo_smat *sD, int di, int dj)
	{
	if(ci!=0 | di!=0)
		{
#if defined(BLASFEO_REF_API)
		blasfeo_ref_sgetrf_np(m, n, sC, ci, cj, sD, di, dj);
		return;
#else
#ifdef EXT_DEP
		printf("\nblasfeo_sgetf_np: feature not implemented yet: ci=%d, di=%d\n", ci, di);
#endif	
		exit(1);
#endif
		}
	const int ps = 4;
	int sdc = sC->cn;
	int sdd = sD->cn;
	float *pC = sC->pA + cj*ps;
	float *pD = sD->pA + dj*ps;
	float *inv_diag_D = sD->dA; // XXX what to do if di and dj are not zero

	if(m<=0 || n<=0)
		return;
	
	if(di==0 && dj==0)
		sD->use_dA = 1;
	else
		sD->use_dA = 0;

	int ii, jj, ie;

	float d_1 = 1.0;

	// main loop
	ii = 0;
	for( ; ii<m-3; ii+=4)
		{
		jj = 0;
		// solve lower
		ie = n<ii ? n : ii; // ie is multiple of 4
		for( ; jj<ie-3; jj+=4)
			{
			kernel_strsm_nn_ru_inv_4x4_lib4(jj, &pD[ii*sdd], &pD[jj*ps], sdd, &d_1, &pC[jj*ps+ii*sdc], &pD[jj*ps+ii*sdd], &pD[jj*ps+jj*sdd], &inv_diag_D[jj]);
			}
		if(jj<ie)
			{
			kernel_strsm_nn_ru_inv_4x4_vs_lib4(jj, &pD[ii*sdd], &pD[jj*ps], sdd, &d_1, &pC[jj*ps+ii*sdc], &pD[jj*ps+ii*sdd], &pD[jj*ps+jj*sdd], &inv_diag_D[jj], m-ii, ie-jj);
			jj+=4;
			}
		// factorize
		if(jj<n-3)
			{
			kernel_sgetrf_nn_4x4_lib4(jj, &pD[ii*sdd], &pD[jj*ps], sdd, &pC[jj*ps+ii*sdc], &pD[jj*ps+ii*sdd], &inv_diag_D[jj]);
			jj+=4;
			}
		else if(jj<n)
			{
			kernel_sgetrf_nn_4x4_vs_lib4(jj, &pD[ii*sdd], &pD[jj*ps], sdd, &pC[jj*ps+ii*sdc], &pD[jj*ps+ii*sdd], &inv_diag_D[jj], m-ii, n-jj);
			jj+=4;
			}
		// solve upper 
		for( ; jj<n-3; jj+=4)
			{
			kernel_strsm_nn_ll_one_4x4_lib4(ii, &pD[ii*sdd], &pD[jj*ps], sdd, &pC[jj*ps+ii*sdc], &pD[jj*ps+ii*sdd], &pD[ii*ps+ii*sdd]);
			}
		if(jj<n)
			{
			kernel_strsm_nn_ll_one_4x4_vs_lib4(ii, &pD[ii*sdd], &pD[jj*ps], sdd, &pC[jj*ps+ii*sdc], &pD[jj*ps+ii*sdd], &pD[ii*ps+ii*sdd], m-ii, n-jj);
			}
		}
	if(m>ii)
		{
		goto left_4;
		}

	// common return if i==m
	return;

	left_4:
	jj = 0;
	// solve lower
	ie = n<ii ? n : ii; // ie is multiple of 4
	for( ; jj<ie; jj+=4)
		{
		kernel_strsm_nn_ru_inv_4x4_vs_lib4(jj, &pD[ii*sdd], &pD[jj*ps], sdd, &d_1, &pC[jj*ps+ii*sdc], &pD[jj*ps+ii*sdd], &pD[jj*ps+jj*sdd], &inv_diag_D[jj], m-ii, ie-jj);
		}
	// factorize
	if(jj<n)
		{
		kernel_sgetrf_nn_4x4_vs_lib4(jj, &pD[ii*sdd], &pD[jj*ps], sdd, &pC[jj*ps+ii*sdc], &pD[jj*ps+ii*sdd], &inv_diag_D[jj], m-ii, n-jj);
		jj+=4;
		}
	// solve upper 
	for( ; jj<n; jj+=4)
		{
		kernel_strsm_nn_ll_one_4x4_vs_lib4(ii, &pD[ii*sdd], &pD[jj*ps], sdd, &pC[jj*ps+ii*sdc], &pD[jj*ps+ii*sdd], &pD[ii*ps+ii*sdd], m-ii, n-jj);
		}
	return;

	}




// dgetrf row pivoting
void blasfeo_hp_sgetrf_rp(int m, int n, struct blasfeo_smat *sC, int ci, int cj, struct blasfeo_smat *sD, int di, int dj, int *ipiv)
	{
	if(ci!=0 | di!=0)
		{
#if defined(BLASFEO_REF_API)
		blasfeo_ref_sgetrf_rp(m, n, sC, ci, cj, sD, di, dj, ipiv);
		return;
#else
#ifdef EXT_DEP
		printf("\nblasfeo_sgetrf_rp: feature not implemented yet: ci=%d, di=%d\n", ci, di);
#endif	
		exit(1);
#endif
		}
	const int ps = 4;
	int sdc = sC->cn;
	int sdd = sD->cn;
	float *pC = sC->pA + cj*ps;
	float *pD = sD->pA + dj*ps;
	float *inv_diag_D = sD->dA; // XXX what to do if di and dj are not zero

	// needs to perform row-excanges on the yet-to-be-factorized matrix too
	if(pC!=pD)
		blasfeo_sgecp(m, n, sC, ci, cj, sD, di, dj);

	if(m<=0)
		return;
	
	if(di==0 && dj==0)
		sD->use_dA = 1;
	else
		sD->use_dA = 0;

	int ii, jj, i0, i1, j0, ll, p;

	float d1 = 1.0;
	float dm1 = -1.0;

//	// needs to perform row-excanges on the yet-to-be-factorized matrix too
//	if(pC!=pD)
//		sgecp_lib(m, n, 1.0, 0, pC, sdc, 0, pD, sdd);

	// minimum matrix size
	p = n<m ? n : m; // XXX

	// main loop
	// 4 columns at a time
	jj = 0;
	for(; jj<p-3; jj+=4) // XXX
		{
		// pivot & factorize & solve lower
		ii = jj;
		i0 = ii;
		for( ; ii<m-3; ii+=4)
			{
			kernel_sgemm_nn_4x4_lib4(jj, &dm1, &pD[ii*sdd], 0, &pD[jj*ps], sdd, &d1, &pD[jj*ps+ii*sdd], &pD[jj*ps+ii*sdd]);
			}
		if(m-ii>0)
			{
			kernel_sgemm_nn_4x4_vs_lib4(jj, &dm1, &pD[ii*sdd], 0, &pD[jj*ps], sdd, &d1, &pD[jj*ps+ii*sdd], &pD[jj*ps+ii*sdd], m-ii, 4);
			}
		kernel_sgetrf_pivot_4_lib4(m-i0, &pD[jj*ps+i0*sdd], sdd, &inv_diag_D[jj], &ipiv[i0]);
		ipiv[i0+0] += i0;
		if(ipiv[i0+0]!=i0+0)
			{
			srowsw_lib(jj, pD+(i0+0)/ps*ps*sdd+(i0+0)%ps, pD+(ipiv[i0+0])/ps*ps*sdd+(ipiv[i0+0])%ps);
			srowsw_lib(n-jj-4, pD+(i0+0)/ps*ps*sdd+(i0+0)%ps+(jj+4)*ps, pD+(ipiv[i0+0])/ps*ps*sdd+(ipiv[i0+0])%ps+(jj+4)*ps);
			}
		ipiv[i0+1] += i0;
		if(ipiv[i0+1]!=i0+1)
			{
			srowsw_lib(jj, pD+(i0+1)/ps*ps*sdd+(i0+1)%ps, pD+(ipiv[i0+1])/ps*ps*sdd+(ipiv[i0+1])%ps);
			srowsw_lib(n-jj-4, pD+(i0+1)/ps*ps*sdd+(i0+1)%ps+(jj+4)*ps, pD+(ipiv[i0+1])/ps*ps*sdd+(ipiv[i0+1])%ps+(jj+4)*ps);
			}
		ipiv[i0+2] += i0;
		if(ipiv[i0+2]!=i0+2)
			{
			srowsw_lib(jj, pD+(i0+2)/ps*ps*sdd+(i0+2)%ps, pD+(ipiv[i0+2])/ps*ps*sdd+(ipiv[i0+2])%ps);
			srowsw_lib(n-jj-4, pD+(i0+2)/ps*ps*sdd+(i0+2)%ps+(jj+4)*ps, pD+(ipiv[i0+2])/ps*ps*sdd+(ipiv[i0+2])%ps+(jj+4)*ps);
			}
		ipiv[i0+3] += i0;
		if(ipiv[i0+3]!=i0+3)
			{
			srowsw_lib(jj, pD+(i0+3)/ps*ps*sdd+(i0+3)%ps, pD+(ipiv[i0+3])/ps*ps*sdd+(ipiv[i0+3])%ps);
			srowsw_lib(n-jj-4, pD+(i0+3)/ps*ps*sdd+(i0+3)%ps+(jj+4)*ps, pD+(ipiv[i0+3])/ps*ps*sdd+(ipiv[i0+3])%ps+(jj+4)*ps);
			}

		// solve upper
		ll = jj+4;
		for( ; ll<n-3; ll+=4)
			{
			kernel_strsm_nn_ll_one_4x4_lib4(i0, &pD[i0*sdd], &pD[ll*ps], sdd, &pD[ll*ps+i0*sdd], &pD[ll*ps+i0*sdd], &pD[i0*ps+i0*sdd]);
			}
		if(n-ll>0)
			{
			kernel_strsm_nn_ll_one_4x4_vs_lib4(i0, &pD[i0*sdd], &pD[ll*ps], sdd, &pD[ll*ps+i0*sdd], &pD[ll*ps+i0*sdd], &pD[i0*ps+i0*sdd], 4, n-ll);
			}
		}
	if(m>=n)
		{
		if(n-jj>0)
			{
			goto left_n_4;
			}
		}
	else
		{
		if(m-jj>0)
			{
			goto left_m_4;
			}
		}

	// common return if jj==n
	return;

	// clean up

	left_n_4:
	// 1-4 columns at a time
	// pivot & factorize & solve lower
	ii = jj;
	i0 = ii;
	for( ; ii<m; ii+=4)
		{
		kernel_sgemm_nn_4x4_vs_lib4(jj, &dm1, &pD[ii*sdd], 0, &pD[jj*ps], sdd, &d1, &pD[jj*ps+ii*sdd], &pD[jj*ps+ii*sdd], m-ii, n-jj);
		}
	kernel_sgetrf_pivot_4_vs_lib4(m-i0, n-jj, &pD[jj*ps+i0*sdd], sdd, &inv_diag_D[jj], &ipiv[i0]);
	ipiv[i0+0] += i0;
	if(ipiv[i0+0]!=i0+0)
		{
		srowsw_lib(jj, pD+(i0+0)/ps*ps*sdd+(i0+0)%ps, pD+(ipiv[i0+0])/ps*ps*sdd+(ipiv[i0+0])%ps);
		srowsw_lib(n-jj-4, pD+(i0+0)/ps*ps*sdd+(i0+0)%ps+(jj+4)*ps, pD+(ipiv[i0+0])/ps*ps*sdd+(ipiv[i0+0])%ps+(jj+4)*ps);
		}
	if(n-jj>1)
		{
		ipiv[i0+1] += i0;
		if(ipiv[i0+1]!=i0+1)
			{
			srowsw_lib(jj, pD+(i0+1)/ps*ps*sdd+(i0+1)%ps, pD+(ipiv[i0+1])/ps*ps*sdd+(ipiv[i0+1])%ps);
			srowsw_lib(n-jj-4, pD+(i0+1)/ps*ps*sdd+(i0+1)%ps+(jj+4)*ps, pD+(ipiv[i0+1])/ps*ps*sdd+(ipiv[i0+1])%ps+(jj+4)*ps);
			}
		if(n-jj>2)
			{
			ipiv[i0+2] += i0;
			if(ipiv[i0+2]!=i0+2)
				{
				srowsw_lib(jj, pD+(i0+2)/ps*ps*sdd+(i0+2)%ps, pD+(ipiv[i0+2])/ps*ps*sdd+(ipiv[i0+2])%ps);
				srowsw_lib(n-jj-4, pD+(i0+2)/ps*ps*sdd+(i0+2)%ps+(jj+4)*ps, pD+(ipiv[i0+2])/ps*ps*sdd+(ipiv[i0+2])%ps+(jj+4)*ps);
				}
			if(n-jj>3)
				{
				ipiv[i0+3] += i0;
				if(ipiv[i0+3]!=i0+3)
					{
					srowsw_lib(jj, pD+(i0+3)/ps*ps*sdd+(i0+3)%ps, pD+(ipiv[i0+3])/ps*ps*sdd+(ipiv[i0+3])%ps);
					srowsw_lib(n-jj-4, pD+(i0+3)/ps*ps*sdd+(i0+3)%ps+(jj+4)*ps, pD+(ipiv[i0+3])/ps*ps*sdd+(ipiv[i0+3])%ps+(jj+4)*ps);
					}
				}
			}
		}

	// solve upper
	if(0) // there is no upper
		{
		ll = jj+4;
		for( ; ll<n; ll+=4)
			{
			kernel_strsm_nn_ll_one_4x4_vs_lib4(i0, &pD[i0*sdd], &pD[ll*ps], sdd, &pD[ll*ps+i0*sdd], &pD[ll*ps+i0*sdd], &pD[i0*ps+i0*sdd], m-i0, n-ll);
			}
		}
	return;


	left_m_4:
	// 1-4 rows at a time
	// pivot & factorize & solve lower
	ii = jj;
	i0 = ii;
	kernel_sgemm_nn_4x4_vs_lib4(jj, &dm1, &pD[ii*sdd], 0, &pD[jj*ps], sdd, &d1, &pD[jj*ps+ii*sdd], &pD[jj*ps+ii*sdd], m-ii, n-jj);
	kernel_sgetrf_pivot_4_vs_lib4(m-i0, n-jj, &pD[jj*ps+i0*sdd], sdd, &inv_diag_D[jj], &ipiv[i0]);
	ipiv[i0+0] += i0;
	if(ipiv[i0+0]!=i0+0)
		{
		srowsw_lib(jj, pD+(i0+0)/ps*ps*sdd+(i0+0)%ps, pD+(ipiv[i0+0])/ps*ps*sdd+(ipiv[i0+0])%ps);
		srowsw_lib(n-jj-4, pD+(i0+0)/ps*ps*sdd+(i0+0)%ps+(jj+4)*ps, pD+(ipiv[i0+0])/ps*ps*sdd+(ipiv[i0+0])%ps+(jj+4)*ps);
		}
	if(m-i0>1)
		{
		ipiv[i0+1] += i0;
		if(ipiv[i0+1]!=i0+1)
			{
			srowsw_lib(jj, pD+(i0+1)/ps*ps*sdd+(i0+1)%ps, pD+(ipiv[i0+1])/ps*ps*sdd+(ipiv[i0+1])%ps);
			srowsw_lib(n-jj-4, pD+(i0+1)/ps*ps*sdd+(i0+1)%ps+(jj+4)*ps, pD+(ipiv[i0+1])/ps*ps*sdd+(ipiv[i0+1])%ps+(jj+4)*ps);
			}
		if(m-i0>2)
			{
			ipiv[i0+2] += i0;
			if(ipiv[i0+2]!=i0+2)
				{
				srowsw_lib(jj, pD+(i0+2)/ps*ps*sdd+(i0+2)%ps, pD+(ipiv[i0+2])/ps*ps*sdd+(ipiv[i0+2])%ps);
				srowsw_lib(n-jj-4, pD+(i0+2)/ps*ps*sdd+(i0+2)%ps+(jj+4)*ps, pD+(ipiv[i0+2])/ps*ps*sdd+(ipiv[i0+2])%ps+(jj+4)*ps);
				}
			if(m-i0>3)
				{
				ipiv[i0+3] += i0;
				if(ipiv[i0+3]!=i0+3)
					{
					srowsw_lib(jj, pD+(i0+3)/ps*ps*sdd+(i0+3)%ps, pD+(ipiv[i0+3])/ps*ps*sdd+(ipiv[i0+3])%ps);
					srowsw_lib(n-jj-4, pD+(i0+3)/ps*ps*sdd+(i0+3)%ps+(jj+4)*ps, pD+(ipiv[i0+3])/ps*ps*sdd+(ipiv[i0+3])%ps+(jj+4)*ps);
					}
				}
			}
		}

	// solve upper
	ll = jj+4;
	for( ; ll<n; ll+=4)
		{
		kernel_strsm_nn_ll_one_4x4_vs_lib4(i0, &pD[i0*sdd], &pD[ll*ps], sdd, &pD[ll*ps+i0*sdd], &pD[ll*ps+i0*sdd], &pD[i0*ps+i0*sdd], m-i0, n-ll);
		}
	return;


	return;
	}



// sgetrf full (complete) pivoting, rank revealing
void blasfeo_hp_sgetrf_fp(int m, int n, float tol, struct blasfeo_smat *sC, int ci, int cj, struct blasfeo_smat *sD, int di, int dj, int *ipiv, int *jpiv, int *rank)
	{

	const int ps = 4;

	int cir = ci & (ps-1);
	int dir = di & (ps-1);

	int sdc = sC->cn;
	int sdd = sD->cn;
	float *pC = sC->pA + cj*ps + (ci - cir)*sdc;
	float *pD = sD->pA + dj*ps + (di - dir)*sdd;
	float *dD = sD->dA;

	if(di==0 && dj==0)
		sD->use_dA = 1;
	else
		sD->use_dA = 0;

	*rank = 0;

	if(m<=0 | n<=0)
		return;

	if(pC+cir!=pD+dir)
		blasfeo_sgecp(m, n, sC, ci, cj, sD, di, dj);

	int ii, jj, kk, ll, ip, jp, k0, k1, kp;
	float tmp, dmax, d_00_inv, d_ij, c_0, m_0;
	float l_v[4], m_v[4];
	float *pDi, *pDj, *pDk;

	int p = m<n ? m : n;
	// rows are indexed from the beginning of the first panel of D
	int me = m + dir;

	// the pivot is the element with largest absolute value, first in column-major order;
	// the trailing matrix is processed one panel of rows at a time, keeping track of the
	// largest absolute value in the panel; only if this is not smaller than the current
	// pivot, the panel is searched again for the column (jp) and later the row (ip)

	// pivot search
	dmax = 0.0;
	kp = dir;
	jp = 0;
	for(k0=dir; k0<me; k0=k1)
		{
		k1 = (k0 | (ps-1)) + 1;
		k1 = k1<me ? k1 : me;
		pDk = pD + (k0 & ~(ps-1))*sdd;
		m_0 = 0.0;
		if(k1-k0==ps)
			{
			for(ll=0; ll<ps; ll++)
				{
				m_v[ll] = 0.0;
				}
			for(jj=0; jj<n; jj++)
				{
				for(ll=0; ll<ps; ll++)
					{
					c_0 = fabsf(pDk[ll+jj*ps]);
					m_v[ll] = m_v[ll]>c_0 ? m_v[ll] : c_0;
					}
				}
			for(ll=0; ll<ps; ll++)
				{
				m_0 = m_0>m_v[ll] ? m_0 : m_v[ll];
				}
			}
		else
			{
			for(jj=0; jj<n; jj++)
				{
				for(kk=k0; kk<k1; kk++)
					{
					c_0 = fabsf(pDk[(kk&(ps-1))+jj*ps]);
					m_0 = m_0>c_0 ? m_0 : c_0;
					}
				}
			}
		if(m_0>=dmax & m_0>0.0)
			{
			// search the panel for the first column containing the maximum
			for(jj=0; jj<n; jj++)
				{
				for(kk=k0; kk<k1; kk++)
					{
					if(fabsf(pDk[(kk&(ps-1))+jj*ps])==m_0)
						goto found_0;
					}
				}
			found_0:
			if(jj<n & (m_0>dmax | jj<jp))
				{
				dmax = m_0;
				kp = k0;
				jp = jj;
				}
			}
		}

	// factorize
	for(ii=0; ii<p; ii++)
		{
		if(!(dmax>tol) | dmax==0.0)
			break;
		// search the panel for the row of the pivot
		ip = kp;
		k1 = (kp | (ps-1)) + 1;
		k1 = k1<me ? k1 : me;
		pDk = pD + (kp & ~(ps-1))*sdd + jp*ps;
		for(kk=kp; kk<k1; kk++)
			{
			if(fabsf(pDk[kk&(ps-1)])==dmax)
				{
				ip = kk;
				break;
				}
			}
		ipiv[ii] = ip - dir;
		jpiv[ii] = jp;
		// pointer to row ii
		kk = ii + dir;
		pDi = pD + (kk & ~(ps-1))*sdd + (kk&(ps-1));
		// swap rows
		if(ip!=kk)
			{
			pDj = pD + (ip & ~(ps-1))*sdd + (ip&(ps-1));
			for(jj=0; jj<n; jj++)
				{
				tmp = pDi[jj*ps];
				pDi[jj*ps] = pDj[jj*ps];
				pDj[jj*ps] = tmp;
				}
			}
		// swap columns
		if(jp!=ii)
			{
			for(k0=dir; k0<me; k0=k1)
				{
				k1 = (k0 | (ps-1)) + 1;
				k1 = k1<me ? k1 : me;
				pDk = pD + (k0 & ~(ps-1))*sdd;
				for(kk=k0; kk<k1; kk++)
					{
					tmp = pDk[(kk&(ps-1))+ii*ps];
					pDk[(kk&(ps-1))+ii*ps] = pDk[(kk&(ps-1))+jp*ps];
					pDk[(kk&(ps-1))+jp*ps] = tmp;
					}
				}
			}
		d_00_inv = 1.0/pDi[ii*ps];
		dD[ii] = d_00_inv;
		// scale column of L, rank-1 update of the trailing matrix and pivot search
		dmax = 0.0;
		kp = ii+1+dir;
		jp = ii+1;
		for(k0=ii+1+dir; k0<me; k0=k1)
			{
			k1 = (k0 | (ps-1)) + 1;
			k1 = k1<me ? k1 : me;
			pDk = pD + (k0 & ~(ps-1))*sdd;
			m_0 = 0.0;
			if(k1-k0==ps)
				{
				for(ll=0; ll<ps; ll++)
					{
					l_v[ll] = pDk[ll+ii*ps] * d_00_inv;
					pDk[ll+ii*ps] = l_v[ll];
					m_v[ll] = 0.0;
					}
				for(jj=ii+1; jj<n; jj++)
					{
					d_ij = pDi[jj*ps];
					for(ll=0; ll<ps; ll++)
						{
						c_0 = pDk[ll+jj*ps] - l_v[ll] * d_ij;
						pDk[ll+jj*ps] = c_0;
						c_0 = fabsf(c_0);
						m_v[ll] = m_v[ll]>c_0 ? m_v[ll] : c_0;
						}
					}
				for(ll=0; ll<ps; ll++)
					{
					m_0 = m_0>m_v[ll] ? m_0 : m_v[ll];
					}
				}
			else
				{
				for(kk=k0; kk<k1; kk++)
					{
					pDk[(kk&(ps-1))+ii*ps] *= d_00_inv;
					}
				for(jj=ii+1; jj<n; jj++)
					{
					d_ij = pDi[jj*ps];
					for(kk=k0; kk<k1; kk++)
						{
						c_0 = pDk[(kk&(ps-1))+jj*ps] - pDk[(kk&(ps-1))+ii*ps] * d_ij;
						pDk[(kk&(ps-1))+jj*ps] = c_0;
						c_0 = fabsf(c_0);
						m_0 = m_0>c_0 ? m_0 : c_0;
						}
					}
				}
			if(m_0>=dmax & m_0>0.0)
				{
				// search the panel for the first column containing the maximum
				for(jj=ii+1; jj<n; jj++)
					{
					for(kk=k0; kk<k1; kk++)
						{
						if(fabsf(pDk[(kk&(ps-1))+jj*ps])==m_0)
							goto found_1;
						}
					}
				found_1:
				if(jj<n & (m_0>dmax | jj<jp))
					{
					dmax = m_0;
					kp = k0;
					jp = jj;
					}
				}
			}
		}

	*rank = ii;

	// no pivoting on the rank deficient part
	for(; ii<p; ii++)
		{
		ipiv[ii] = ii;
		jpiv[ii] = ii;
		dD[ii] = 0.0;
		}

	return;

	}


// largest absolute value (first in case of ties) in rows k0,...,me-1 of a column of a lib4 matrix;
// pD points to the column in the first panel, rows are indexed from the beginning of the first panel
static inline void blasfeo_hp_scolamax_lib4(int k0, int me, float *pD, int sdd, float *ptr_amax, int *ptr_iamax)
	{
	const int ps = 4;
	int k1, kk, ll, kp;
	float c_0, m_0, amax;
	float m_v[4];
	float *pDk;
	amax = 0.0;
	kp = k0;
	// largest absolute value per panel (branchless), keep the first panel with the overall largest
	for(kk=k0; kk<me; kk=k1)
		{
		k1 = (kk | (ps-1)) + 1;
		k1 = k1<me ? k1 : me;
		pDk = pD + (kk & ~(ps-1))*sdd;
		if(k1-kk==ps)
			{
			for(ll=0; ll<ps; ll++)
				m_v[ll] = fabsf(pDk[ll]);
			m_0 = m_v[0]>m_v[1] ? m_v[0] : m_v[1];
			c_0 = m_v[2]>m_v[3] ? m_v[2] : m_v[3];
			m_0 = m_0>c_0 ? m_0 : c_0;
			}
		else
			{
			m_0 = 0.0;
			for(ll=kk; ll<k1; ll++)
				{
				c_0 = fabsf(pDk[ll&(ps-1)]);
				m_0 = m_0>c_0 ? m_0 : c_0;
				}
			}
		if(m_0>amax)
			{
			amax = m_0;
			kp = kk;
			}
		}
	// first row of the panel with the largest absolute value
	if(amax>0.0)
		{
		pDk = pD + (kp & ~(ps-1))*sdd;
		while(fabsf(pDk[kp&(ps-1)])!=amax)
			kp++;
		}
	else
		{
		kp = k0;
		}
	*ptr_amax = amax;
	*ptr_iamax = kp;
	return;
	}



// largest absolute value (first in column-major order) in rows k0,...,me-1 and columns j0,...,n-1
// of a lib4 matrix; rows are indexed from the beginning of the first panel
static void blasfeo_hp_sgeamax_lib4(int k0, int me, int j0, int n, float *pD, int sdd, float *ptr_amax, int *ptr_iamax, int *ptr_jamax)
	{
	const int ps = 4;
	int jj, it, imax, jmax;
	float c_0, amax;
	amax = 0.0;
	imax = k0;
	jmax = j0;
	for(jj=j0; jj<n; jj++)
		{
		blasfeo_hp_scolamax_lib4(k0, me, pD+jj*ps, sdd, &c_0, &it);
		if(c_0>amax)
			{
			amax = c_0;
			imax = it;
			jmax = jj;
			}
		}
	*ptr_amax = amax;
	*ptr_iamax = imax;
	*ptr_jamax = jmax;
	return;
	}



// blocked rook pivoting: the trailing matrix is updated once per block of S_GETRF_RK_NB pivots
// using dgemm; within a block, the searched rows and columns are computed on the fly with dgemv
// from the delayed updates
#define S_GETRF_RK_NB 16
// use the blocked algorithm for min(m,n) >= S_GETRF_RK_BLOCK_MIN
#define S_GETRF_RK_BLOCK_MIN 96



// largest absolute value (first in case of ties) of the n entries of x
static void blasfeo_hp_samax(int n, float *x, float *ptr_amax, int *ptr_iamax)
	{
	int ii, imax;
	float c_0, amax;
	amax = 0.0;
	imax = 0;
	for(ii=0; ii<n; ii++)
		{
		c_0 = fabsf(x[ii]);
		if(c_0>amax)
			{
			amax = c_0;
			imax = ii;
			}
		}
	*ptr_amax = amax;
	*ptr_iamax = imax;
	return;
	}



// column c of the trailing matrix in rows r,...,m-1 (relative to di), including the delayed update
// from the nl columns of L and rows of U starting at k0: w = D(r:m,c) - L(r:m,k0:k0+nl) * U(k0:k0+nl,c)
static void blasfeo_hp_sgetrf_rk_col(int m, int r, int c, int k0, int nl, struct blasfeo_smat *sD, int di, int dj, struct blasfeo_svec *sw, struct blasfeo_svec *su)
	{
	const int ps = 4;
	int dir = di & (ps-1);
	int sdd = sD->cn;
	float *pD = sD->pA + dj*ps + (di - dir)*sdd;
	float *w = sw->pa;
	float *u = su->pa;
	float *pDk;
	int k0r, k1, kk, l;
	// gather column
	for(k0r=r+dir; k0r<m+dir; k0r=k1)
		{
		k1 = (k0r | (ps-1)) + 1;
		k1 = k1<m+dir ? k1 : m+dir;
		pDk = pD + (k0r & ~(ps-1))*sdd + c*ps;
		for(kk=k0r; kk<k1; kk++)
			w[kk-dir-r] = pDk[kk&(ps-1)];
		}
	if(nl>0)
		{
		for(l=0; l<nl; l++)
			{
			kk = k0 + l + dir;
			u[l] = pD[(kk & ~(ps-1))*sdd+(kk&(ps-1))+c*ps];
			}
		blasfeo_sgemv_n(m-r, nl, -1.0, sD, di+r, dj+k0, su, 0, 1.0, sw, 0, sw, 0);
		}
	return;
	}



// row r (relative to di) of the trailing matrix in columns c,...,n-1, including the delayed update
// from the nl columns of L and rows of U starting at k0: v = D(r,c:n) - L(r,k0:k0+nl) * U(k0:k0+nl,c:n)
static void blasfeo_hp_sgetrf_rk_row(int n, int r, int c, int k0, int nl, struct blasfeo_smat *sD, int di, int dj, struct blasfeo_svec *sv, struct blasfeo_svec *sl)
	{
	const int ps = 4;
	int dir = di & (ps-1);
	int sdd = sD->cn;
	int rr = r + dir;
	float *pr = sD->pA + dj*ps + (di - dir)*sdd + (rr & ~(ps-1))*sdd + (rr&(ps-1));
	float *v = sv->pa;
	float *lr = sl->pa;
	int jj, l;
	// gather row
	for(jj=c; jj<n; jj++)
		v[jj-c] = pr[jj*ps];
	if(nl>0)
		{
		for(l=0; l<nl; l++)
			lr[l] = pr[(k0+l)*ps];
		blasfeo_sgemv_t(nl, n-c, -1.0, sD, di+k0, dj+c, sl, 0, 1.0, sv, 0, sv, 0);
		}
	return;
	}



// blocked rook pivoting on D (already holding a copy of C), returns the rank
static int blasfeo_hp_sgetrf_rk_blk_lib4(int m, int n, float tol, struct blasfeo_smat *sD, int di, int dj, int *ipiv, int *jpiv)
	{
	const int ps = 4;
	int dir = di & (ps-1);
	int sdd = sD->cn;
	float *pD = sD->pA + dj*ps + (di - dir)*sdd;
	float *dD = sD->dA;

	int ii, jj, kk, k0, k1, kb, nl, rr, ip, jp, it, jt, iw, rank;
	float tmp, dmax, rmax, cmax, wmax, d_00_inv;
	float *pDi, *pDj, *pDk;

	int p = m<n ? m : n;
	// rows are indexed from the beginning of the first panel of D
	int me = m + dir;

	// work space: column w (m), row v (n), U column u and L row l (S_GETRF_RK_NB each)
	int mw = (m+7)/8*8;
	int nw = (n+7)/8*8;
	int sw = mw + nw + 2*S_GETRF_RK_NB;
	ALIGNED( float work0[4*K_MAX_STACK], 64 );
	float *work = work0;
	void *mem = NULL;
	if(sw>4*K_MAX_STACK)
		{
		mem = malloc(sw*sizeof(float)+64);
		work = (float *) ((((size_t) mem) + 63) / 64 * 64);
		}
	struct blasfeo_svec svw, svv, svu, svl;
	blasfeo_create_svec(m, &svw, work);
	blasfeo_create_svec(n, &svv, work+mw);
	blasfeo_create_svec(S_GETRF_RK_NB, &svu, work+mw+nw);
	blasfeo_create_svec(S_GETRF_RK_NB, &svl, work+mw+nw+S_GETRF_RK_NB);
	float *w = svw.pa;
	float *v = svv.pa;

	rank = p;
	for(k0=0; k0<p; k0=kb)
		{
		kb = k0+S_GETRF_RK_NB<p ? k0+S_GETRF_RK_NB : p;
		for(ii=k0; ii<kb; ii++)
			{
			// number of delayed updates
			nl = ii - k0;
			// row ii
			rr = ii + dir;
			// largest absolute value in the first column of the trailing matrix
			jp = ii;
			blasfeo_hp_sgetrf_rk_col(m, ii, jp, k0, nl, sD, di, dj, &svw, &svu);
			blasfeo_hp_samax(m-ii, w, &dmax, &ip);
			// largest absolute value of the computed column w, and its first row
			wmax = dmax;
			iw = ip;
			ip += ii;
			// rook search: on exit, w holds column jp and v holds row ip, both updated
			while(1)
				{
				blasfeo_hp_sgetrf_rk_row(n, ip, ii, k0, nl, sD, di, dj, &svv, &svl);
				blasfeo_hp_samax(n-ii, v, &rmax, &jt);
				if(!(rmax>dmax))
					break;
				dmax = rmax;
				jp = jt + ii;
				blasfeo_hp_sgetrf_rk_col(m, ii, jp, k0, nl, sD, di, dj, &svw, &svu);
				blasfeo_hp_samax(m-ii, w, &cmax, &it);
				wmax = cmax;
				iw = it;
				if(!(cmax>dmax))
					break;
				dmax = cmax;
				ip = it + ii;
				}
			// small rook pivot: verify the rank decision with a full search of the trailing matrix, as in
			// complete pivoting; the delayed update is applied first, and the block restarts at ii
			if(!(dmax>tol & dmax!=0.0))
				{
				if(nl>0)
					{
					blasfeo_sgemm_nn(m-ii, n-ii, nl, -1.0, sD, di+ii, dj+k0, sD, di+k0, dj+ii, 1.0, sD, di+ii, dj+ii, sD, di+ii, dj+ii);
					k0 = ii;
					nl = 0;
					}
				blasfeo_hp_sgeamax_lib4(rr, me, ii, n, pD, sdd, &dmax, &ip, &jp);
				if(!(dmax>tol & dmax!=0.0))
					{
					// numerical rank ii: the trailing matrix holds the Schur complement
					rank = ii;
					goto end;
					}
				// complete pivoting step
				ip -= dir;
				blasfeo_hp_sgetrf_rk_col(m, ii, jp, k0, nl, sD, di, dj, &svw, &svu);
				blasfeo_hp_samax(m-ii, w, &wmax, &iw);
				blasfeo_hp_sgetrf_rk_row(n, ip, ii, k0, nl, sD, di, dj, &svv, &svl);
				}
			// the entries of the searched rows and columns are computed separately (gemv_t and gemv_n) and
			// rounded differently, which matters if the trailing matrix is at rounding level (rank deficient
			// matrices): make the pivot the largest in absolute value in its computed column w, so that
			// |L| <= 1, and (in a few steps of rook search on the computed values) in its computed row v
			for(it=0; it<4; it++)
				{
				if(wmax>fabsf(w[ip-ii]))
					{
					ip = iw + ii;
					blasfeo_hp_sgetrf_rk_row(n, ip, ii, k0, nl, sD, di, dj, &svv, &svl);
					}
				blasfeo_hp_samax(n-ii, v, &rmax, &jt);
				if(!(rmax>fabsf(w[ip-ii])) | it==3)
					break;
				jp = jt + ii;
				blasfeo_hp_sgetrf_rk_col(m, ii, jp, k0, nl, sD, di, dj, &svw, &svu);
				blasfeo_hp_samax(m-ii, w, &wmax, &iw);
				}
			ipiv[ii] = ip;
			jpiv[ii] = jp;
			// pointer to row ii
			pDi = pD + (rr & ~(ps-1))*sdd + (rr&(ps-1));
			// swap rows
			if(ip!=ii)
				{
				kk = ip + dir;
				pDj = pD + (kk & ~(ps-1))*sdd + (kk&(ps-1));
				for(jj=0; jj<n; jj++)
					{
					tmp = pDi[jj*ps];
					pDi[jj*ps] = pDj[jj*ps];
					pDj[jj*ps] = tmp;
					}
				tmp = w[0];
				w[0] = w[ip-ii];
				w[ip-ii] = tmp;
				}
			// swap columns
			if(jp!=ii)
				{
				for(k1=dir; k1<me; k1+=ps-(k1&(ps-1)))
					{
					pDk = pD + (k1 & ~(ps-1))*sdd;
					for(kk=k1; kk<me & kk<((k1|(ps-1))+1); kk++)
						{
						tmp = pDk[(kk&(ps-1))+ii*ps];
						pDk[(kk&(ps-1))+ii*ps] = pDk[(kk&(ps-1))+jp*ps];
						pDk[(kk&(ps-1))+jp*ps] = tmp;
						}
					}
				tmp = v[0];
				v[0] = v[jp-ii];
				v[jp-ii] = tmp;
				}
			// store the updated column of L, scaled by the inverse of the pivot, and row of U
			d_00_inv = 1.0/w[0];
			dD[ii] = d_00_inv;
			pDi[ii*ps] = w[0];
			// if the pivot is so small that its reciprocal overflows, divide by it
			if(fabsf(w[0])<FLT_MIN)
				{
				for(kk=1; kk<m-ii; kk++)
					w[kk] = w[kk] / w[0];
				d_00_inv = 1.0;
				}
			for(k1=rr+1; k1<me; k1+=ps-(k1&(ps-1)))
				{
				pDk = pD + (k1 & ~(ps-1))*sdd + ii*ps;
				for(kk=k1; kk<me & kk<((k1|(ps-1))+1); kk++)
					{
					pDk[kk&(ps-1)] = w[kk-rr] * d_00_inv;
					}
				}
			if(nl>0)
				{
				for(jj=ii+1; jj<n; jj++)
					pDi[jj*ps] = v[jj-ii];
				}
			}
		// update of the trailing matrix
		if(kb<m & kb<n)
			blasfeo_sgemm_nn(m-kb, n-kb, kb-k0, -1.0, sD, di+kb, dj+k0, sD, di+k0, dj+kb, 1.0, sD, di+kb, dj+kb, sD, di+kb, dj+kb);
		}

	end:
	if(mem!=NULL)
		free(mem);

	return rank;
	}



// sgetrf rook pivoting, rank revealing
void blasfeo_hp_sgetrf_rk(int m, int n, float tol, struct blasfeo_smat *sC, int ci, int cj, struct blasfeo_smat *sD, int di, int dj, int *ipiv, int *jpiv, int *rank)
	{

	const int ps = 4;

	int cir = ci & (ps-1);
	int dir = di & (ps-1);

	int sdc = sC->cn;
	int sdd = sD->cn;
	float *pC = sC->pA + cj*ps + (ci - cir)*sdc;
	float *pD = sD->pA + dj*ps + (di - dir)*sdd;
	float *dD = sD->dA;

	if(di==0 && dj==0)
		sD->use_dA = 1;
	else
		sD->use_dA = 0;

	*rank = 0;

	if(m<=0 | n<=0)
		return;

	if(pC+cir!=pD+dir)
		blasfeo_sgecp(m, n, sC, ci, cj, sD, di, dj);

	int ii, jj, kk, ll, ip, jp, it, jt, k0, k1;

	int p = m<n ? m : n;

	if(p>=S_GETRF_RK_BLOCK_MIN)
		{
		ii = blasfeo_hp_sgetrf_rk_blk_lib4(m, n, tol, sD, di, dj, ipiv, jpiv);
		*rank = ii;
		// no pivoting on the rank deficient part
		for(; ii<p; ii++)
			{
			ipiv[ii] = ii;
			jpiv[ii] = ii;
			dD[ii] = 0.0;
			}
		// dgemm invalidates the stored inverse diagonal
		if(di==0 && dj==0)
			sD->use_dA = 1;
		else
			sD->use_dA = 0;
		return;
		}

	float tmp, dmax, cmax, rmax, d_00_inv, d_ij, c_0, m_0, dmax_n;
	float l_v[4], m_v[4];
	int ip_n;
	float *pDi, *pDj, *pDk;

	// rows are indexed from the beginning of the first panel of D
	int me = m + dir;

	// the pivot is the largest in absolute value in both its row and its column of the trailing matrix:
	// starting from the first column, the row and the column of the current candidate are searched
	// alternately, until it is not improved; ties are broken by the first index, as in the reference
	// implementation; if the rook pivot is <= tol (or zero), a full search of the trailing matrix
	// verifies the rank decision (complete pivoting step)

	// largest absolute value in the first column; in the following steps, it is computed for
	// column ii+1 during the rank-1 update
	blasfeo_hp_scolamax_lib4(dir, me, pD, sdd, &dmax_n, &ip_n);

	// factorize
	for(ii=0; ii<p; ii++)
		{
		// row ii
		kk = ii + dir;
		// largest absolute value in the first column of the trailing matrix (from the previous update)
		dmax = dmax_n;
		ip = ip_n;
		jp = ii;
		// rook search
		while(1)
			{
			// row search: largest absolute value (branchless), then its first column only if it improves
			pDk = pD + (ip & ~(ps-1))*sdd + (ip&(ps-1));
			rmax = 0.0;
			for(jj=ii; jj<n; jj++)
				{
				c_0 = fabsf(pDk[jj*ps]);
				rmax = rmax>c_0 ? rmax : c_0;
				}
			if(!(rmax>dmax))
				break;
			for(jt=ii; fabsf(pDk[jt*ps])!=rmax; jt++)
				;
			dmax = rmax;
			jp = jt;
			// column search
			blasfeo_hp_scolamax_lib4(kk, me, pD+jp*ps, sdd, &cmax, &it);
			if(!(cmax>dmax))
				break;
			dmax = cmax;
			ip = it;
			}
		// small rook pivot: verify the rank decision with a full search of the trailing matrix, as in
		// complete pivoting: stop if all its entries are <= tol (or zero), otherwise use its largest entry
		if(!(dmax>tol & dmax!=0.0))
			{
			blasfeo_hp_sgeamax_lib4(kk, me, ii, n, pD, sdd, &dmax, &ip, &jp);
			if(!(dmax>tol & dmax!=0.0))
				break;
			}
		ipiv[ii] = ip - dir;
		jpiv[ii] = jp;
		// pointer to row ii
		pDi = pD + (kk & ~(ps-1))*sdd + (kk&(ps-1));
		// swap rows
		if(ip!=kk)
			{
			pDj = pD + (ip & ~(ps-1))*sdd + (ip&(ps-1));
			for(jj=0; jj<n; jj++)
				{
				tmp = pDi[jj*ps];
				pDi[jj*ps] = pDj[jj*ps];
				pDj[jj*ps] = tmp;
				}
			}
		// swap columns
		if(jp!=ii)
			{
			for(k0=dir; k0<me; k0=k1)
				{
				k1 = (k0 | (ps-1)) + 1;
				k1 = k1<me ? k1 : me;
				pDk = pD + (k0 & ~(ps-1))*sdd;
				for(it=k0; it<k1; it++)
					{
					tmp = pDk[(it&(ps-1))+ii*ps];
					pDk[(it&(ps-1))+ii*ps] = pDk[(it&(ps-1))+jp*ps];
					pDk[(it&(ps-1))+jp*ps] = tmp;
					}
				}
			}
		d_00_inv = 1.0/pDi[ii*ps];
		dD[ii] = d_00_inv;
		// if the pivot is so small that its reciprocal overflows, divide the column of L by it here,
		// and scale by one in the rank-1 update
		if(fabsf(pDi[ii*ps])<FLT_MIN)
			{
			for(k0=kk+1; k0<me; k0=k1)
				{
				k1 = (k0 | (ps-1)) + 1;
				k1 = k1<me ? k1 : me;
				pDk = pD + (k0 & ~(ps-1))*sdd;
				for(it=k0; it<k1; it++)
					pDk[(it&(ps-1))+ii*ps] = pDk[(it&(ps-1))+ii*ps] / pDi[ii*ps];
				}
			d_00_inv = 1.0;
			}
		// scale column of L and rank-1 update of the trailing matrix;
		// the largest absolute value of column ii+1 is computed on the fly (first panel containing it)
		dmax_n = 0.0;
		ip_n = kk+1;
		for(k0=kk+1; k0<me; k0=k1)
			{
			k1 = (k0 | (ps-1)) + 1;
			k1 = k1<me ? k1 : me;
			pDk = pD + (k0 & ~(ps-1))*sdd;
			m_0 = 0.0;
			if(k1-k0==ps)
				{
				for(ll=0; ll<ps; ll++)
					{
					l_v[ll] = pDk[ll+ii*ps] * d_00_inv;
					pDk[ll+ii*ps] = l_v[ll];
					}
				if(ii+1<n)
					{
					d_ij = pDi[(ii+1)*ps];
					for(ll=0; ll<ps; ll++)
						{
						c_0 = pDk[ll+(ii+1)*ps] - l_v[ll] * d_ij;
						pDk[ll+(ii+1)*ps] = c_0;
						m_v[ll] = fabsf(c_0);
						}
					m_0 = m_v[0]>m_v[1] ? m_v[0] : m_v[1];
					c_0 = m_v[2]>m_v[3] ? m_v[2] : m_v[3];
					m_0 = m_0>c_0 ? m_0 : c_0;
					}
				for(jj=ii+2; jj<n; jj++)
					{
					d_ij = pDi[jj*ps];
					for(ll=0; ll<ps; ll++)
						{
						pDk[ll+jj*ps] = pDk[ll+jj*ps] - l_v[ll] * d_ij;
						}
					}
				}
			else
				{
				for(it=k0; it<k1; it++)
					{
					pDk[(it&(ps-1))+ii*ps] *= d_00_inv;
					}
				if(ii+1<n)
					{
					d_ij = pDi[(ii+1)*ps];
					for(it=k0; it<k1; it++)
						{
						c_0 = pDk[(it&(ps-1))+(ii+1)*ps] - pDk[(it&(ps-1))+ii*ps] * d_ij;
						pDk[(it&(ps-1))+(ii+1)*ps] = c_0;
						c_0 = fabsf(c_0);
						m_0 = m_0>c_0 ? m_0 : c_0;
						}
					}
				for(jj=ii+2; jj<n; jj++)
					{
					d_ij = pDi[jj*ps];
					for(it=k0; it<k1; it++)
						{
						pDk[(it&(ps-1))+jj*ps] = pDk[(it&(ps-1))+jj*ps] - pDk[(it&(ps-1))+ii*ps] * d_ij;
						}
					}
				}
			if(m_0>dmax_n)
				{
				dmax_n = m_0;
				ip_n = k0;
				}
			}
		// first row of the panel with the largest absolute value in column ii+1
		if(dmax_n>0.0)
			{
			pDk = pD + (ip_n & ~(ps-1))*sdd + (ii+1)*ps;
			while(fabsf(pDk[ip_n&(ps-1)])!=dmax_n)
				ip_n++;
			}
		}

	*rank = ii;

	// no pivoting on the rank deficient part
	for(; ii<p; ii++)
		{
		ipiv[ii] = ii;
		jpiv[ii] = ii;
		dD[ii] = 0.0;
		}

	return;

	}



int blasfeo_hp_sgeqrf_worksize(int m, int n)
	{
	return 0;
	}



void blasfeo_hp_sgeqrf(int m, int n, struct blasfeo_smat *sC, int ci, int cj, struct blasfeo_smat *sD, int di, int dj, void *work)
	{
	if(m<=0 | n<=0)
		return;
#if defined(BLASFEO_REF_API)
	blasfeo_ref_sgeqrf(m, n, sC, ci, cj, sD, di, dj, work);
	return;
#else
#ifdef EXT_DEP
	printf("\nblasfeo_sgeqrf: feature not implemented yet\n");
#endif	
	exit(1);
#endif
	return;
	}



int blasfeo_hp_sgelqf_worksize(int m, int n)
	{
	return 0;
	}



void blasfeo_hp_sgelqf(int m, int n, struct blasfeo_smat *sC, int ci, int cj, struct blasfeo_smat *sD, int di, int dj, void *work)
	{
	if(m<=0 | n<=0)
		return;
#if defined(BLASFEO_REF_API)
	blasfeo_ref_sgelqf(m, n, sC, ci, cj, sD, di, dj, work);
	return;
#else
#ifdef EXT_DEP
	printf("\nblasfeo_sgelqf: feature not implemented yet\n");
#endif	
	exit(1);
#endif
	return;
	}



int blasfeo_hp_sorglq_worksize(int m, int n, int k)
	{
	return 0;
	}



void blasfeo_hp_sorglq(int m, int n, int k, struct blasfeo_smat *sC, int ci, int cj, struct blasfeo_smat *sD, int di, int dj, void *work)
	{
	if(m<=0 | n<=0)
		return;
#if defined(BLASFEO_REF_API)
	blasfeo_ref_sorglq(m, n, k, sC, ci, cj, sD, di, dj, work);
	return;
#else
#ifdef EXT_DEP
	printf("\nblasfeo_sorglq: feature not implemented yet\n");
#endif	
	exit(1);
#endif
	return;
	}



// LQ factorization with positive diagonal elements
void blasfeo_hp_sgelqf_pd(int m, int n, struct blasfeo_smat *sC, int ci, int cj, struct blasfeo_smat *sD, int di, int dj, void *work)
	{
	if(m<=0 | n<=0)
		return;
#if defined(BLASFEO_REF_API)
	blasfeo_ref_sgelqf_pd(m, n, sC, ci, cj, sD, di, dj, work);
	return;
#else
#ifdef EXT_DEP
	printf("\nblasfeo_sgelqf_pd: feature not implemented yet\n");
#endif	
	exit(1);
#endif
	return;
	}



// LQ factorization with positive diagonal elements, array of matrices
// [L, A] <= lq( [L. A] )
// L lower triangular, of size (m)x(m)
// A full of size (m)x(n1)
void blasfeo_hp_sgelqf_pd_la(int m, int n1, struct blasfeo_smat *sD, int di, int dj, struct blasfeo_smat *sA, int ai, int aj, void *work)
	{
	if(m<=0)
		return;
#if defined(BLASFEO_REF_API)
	blasfeo_ref_sgelqf_pd_la(m, n1, sD, di, dj, sA, ai, aj, work);
	return;
#else
#ifdef EXT_DEP
	printf("\nblasfeo_sgelqf_pd_la: feature not implemented yet\n");
#endif	
	exit(1);
#endif
	return;
	}



// LQ factorization with positive diagonal elements, array of matrices
// [L, L, A] <= lq( [L. L, A] )
// L lower triangular, of size (m)x(m)
// A full of size (m)x(n1)
void blasfeo_hp_sgelqf_pd_lla(int m, int n1, struct blasfeo_smat *sD, int di, int dj, struct blasfeo_smat *sL, int li, int lj, struct blasfeo_smat *sA, int ai, int aj, void *work)
	{
	if(m<=0)
		return;
#if defined(BLASFEO_REF_API)
	blasfeo_ref_sgelqf_pd_lla(m, n1, sD, di, dj, sL, li, lj, sA, ai, aj, work);
	return;
#else
#ifdef EXT_DEP
	printf("\nblasfeo_sgelqf_pd_lla: feature not implemented yet\n");
#endif	
	exit(1);
#endif
	}



#if defined(LA_HIGH_PERFORMANCE)



void blasfeo_spotrf_l(int m, struct blasfeo_smat *sC, int ci, int cj, struct blasfeo_smat *sD, int di, int dj)
	{
	blasfeo_hp_spotrf_l(m, sC, ci, cj, sD, di, dj);
	}



void blasfeo_spotrf_l_mn(int m, int n, struct blasfeo_smat *sC, int ci, int cj, struct blasfeo_smat *sD, int di, int dj)
	{
	blasfeo_hp_spotrf_l_mn(m, n, sC, ci, cj, sD, di, dj);
	}



void blasfeo_spotrf_u(int m, struct blasfeo_smat *sC, int ci, int cj, struct blasfeo_smat *sD, int di, int dj)
	{
	blasfeo_hp_spotrf_u(m, sC, ci, cj, sD, di, dj);
	}



void blasfeo_ssyrk_spotrf_ln_mn(int m, int n, int k, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_smat *sB, int bi, int bj, struct blasfeo_smat *sC, int ci, int cj, struct blasfeo_smat *sD, int di, int dj)
	{
	blasfeo_hp_ssyrk_spotrf_ln_mn(m, n, k, sA, ai, aj, sB, bi, bj, sC, ci, cj, sD, di, dj);
	}



void blasfeo_ssyrk_spotrf_ln(int m, int k, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_smat *sB, int bi, int bj, struct blasfeo_smat *sC, int ci, int cj, struct blasfeo_smat *sD, int di, int dj)
	{
	blasfeo_hp_ssyrk_spotrf_ln(m, k, sA, ai, aj, sB, bi, bj, sC, ci, cj, sD, di, dj);
	}



void blasfeo_sgetrf_np(int m, int n, struct blasfeo_smat *sC, int ci, int cj, struct blasfeo_smat *sD, int di, int dj)
	{
	blasfeo_hp_sgetrf_np(m, n, sC, ci, cj, sD, di, dj);
	}



void blasfeo_sgetrf_rp(int m, int n, struct blasfeo_smat *sC, int ci, int cj, struct blasfeo_smat *sD, int di, int dj, int *ipiv)
	{
	blasfeo_hp_sgetrf_rp(m, n, sC, ci, cj, sD, di, dj, ipiv);
	}



void blasfeo_sgetrf_fp(int m, int n, float tol, struct blasfeo_smat *sC, int ci, int cj, struct blasfeo_smat *sD, int di, int dj, int *ipiv, int *jpiv, int *rank)
	{
	blasfeo_hp_sgetrf_fp(m, n, tol, sC, ci, cj, sD, di, dj, ipiv, jpiv, rank);
	}



void blasfeo_sgetrf_rk(int m, int n, float tol, struct blasfeo_smat *sC, int ci, int cj, struct blasfeo_smat *sD, int di, int dj, int *ipiv, int *jpiv, int *rank)
	{
	blasfeo_hp_sgetrf_rk(m, n, tol, sC, ci, cj, sD, di, dj, ipiv, jpiv, rank);
	}



int blasfeo_sgeqrf_worksize(int m, int n)
	{
	return blasfeo_hp_sgeqrf_worksize(m, n);
	}



void blasfeo_sgeqrf(int m, int n, struct blasfeo_smat *sC, int ci, int cj, struct blasfeo_smat *sD, int di, int dj, void *v_work)
	{
	blasfeo_hp_sgeqrf(m, n, sC, ci, cj, sD, di, dj, v_work);
	}



int blasfeo_sgelqf_worksize(int m, int n)
	{
	return blasfeo_hp_sgelqf_worksize(m, n);
	}



void blasfeo_sgelqf(int m, int n, struct blasfeo_smat *sC, int ci, int cj, struct blasfeo_smat *sD, int di, int dj, void *work)
	{
	blasfeo_hp_sgelqf(m, n, sC, ci, cj, sD, di, dj, work);
	}



int blasfeo_sorglq_worksize(int m, int n, int k)
	{
	return blasfeo_hp_sorglq_worksize(m, n, k);
	}



void blasfeo_sorglq(int m, int n, int k, struct blasfeo_smat *sC, int ci, int cj, struct blasfeo_smat *sD, int di, int dj, void *work)
	{
	blasfeo_hp_sorglq(m, n, k, sC, ci, cj, sD, di, dj, work);
	}



void blasfeo_sgelqf_pd(int m, int n, struct blasfeo_smat *sC, int ci, int cj, struct blasfeo_smat *sD, int di, int dj, void *work)
	{
	blasfeo_hp_sgelqf_pd(m, n, sC, ci, cj, sD, di, cj, work);
	}



void blasfeo_sgelqf_pd_la(int m, int n1, struct blasfeo_smat *sD, int di, int dj, struct blasfeo_smat *sA, int ai, int aj, void *work)
	{
	blasfeo_hp_sgelqf_pd_la(m, n1, sD, di, dj, sA, ai, aj, work);
	}



void blasfeo_sgelqf_pd_lla(int m, int n1, struct blasfeo_smat *sD, int di, int dj, struct blasfeo_smat *sL, int li, int lj, struct blasfeo_smat *sA, int ai, int aj, void *work)
	{
	blasfeo_hp_sgelqf_pd_lla(m, n1, sD, di, dj, sL, li, lj, sA, ai, aj, work);
	}



#endif
