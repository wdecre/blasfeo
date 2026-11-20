/**************************************************************************************************
*                                                                                                 *
* This file is part of BLASFEO.                                                                   *
*                                                                                                 *
* BLASFEO -- BLAS For Embedded Optimization.                                                      *
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

#include <blasfeo.h>

// largest absolute value of the entries of a m x n matrix
static double dmat_amax(int m, int n, struct blasfeo_dmat *sA)
	{
	int ii, jj;
	double tmp, amax = 0.0;
	for(jj=0; jj<n; jj++)
		for(ii=0; ii<m; ii++)
			{
			tmp = fabs(blasfeo_dgeex1(sA, ii, jj));
			amax = tmp>amax ? tmp : amax;
			}
	return amax;
	}



// stability metrics of the LU factorization P*A*Q = L*U of the m x n matrix A, stored in LU
//   bwd     : backward error max|P*A*Q - L*U| / max|A|,
//             with L (U) the first rank columns (rows) of the factors, i.e. including the truncation error if rank < min(m,n)
//   growth  : growth factor max|U| / max|A|
//   lmax    : max|L(i,j)|, i>j (<= 1 for both full and rook pivoting)
//   uratio  : max|U(k,j)| / |U(k,k)|, j>k (<= 1 for both full and rook pivoting)
//   pivratio: min|U(k,k)| / max|U(k,k)|, a cheap estimate of the reciprocal condition number of the factorized part
static void lu_metrics(int m, int n, struct blasfeo_dmat *sA, struct blasfeo_dmat *sLU, int *ipiv, int *jpiv, int rank,
		double *bwd, double *growth, double *lmax, double *uratio, double *pivratio)
	{
	int ii, jj;
	int k = m<n ? m : n;
	double tmp, dmin, dmax;

	struct blasfeo_dmat sL; blasfeo_allocate_dmat(m, k, &sL);
	struct blasfeo_dmat sU; blasfeo_allocate_dmat(k, n, &sU);
	struct blasfeo_dmat sE; blasfeo_allocate_dmat(m, n, &sE);

	for(jj=0; jj<k; jj++)
		for(ii=0; ii<m; ii++)
			blasfeo_dgein1(ii==jj ? 1.0 : (ii>jj & jj<rank ? blasfeo_dgeex1(sLU, ii, jj) : 0.0), &sL, ii, jj);
	for(jj=0; jj<n; jj++)
		for(ii=0; ii<k; ii++)
			blasfeo_dgein1(ii<=jj & ii<rank ? blasfeo_dgeex1(sLU, ii, jj) : 0.0, &sU, ii, jj);

	// E = P*A*Q - L*U
	blasfeo_dgecp(m, n, sA, 0, 0, &sE, 0, 0);
	blasfeo_drowpe(rank, ipiv, n, &sE, 0, 0);
	blasfeo_dcolpe(rank, jpiv, m, &sE, 0, 0);
	blasfeo_dgemm_nn(m, n, k, -1.0, &sL, 0, 0, &sU, 0, 0, 1.0, &sE, 0, 0, &sE, 0, 0);

	double anrm = dmat_amax(m, n, sA);
	*bwd = dmat_amax(m, n, &sE) / anrm;
	*growth = dmat_amax(k, n, &sU) / anrm;

	*lmax = 0.0;
	for(jj=0; jj<rank; jj++)
		for(ii=jj+1; ii<m; ii++)
			{
			tmp = fabs(blasfeo_dgeex1(&sL, ii, jj));
			*lmax = tmp>*lmax ? tmp : *lmax;
			}

	*uratio = 0.0;
	dmin = 0.0;
	dmax = 0.0;
	for(ii=0; ii<rank; ii++)
		{
		tmp = fabs(blasfeo_dgeex1(&sU, ii, ii));
		dmin = ii==0 | tmp<dmin ? tmp : dmin;
		dmax = tmp>dmax ? tmp : dmax;
		for(jj=ii+1; jj<n; jj++)
			{
			tmp = fabs(blasfeo_dgeex1(&sU, ii, jj)) / fabs(blasfeo_dgeex1(&sU, ii, ii));
			*uratio = tmp>*uratio ? tmp : *uratio;
			}
		}
	*pivratio = rank>0 ? dmin / dmax : 0.0;

	blasfeo_free_dmat(&sL);
	blasfeo_free_dmat(&sU);
	blasfeo_free_dmat(&sE);
	}



// accuracy of the solution of A*x = b for a square full rank matrix A, using P*A*Q = L*U:
// x = Q * U^{-1} * L^{-1} * P * b, with b = A * x_true
//   fwd: forward error max|x - x_true| / max|x_true|
//   bwd: normwise backward error ||b - A*x||_inf / (||A||_inf ||x||_inf + ||b||_inf)
static void lu_solve_metrics(int n, struct blasfeo_dmat *sA, struct blasfeo_dmat *sLU, int *ipiv, int *jpiv, double *fwd, double *bwd)
	{
	int ii, jj;
	double tmp, xnrm, bnrm, rnrm, enrm, anrm;

	struct blasfeo_dvec sxt; blasfeo_allocate_dvec(n, &sxt);
	struct blasfeo_dvec sb;  blasfeo_allocate_dvec(n, &sb);
	struct blasfeo_dvec sx;  blasfeo_allocate_dvec(n, &sx);
	struct blasfeo_dvec sy;  blasfeo_allocate_dvec(n, &sy);
	struct blasfeo_dvec sr;  blasfeo_allocate_dvec(n, &sr);

	for(ii=0; ii<n; ii++)
		blasfeo_dvecin1((ii%2 ? -1.0 : 1.0) * (1.0 + ii%5), &sxt, ii);
	blasfeo_dgemv_n(n, n, 1.0, sA, 0, 0, &sxt, 0, 0.0, &sb, 0, &sb, 0);

	// x = Q * U^{-1} * L^{-1} * P * b
	blasfeo_dveccp(n, &sb, 0, &sx, 0);
	blasfeo_dvecpe(n, ipiv, &sx, 0);
	blasfeo_dtrsv_lnu(n, sLU, 0, 0, &sx, 0, &sy, 0);
	blasfeo_dtrsv_unn(n, sLU, 0, 0, &sy, 0, &sx, 0);
	blasfeo_dvecpei(n, jpiv, &sx, 0);

	// r = b - A * x
	blasfeo_dgemv_n(n, n, -1.0, sA, 0, 0, &sx, 0, 1.0, &sb, 0, &sr, 0);

	// e = x - x_true
	blasfeo_daxpy(n, -1.0, &sxt, 0, &sx, 0, &sy, 0);

	blasfeo_dvecnrm_inf(n, &sx, 0, &xnrm);
	blasfeo_dvecnrm_inf(n, &sb, 0, &bnrm);
	blasfeo_dvecnrm_inf(n, &sr, 0, &rnrm);
	blasfeo_dvecnrm_inf(n, &sy, 0, &enrm);
	blasfeo_dvecnrm_inf(n, &sxt, 0, &tmp);
	*fwd = enrm / tmp;

	anrm = 0.0;
	for(ii=0; ii<n; ii++)
		{
		tmp = 0.0;
		for(jj=0; jj<n; jj++)
			tmp += fabs(blasfeo_dgeex1(sA, ii, jj));
		anrm = tmp>anrm ? tmp : anrm;
		}
	*bwd = rnrm / (anrm*xnrm + bnrm);

	blasfeo_free_dvec(&sxt);
	blasfeo_free_dvec(&sb);
	blasfeo_free_dvec(&sx);
	blasfeo_free_dvec(&sy);
	blasfeo_free_dvec(&sr);
	}



// average time of a factorization with full (method 0) or rook (method 1) pivoting,
// repeated until at least 0.1 s have elapsed
static double time_getrf(int method, int m, int n, double tol, struct blasfeo_dmat *sA, struct blasfeo_dmat *sLU, int *ipiv, int *jpiv)
	{
	blasfeo_timer timer;
	int rep, nrep, rank;
	double time;
	for(nrep=1; ; nrep*=2)
		{
		blasfeo_tic(&timer);
		for(rep=0; rep<nrep; rep++)
			{
			if(method==0)
				blasfeo_dgetrf_fp(m, n, tol, sA, 0, 0, sLU, 0, 0, ipiv, jpiv, &rank);
			else
				blasfeo_dgetrf_rk(m, n, tol, sA, 0, 0, sLU, 0, 0, ipiv, jpiv, &rank);
			}
		time = blasfeo_toc(&timer);
		if(time>=0.1)
			return time / nrep;
		}
	}



// factorize the m x n column-major matrix A with full (fp) and rook (rk) pivoting and print the metrics
static void compare_fp_rk(const char *name, int m, int n, double *A, double tol)
	{
	int ii, rank;
	int k = m<n ? m : n;
	double bwd, growth, lmax, uratio, pivratio, sfwd, sbwd;
	double time[2];

	struct blasfeo_dmat sA;  blasfeo_allocate_dmat(m, n, &sA);
	struct blasfeo_dmat sLU; blasfeo_allocate_dmat(m, n, &sLU);
	int *ipiv; int_zeros(&ipiv, k, 1);
	int *jpiv; int_zeros(&jpiv, k, 1);

	blasfeo_pack_dmat(m, n, A, m, &sA, 0, 0);

	// timings
	for(ii=0; ii<2; ii++)
		time[ii] = time_getrf(ii, m, n, tol, &sA, &sLU, ipiv, jpiv);

	for(ii=0; ii<2; ii++)
		{
		if(ii==0)
			blasfeo_dgetrf_fp(m, n, tol, &sA, 0, 0, &sLU, 0, 0, ipiv, jpiv, &rank);
		else
			blasfeo_dgetrf_rk(m, n, tol, &sA, 0, 0, &sLU, 0, 0, ipiv, jpiv, &rank);
		lu_metrics(m, n, &sA, &sLU, ipiv, jpiv, rank, &bwd, &growth, &lmax, &uratio, &pivratio);
		printf("%-18s %4d %4d %3s %5d %10.2e %9.2e %7.3f %8.3f %10.2e", ii==0 ? name : "", m, n, ii==0 ? "fp" : "rk", rank, bwd, growth, lmax, uratio, pivratio);
		if(m==n & rank==n)
			{
			lu_solve_metrics(n, &sA, &sLU, ipiv, jpiv, &sfwd, &sbwd);
			printf(" %10.2e %10.2e", sfwd, sbwd);
			}
		else
			{
			printf(" %10s %10s", "-", "-");
			}
		printf(" %10.2e %8.2f\n", time[ii], time[0]/time[ii]);
		}

	int_free(ipiv);
	int_free(jpiv);
	blasfeo_free_dmat(&sA);
	blasfeo_free_dmat(&sLU);
	}




int main()
	{

	printf("\nExample of LU factorization with complete pivoting of a rectangular matrix\n\n");

#if defined(LA_HIGH_PERFORMANCE)

	printf("\nLA provided by BLASFEO\n\n");

#elif defined(LA_REFERENCE)

	printf("\nLA provided by REFERENCE\n\n");

#elif defined(LA_EXTERNAL_BLAS_WRAPPER)

	printf("\nLA provided by EXTERNAL_BLAS_WRAPPER\n\n");

#else

	printf("\nLA provided by ???\n\n");
	exit(2);

#endif

	int ii, jj;

	int m = 5;
	int n = 7;
	int k = m<n ? m : n;

	//
	// matrix A in column-major format
	//

	double *A; d_zeros(&A, m, n);
	for(jj=0; jj<n; jj++)
		for(ii=0; ii<m; ii++)
			A[ii+m*jj] = (double) ((3*ii + 5*jj + 2*ii*jj) % 13) - 6.0;

	// row and column pivot indices
	int *ipiv; int_zeros(&ipiv, k, 1);
	int *jpiv; int_zeros(&jpiv, k, 1);
	int rank;

	//
	// matrices in matrix struct format
	//

	struct blasfeo_dmat sA;  blasfeo_allocate_dmat(m, n, &sA);  // original matrix
	struct blasfeo_dmat sLU; blasfeo_allocate_dmat(m, n, &sLU); // packed factorization
	struct blasfeo_dmat sP;  blasfeo_allocate_dmat(m, m, &sP);  // row permutation
	struct blasfeo_dmat sQ;  blasfeo_allocate_dmat(n, n, &sQ);  // column permutation
	struct blasfeo_dmat sL;  blasfeo_allocate_dmat(m, k, &sL);  // unit lower triangular
	struct blasfeo_dmat sU;  blasfeo_allocate_dmat(k, n, &sU);  // upper triangular
	struct blasfeo_dmat sPA; blasfeo_allocate_dmat(m, n, &sPA); // P*A
	struct blasfeo_dmat sPAQ; blasfeo_allocate_dmat(m, n, &sPAQ); // P*A*Q
	struct blasfeo_dmat sR;  blasfeo_allocate_dmat(m, n, &sR);  // residual P*A*Q - L*U

	blasfeo_pack_dmat(m, n, A, m, &sA, 0, 0);
	printf("\nA = \n");
	blasfeo_print_dmat(m, n, &sA, 0, 0);

	//
	// LU factorization with complete pivoting: P * A * Q = L * U
	//

	double tol = 0.0; // stop only at an exactly zero pivot
	blasfeo_dgetrf_fp(m, n, tol, &sA, 0, 0, &sLU, 0, 0, ipiv, jpiv, &rank);

	printf("\nrank = %d\n", rank);
	printf("\nLU (packed) = \n");
	blasfeo_print_dmat(m, n, &sLU, 0, 0);
	printf("\nipiv = \n");
	int_print_mat(1, rank, ipiv, 1);
	printf("\njpiv = \n");
	int_print_mat(1, rank, jpiv, 1);

	//
	// build the permutation matrices
	//

	// P = row swaps applied to the m x m identity
	blasfeo_dgese(m, m, 0.0, &sP, 0, 0);
	blasfeo_ddiare(m, 1.0, &sP, 0, 0);
	blasfeo_drowpe(rank, ipiv, m, &sP, 0, 0);
	printf("\nP = \n");
	blasfeo_print_dmat(m, m, &sP, 0, 0);

	// Q = column swaps applied to the n x n identity
	blasfeo_dgese(n, n, 0.0, &sQ, 0, 0);
	blasfeo_ddiare(n, 1.0, &sQ, 0, 0);
	blasfeo_dcolpe(rank, jpiv, n, &sQ, 0, 0);
	printf("\nQ = \n");
	blasfeo_print_dmat(n, n, &sQ, 0, 0);

	//
	// extract L (m x k, unit lower triangular) and U (k x n, upper triangular)
	// if rank < k, the factorization stopped early: the columns of L past rank are set
	// to the identity and the rows of U past rank to zero, so P*A*Q - L*U equals the
	// trailing (m-rank) x (n-rank) Schur complement, whose entries are all <= tol
	//

	for(jj=0; jj<k; jj++)
		for(ii=0; ii<m; ii++)
			blasfeo_dgein1(ii==jj ? 1.0 : (ii>jj & jj<rank ? blasfeo_dgeex1(&sLU, ii, jj) : 0.0), &sL, ii, jj);
	printf("\nL = \n");
	blasfeo_print_dmat(m, k, &sL, 0, 0);

	for(jj=0; jj<n; jj++)
		for(ii=0; ii<k; ii++)
			blasfeo_dgein1(ii<=jj & ii<rank ? blasfeo_dgeex1(&sLU, ii, jj) : 0.0, &sU, ii, jj);
	printf("\nU = \n");
	blasfeo_print_dmat(k, n, &sU, 0, 0);

	//
	// check: P * A * Q - L * U
	//

	blasfeo_dgemm_nn(m, n, m, 1.0, &sP, 0, 0, &sA, 0, 0, 0.0, &sPA, 0, 0, &sPA, 0, 0);
	blasfeo_dgemm_nn(m, n, n, 1.0, &sPA, 0, 0, &sQ, 0, 0, 0.0, &sPAQ, 0, 0, &sPAQ, 0, 0);
	printf("\nP*A*Q = \n");
	blasfeo_print_dmat(m, n, &sPAQ, 0, 0);

	blasfeo_dgemm_nn(m, n, k, -1.0, &sL, 0, 0, &sU, 0, 0, 1.0, &sPAQ, 0, 0, &sR, 0, 0);
	printf("\nP*A*Q - L*U = \n");
	blasfeo_print_exp_dmat(m, n, &sR, 0, 0);

	double res = 0.0;
	for(jj=0; jj<n; jj++)
		for(ii=0; ii<m; ii++)
			res = fmax(res, fabs(blasfeo_dgeex1(&sR, ii, jj)));
	printf("\nmax |P*A*Q - L*U| = %e\n\n", res);

	//
	// stability and accuracy: full (fp) vs. rook (rk) pivoting
	//

	printf("\nStability and accuracy of full (fp) and rook (rk) pivoting:\n");
	printf("  bwd err  : max|P*A*Q - L*U| / max|A|\n");
	printf("  growth   : max|U| / max|A|\n");
	printf("  max|L|   : max|L(i,j)|, i>j\n");
	printf("  max|Ukj| : max|U(k,j)| / |U(k,k)|, j>k\n");
	printf("  piv ratio: min|U(k,k)| / max|U(k,k)|\n");
	printf("  solve    : forward error and normwise backward error of A*x = b (square full rank matrices only)\n");
	printf("  time     : average time of the factorization [s]\n");
	printf("  speed-up : time of fp / time of the method\n\n");
	printf("%-18s %4s %4s %3s %5s %10s %9s %7s %8s %10s %10s %10s %10s %8s\n", "matrix", "m", "n", "", "rank", "bwd err", "growth", "max|L|", "max|Ukj|", "piv ratio", "solve fwd", "solve bwd", "time", "speed-up");

	double *B;
	int mb, nb, rb, ll;

	// the example above
	compare_fp_rk("example", m, n, A, tol);

	// random rectangular matrix
	mb = 200; nb = 280;
	d_zeros(&B, mb, nb);
	for(ii=0; ii<mb*nb; ii++)
		B[ii] = (double) rand() / RAND_MAX - 0.5;
	compare_fp_rk("random", mb, nb, B, 0.0);
	d_free(B);

	// random square matrix
	mb = 200; nb = 200;
	d_zeros(&B, mb, nb);
	for(ii=0; ii<mb*nb; ii++)
		B[ii] = (double) rand() / RAND_MAX - 0.5;
	compare_fp_rk("random square", mb, nb, B, 0.0);
	d_free(B);

	// random rank deficient matrix (rank 40) = X * Y, factorized with a tolerance
	mb = 100; nb = 140; rb = 40;
	double *X; d_zeros(&X, mb, rb);
	double *Y; d_zeros(&Y, rb, nb);
	for(ii=0; ii<mb*rb; ii++)
		X[ii] = (double) rand() / RAND_MAX - 0.5;
	for(ii=0; ii<rb*nb; ii++)
		Y[ii] = (double) rand() / RAND_MAX - 0.5;
	d_zeros(&B, mb, nb);
	for(jj=0; jj<nb; jj++)
		for(ii=0; ii<mb; ii++)
			for(ll=0; ll<rb; ll++)
				B[ii+mb*jj] += X[ii+mb*ll] * Y[ll+rb*jj];
	compare_fp_rk("rank 40, tol 1e-10", mb, nb, B, 1e-10);
	d_free(X);
	d_free(Y);
	d_free(B);

	// Wilkinson matrix: 1 on the diagonal and in the last column, -1 below the diagonal;
	// partial pivoting has growth factor 2^(n-1) on it
	mb = 60; nb = 60;
	d_zeros(&B, mb, nb);
	for(jj=0; jj<nb; jj++)
		for(ii=0; ii<mb; ii++)
			B[ii+mb*jj] = ii==jj | jj==nb-1 ? 1.0 : (ii>jj ? -1.0 : 0.0);
	compare_fp_rk("Wilkinson", mb, nb, B, 0.0);
	d_free(B);

	// Hilbert matrix, condition number about 1.6e13:
	// small backward error, but large forward error
	mb = 10; nb = 10;
	d_zeros(&B, mb, nb);
	for(jj=0; jj<nb; jj++)
		for(ii=0; ii<mb; ii++)
			B[ii+mb*jj] = 1.0 / (ii + jj + 1);
	compare_fp_rk("Hilbert", mb, nb, B, 0.0);
	d_free(B);

	printf("\n");

	//
	// timings
	//

	blasfeo_timer timer;
	int rep, nrep;
	double time_fp;

	nrep = 100000;

	blasfeo_tic(&timer);
	for(rep=0; rep<nrep; rep++)
		blasfeo_dgetrf_fp(m, n, tol, &sA, 0, 0, &sLU, 0, 0, ipiv, jpiv, &rank);
	time_fp = blasfeo_toc(&timer) / nrep;

	printf("\nTimings for the %d x %d example (average over %d runs):\n", m, n, nrep);
	printf("  getrf_fp: %e s\n", time_fp);

	// sweep over larger rectangular sizes (m x n, with n = 7/5 m) and random entries
	printf("\nTimings for random m x n matrices (n = 7/5 m):\n\n");
	printf("%6s %6s %14s %10s\n", "m", "n", "time [s]", "Gflops");

	int sizes[] = {4, 8, 16, 32, 64, 128, 256};
	int nsizes = sizeof(sizes)/sizeof(sizes[0]);
	int is, mt, nt, kt, rank_t;
	double flops, tmp;
	for(is=0; is<nsizes; is++)
		{
		mt = sizes[is];
		nt = (7*mt)/5;
		kt = mt<nt ? mt : nt;

		struct blasfeo_dmat sAt;  blasfeo_allocate_dmat(mt, nt, &sAt);
		struct blasfeo_dmat sLUt; blasfeo_allocate_dmat(mt, nt, &sLUt);
		int *ipiv_t; int_zeros(&ipiv_t, kt, 1);
		int *jpiv_t; int_zeros(&jpiv_t, kt, 1);

		for(jj=0; jj<nt; jj++)
			for(ii=0; ii<mt; ii++)
				blasfeo_dgein1((double) rand() / RAND_MAX - 0.5, &sAt, ii, jj);

		// flop count of an m x n LU factorization with m <= n
		flops = (double) mt*mt*nt - (double) mt*mt*mt/3.0;
		// aim at roughly 1e9 flops per measurement, with at least 10 runs
		tmp = 1e9 / flops;
		nrep = tmp<10 ? 10 : (int) tmp;

		blasfeo_tic(&timer);
		for(rep=0; rep<nrep; rep++)
			blasfeo_dgetrf_fp(mt, nt, 0.0, &sAt, 0, 0, &sLUt, 0, 0, ipiv_t, jpiv_t, &rank_t);
		time_fp = blasfeo_toc(&timer) / nrep;

		printf("%6d %6d %14e %10.3f\n", mt, nt, time_fp, 1e-9*flops/time_fp);

		int_free(ipiv_t);
		int_free(jpiv_t);
		blasfeo_free_dmat(&sAt);
		blasfeo_free_dmat(&sLUt);
		}
	printf("\n");

	//
	// free memory
	//

	d_free(A);
	int_free(ipiv);
	int_free(jpiv);
	blasfeo_free_dmat(&sA);
	blasfeo_free_dmat(&sLU);
	blasfeo_free_dmat(&sP);
	blasfeo_free_dmat(&sQ);
	blasfeo_free_dmat(&sL);
	blasfeo_free_dmat(&sU);
	blasfeo_free_dmat(&sPA);
	blasfeo_free_dmat(&sPAQ);
	blasfeo_free_dmat(&sR);

	return 0;

	}
