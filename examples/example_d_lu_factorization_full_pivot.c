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



// the three LU factorizations compared in this example
#define N_METHODS 3
static const char *method_name[N_METHODS] = {"fp", "fp_opt", "rk"};
static const char *method_desc[N_METHODS] = {
	"blasfeo_dgetrf_fp           : complete pivoting, original implementation",
	"blasfeo_dgetrf_fp_optimized : complete pivoting, optimized implementation (same pivots)",
	"blasfeo_dgetrf_rk           : rook pivoting"};

static void getrf_method(int method, int m, int n, double tol, struct blasfeo_dmat *sA, struct blasfeo_dmat *sLU, int *ipiv, int *jpiv, int *rank)
	{
	if(method==0)
		blasfeo_dgetrf_fp(m, n, tol, sA, 0, 0, sLU, 0, 0, ipiv, jpiv, rank);
	else if(method==1)
		blasfeo_dgetrf_fp_optimized(m, n, tol, sA, 0, 0, sLU, 0, 0, ipiv, jpiv, rank);
	else
		blasfeo_dgetrf_rk(m, n, tol, sA, 0, 0, sLU, 0, 0, ipiv, jpiv, rank);
	}



// average time of a factorization, repeated until at least 0.05 s have elapsed; best of three
static double time_getrf(int method, int m, int n, double tol, struct blasfeo_dmat *sA, struct blasfeo_dmat *sLU, int *ipiv, int *jpiv)
	{
	blasfeo_timer timer;
	int rep, nrep, rank, ii;
	double time, best = 1e30;
	for(nrep=1; ; nrep*=2)
		{
		blasfeo_tic(&timer);
		for(rep=0; rep<nrep; rep++)
			getrf_method(method, m, n, tol, sA, sLU, ipiv, jpiv, &rank);
		if(blasfeo_toc(&timer)>=0.05)
			break;
		}
	for(ii=0; ii<3; ii++)
		{
		blasfeo_tic(&timer);
		for(rep=0; rep<nrep; rep++)
			getrf_method(method, m, n, tol, sA, sLU, ipiv, jpiv, &rank);
		time = blasfeo_toc(&timer) / nrep;
		best = time<best ? time : best;
		}
	return best;
	}



// metrics of the factorization P*A*Q = L*U + [0 0; 0 S] of the m x n matrix A, stored in LU, with
// L (U) the first rank columns (rows) of the factors and S the trailing (m-rank) x (n-rank) block:
//   bwd      : backward error max|P*A*Q - L*U - [0 0; 0 S]| / max|A|
//   growth   : growth factor max|U| / max|A|
//   lmax     : max|L(i,j)|, i>j (<= 1 for complete and rook pivoting)
//   uratio   : max|U(k,j)| / |U(k,k)|, j>k (<= 1 for complete and rook pivoting)
//   pmin     : smallest pivot min|U(k,k)|
//   smax     : max|S| (<= tol if the factorization stopped before min(m,n))
//   nonfinite: number of Inf or NaN entries in LU
struct lu_metrics { double bwd, growth, lmax, uratio, pmin, smax; int nonfinite; };

static void lu_check(int m, int n, struct blasfeo_dmat *sA, struct blasfeo_dmat *sLU, int *ipiv, int *jpiv, int rank, struct lu_metrics *out)
	{
	int ii, jj, ll, lmax;
	double s, e, a, anrm = 0.0;
	double *LU = malloc(m*n*sizeof(double));
	double *PAQ = malloc(m*n*sizeof(double));
	struct blasfeo_dmat sB; blasfeo_allocate_dmat(m, n, &sB);
	blasfeo_dgecp(m, n, sA, 0, 0, &sB, 0, 0);
	blasfeo_drowpe(rank, ipiv, n, &sB, 0, 0);
	blasfeo_dcolpe(rank, jpiv, m, &sB, 0, 0);
	blasfeo_unpack_dmat(m, n, &sB, 0, 0, PAQ, m);
	blasfeo_unpack_dmat(m, n, sLU, 0, 0, LU, m);
	out->bwd = 0.0; out->growth = 0.0; out->lmax = 0.0; out->uratio = 0.0; out->pmin = rank>0 ? INFINITY : 0.0; out->smax = 0.0; out->nonfinite = 0;
	for(ii=0; ii<m*n; ii++)
		{
		anrm = fmax(anrm, fabs(PAQ[ii]));
		out->nonfinite += !isfinite(LU[ii]);
		}
	for(jj=0; jj<n; jj++)
		{
		for(ii=0; ii<m; ii++)
			{
			s = 0.0;
			lmax = ii<jj ? ii : jj;
			lmax = lmax<rank-1 ? lmax : rank-1;
			for(ll=0; ll<=lmax; ll++)
				s += (ll==ii ? 1.0 : LU[ii+m*ll]) * LU[ll+m*jj];
			if(ii>=rank & jj>=rank)
				{
				s += LU[ii+m*jj];
				out->smax = fmax(out->smax, fabs(LU[ii+m*jj]));
				}
			e = fabs(PAQ[ii+m*jj] - s);
			out->bwd = e>out->bwd | e!=e ? e : out->bwd;
			if(ii<rank & ii<=jj)
				{
				a = fabs(LU[ii+m*jj]);
				out->growth = fmax(out->growth, a);
				if(ii==jj)
					out->pmin = fmin(out->pmin, a);
				else
					out->uratio = fmax(out->uratio, a / fabs(LU[ii+m*ii]));
				}
			if(jj<rank & ii>jj)
				out->lmax = fmax(out->lmax, fabs(LU[ii+m*jj]));
			}
		}
	out->bwd /= anrm;
	out->growth /= anrm;
	free(LU);
	free(PAQ);
	blasfeo_free_dmat(&sB);
	}



// 1 if the two factorizations are bitwise identical (LU, pivots and rank)
static int same_lu(int m, int n, struct blasfeo_dmat *sLU0, int *ipiv0, int *jpiv0, int rank0, struct blasfeo_dmat *sLU1, int *ipiv1, int *jpiv1, int rank1)
	{
	int ii, jj;
	double x0, x1;
	if(rank0!=rank1)
		return 0;
	for(ii=0; ii<rank0; ii++)
		if(ipiv0[ii]!=ipiv1[ii] | jpiv0[ii]!=jpiv1[ii])
			return 0;
	for(jj=0; jj<n; jj++)
		for(ii=0; ii<m; ii++)
			{
			x0 = blasfeo_dgeex1(sLU0, ii, jj);
			x1 = blasfeo_dgeex1(sLU1, ii, jj);
			if(!(x0==x1) & !(x0!=x0 & x1!=x1))
				return 0;
			}
	return 1;
	}



// singular values (in decreasing order) of the m x n column-major matrix A, by one-sided Jacobi
static void svd_values(int m, int n, double *A, double *sigma)
	{
	int mb = m>=n ? m : n;
	int nb = m>=n ? n : m;
	int ii, jj, pp, qq, sweep, rot;
	double alpha, beta, gamma, zeta, t, c, s, bp, bq;
	double *B = malloc(mb*nb*sizeof(double));
	for(jj=0; jj<nb; jj++)
		for(ii=0; ii<mb; ii++)
			B[ii+mb*jj] = m>=n ? A[ii+m*jj] : A[jj+m*ii];
	for(sweep=0; sweep<100; sweep++)
		{
		rot = 0;
		for(pp=0; pp<nb-1; pp++)
			{
			for(qq=pp+1; qq<nb; qq++)
				{
				alpha = 0.0; beta = 0.0; gamma = 0.0;
				for(ii=0; ii<mb; ii++)
					{
					alpha += B[ii+mb*pp] * B[ii+mb*pp];
					beta += B[ii+mb*qq] * B[ii+mb*qq];
					gamma += B[ii+mb*pp] * B[ii+mb*qq];
					}
				if(gamma==0.0 || fabs(gamma)<=1e-15*sqrt(alpha*beta))
					continue;
				rot = 1;
				zeta = (beta - alpha) / (2.0*gamma);
				t = (zeta>=0.0 ? 1.0 : -1.0) / (fabs(zeta) + sqrt(1.0+zeta*zeta));
				c = 1.0 / sqrt(1.0+t*t);
				s = c * t;
				for(ii=0; ii<mb; ii++)
					{
					bp = B[ii+mb*pp];
					bq = B[ii+mb*qq];
					B[ii+mb*pp] = c*bp - s*bq;
					B[ii+mb*qq] = s*bp + c*bq;
					}
				}
			}
		if(!rot)
			break;
		}
	for(jj=0; jj<nb; jj++)
		{
		s = 0.0;
		for(ii=0; ii<mb; ii++)
			s += B[ii+mb*jj] * B[ii+mb*jj];
		sigma[jj] = sqrt(s);
		}
	// sort in decreasing order
	for(pp=1; pp<nb; pp++)
		for(qq=pp; qq>0 && sigma[qq]>sigma[qq-1]; qq--)
			{
			t = sigma[qq]; sigma[qq] = sigma[qq-1]; sigma[qq-1] = t;
			}
	free(B);
	}



static double rnd()
	{
	return (double) rand() / RAND_MAX - 0.5;
	}

static double amax(int m, int n, double *A)
	{
	int ii;
	double a = 0.0;
	for(ii=0; ii<m*n; ii++)
		a = fmax(a, fabs(A[ii]));
	return a;
	}

// A = X * diag(s) * Y, with X (m x r) and Y (r x n) random: rank r, singular values roughly proportional to s
static double *low_rank(int m, int n, int r, double *s)
	{
	int ii, jj, ll;
	double *A = calloc(m*n, sizeof(double));
	double *X = malloc(m*sizeof(double));
	double *Y = malloc(n*sizeof(double));
	for(ll=0; ll<r; ll++)
		{
		for(ii=0; ii<m; ii++)
			X[ii] = rnd();
		for(jj=0; jj<n; jj++)
			Y[jj] = rnd();
		for(jj=0; jj<n; jj++)
			for(ii=0; ii<m; ii++)
				A[ii+m*jj] += (s!=NULL ? s[ll] : 1.0) * X[ii] * Y[jj];
		}
	free(X);
	free(Y);
	return A;
	}



// rank revealing: compare the rank of the three methods with the numerical rank from the singular values
static void rank_test(const char *name, int m, int n, double *A, double tol)
	{
	int ii, method, rank[N_METHODS];
	int k = m<n ? m : n;
	double *sigma = malloc(k*sizeof(double));
	struct blasfeo_dmat sA; blasfeo_allocate_dmat(m, n, &sA);
	struct blasfeo_dmat sLU[N_METHODS];
	int *ipiv[N_METHODS], *jpiv[N_METHODS];
	struct lu_metrics met;

	blasfeo_pack_dmat(m, n, A, m, &sA, 0, 0);
	svd_values(m, n, A, sigma);
	int rsvd = 0;
	for(ii=0; ii<k; ii++)
		rsvd += sigma[ii]>tol;

	printf("%-26s %4d %4d  tol %8.1e  svd: rank %3d, sigma_r %8.1e, sigma_r+1 %8.1e\n", name, m, n, tol, rsvd, rsvd>0 ? sigma[rsvd-1] : 0.0, rsvd<k ? sigma[rsvd] : 0.0);
	for(method=0; method<N_METHODS; method++)
		{
		blasfeo_allocate_dmat(m, n, &sLU[method]);
		ipiv[method] = malloc(k*sizeof(int));
		jpiv[method] = malloc(k*sizeof(int));
		getrf_method(method, m, n, tol, &sA, &sLU[method], ipiv[method], jpiv[method], &rank[method]);
		lu_check(m, n, &sA, &sLU[method], ipiv[method], jpiv[method], rank[method], &met);
		printf("%28s %-7s rank %3d %-9s smallest pivot %8.1e  max|S| %8.1e  bwd err %8.1e", "", method_name[method], rank[method], rank[method]==rsvd ? "(= svd)" : "(!= svd)", met.pmin, met.smax, met.bwd);
		if(method==1)
			printf("  %s", same_lu(m, n, &sLU[0], ipiv[0], jpiv[0], rank[0], &sLU[1], ipiv[1], jpiv[1], rank[1]) ? "identical to fp" : "DIFFERENT from fp");
		printf("\n");
		}

	for(method=0; method<N_METHODS; method++)
		{
		blasfeo_free_dmat(&sLU[method]);
		free(ipiv[method]);
		free(jpiv[method]);
		}
	blasfeo_free_dmat(&sA);
	free(sigma);
	}



// stability: backward error, growth and bounds on the factors for the three methods
static void stability_test(const char *name, int m, int n, double *A, double tol)
	{
	int method, rank[N_METHODS];
	int k = m<n ? m : n;
	struct blasfeo_dmat sA; blasfeo_allocate_dmat(m, n, &sA);
	struct blasfeo_dmat sLU[N_METHODS];
	int *ipiv[N_METHODS], *jpiv[N_METHODS];
	struct lu_metrics met;

	blasfeo_pack_dmat(m, n, A, m, &sA, 0, 0);
	for(method=0; method<N_METHODS; method++)
		{
		blasfeo_allocate_dmat(m, n, &sLU[method]);
		ipiv[method] = malloc(k*sizeof(int));
		jpiv[method] = malloc(k*sizeof(int));
		getrf_method(method, m, n, tol, &sA, &sLU[method], ipiv[method], jpiv[method], &rank[method]);
		lu_check(m, n, &sA, &sLU[method], ipiv[method], jpiv[method], rank[method], &met);
		printf("%-26s %4d %4d %-7s %4d %9.1e %9.1e %7.3f %8.3f %9.1e %6d", method==0 ? name : "", m, n, method_name[method], rank[method], met.bwd, met.growth, met.lmax, met.uratio, met.pmin, met.nonfinite);
		if(method==1)
			printf("  %s", same_lu(m, n, &sLU[0], ipiv[0], jpiv[0], rank[0], &sLU[1], ipiv[1], jpiv[1], rank[1]) ? "identical to fp" : "DIFFERENT from fp");
		printf("\n");
		}

	for(method=0; method<N_METHODS; method++)
		{
		blasfeo_free_dmat(&sLU[method]);
		free(ipiv[method]);
		free(jpiv[method]);
		}
	blasfeo_free_dmat(&sA);
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
	// comparison of complete pivoting (original and optimized) and rook pivoting on rectangular matrices
	//

	printf("\n\nComparison of the LU factorizations with:\n");
	for(ii=0; ii<N_METHODS; ii++)
		printf("  %s\n", method_desc[ii]);

	double *B;
	int mb, nb, kb, ll, method, rank_t;
	srand(1);

	//
	// 1. timing
	//

	printf("\n\n1. Timing [s] (best of three averages); speed-up = time of fp / time of the method\n\n");
	printf("%-22s %4s %4s %10s %10s %10s %10s %10s\n", "matrix", "m", "n", "fp", "fp_opt", "rk", "fp_opt su", "rk su");
	int tsizes[][2] = {{4,6}, {6,4}, {8,12}, {12,8}, {16,24}, {24,16}, {32,48}, {48,32}, {64,96}, {96,64}, {128,192}, {192,128}, {256,384}, {384,256}};
	int nts = sizeof(tsizes)/sizeof(tsizes[0]);
	int kind;
	for(kind=0; kind<3; kind++)
		{
		for(ll=0; ll<nts; ll++)
			{
			mb = tsizes[ll][0];
			nb = tsizes[ll][1];
			if(kind==2 & mb<64)
				continue;
			kb = mb<nb ? mb : nb;
			double tol_t = 0.0;
			if(kind==2)
				{
				// rank deficient: rank min(m,n)/8, the factorization stops early
				B = low_rank(mb, nb, kb/8, NULL);
				tol_t = 1e-10 * amax(mb, nb, B);
				}
			else
				{
				d_zeros(&B, mb, nb);
				for(ii=0; ii<mb*nb; ii++)
					B[ii] = kind==0 ? rnd() : (rand()%2 ? 1.0 : -1.0);
				}
			struct blasfeo_dmat sAt;  blasfeo_allocate_dmat(mb, nb, &sAt);
			struct blasfeo_dmat sLUt; blasfeo_allocate_dmat(mb, nb, &sLUt);
			int *ipiv_t; int_zeros(&ipiv_t, kb, 1);
			int *jpiv_t; int_zeros(&jpiv_t, kb, 1);
			blasfeo_pack_dmat(mb, nb, B, mb, &sAt, 0, 0);
			double time_t[N_METHODS];
			for(method=0; method<N_METHODS; method++)
				time_t[method] = time_getrf(method, mb, nb, tol_t, &sAt, &sLUt, ipiv_t, jpiv_t);
			char label[64];
			if(kind==2)
				{
				blasfeo_dgetrf_rk(mb, nb, tol_t, &sAt, 0, 0, &sLUt, 0, 0, ipiv_t, jpiv_t, &rank_t);
				sprintf(label, "rank %d (tol 1e-10)", rank_t);
				}
			else
				{
				sprintf(label, kind==0 ? "random" : "random +-1 (ties)");
				}
			printf("%-22s %4d %4d %10.3e %10.3e %10.3e %10.2f %10.2f\n", label, mb, nb, time_t[0], time_t[1], time_t[2], time_t[0]/time_t[1], time_t[0]/time_t[2]);
			int_free(ipiv_t);
			int_free(jpiv_t);
			blasfeo_free_dmat(&sAt);
			blasfeo_free_dmat(&sLUt);
			if(kind==2)
				free(B);
			else
				d_free(B);
			}
		printf("\n");
		}

	//
	// 2. rank revealing
	//

	printf("\n2. Rank revealing: rank of the factorization (stop when all the entries of the trailing block S\n");
	printf("   are <= tol) vs. numerical rank from the singular values (number of sigma > tol)\n\n");

	// full rank
	d_zeros(&B, 40, 60);
	for(ii=0; ii<40*60; ii++)
		B[ii] = rnd();
	rank_test("random (full rank)", 40, 60, B, 1e-10*amax(40, 60, B));
	d_free(B);

	// rank deficient, exact rank
	B = low_rank(60, 40, 25, NULL);
	rank_test("rank 25", 60, 40, B, 1e-10*amax(60, 40, B));
	free(B);
	B = low_rank(60, 80, 10, NULL);
	rank_test("rank 10", 60, 80, B, 1e-10*amax(60, 80, B));
	free(B);
	B = low_rank(80, 60, 1, NULL);
	rank_test("rank 1", 80, 60, B, 1e-10*amax(80, 60, B));
	free(B);

	// rank 15 with 10 zero rows and 10 zero columns interleaved
	B = low_rank(50, 60, 15, NULL);
	for(ii=0; ii<10; ii++)
		{
		for(jj=0; jj<60; jj++)
			B[(5*ii+2)+50*jj] = 0.0;
		for(jj=0; jj<50; jj++)
			B[jj+50*(6*ii+1)] = 0.0;
		}
	rank_test("rank 15, zero rows/cols", 50, 60, B, 1e-10*amax(50, 60, B));
	free(B);

	// numerically rank deficient: rank 20 plus noise of size 1e-12
	B = low_rank(50, 70, 20, NULL);
	for(ii=0; ii<50*70; ii++)
		B[ii] += 1e-12 * rnd();
	rank_test("rank 20 + 1e-12 noise", 50, 70, B, 1e-9*amax(50, 70, B));
	free(B);

	// rank 30 with singular values spanning 1e+5 ... 1e-2, and 10 more of size ~1e-13 below the gap
	double sv[40];
	for(ii=0; ii<40; ii++)
		sv[ii] = ii<30 ? pow(10.0, 5.0-7.0*ii/29) : 1e-13;
	B = low_rank(60, 80, 40, sv);
	rank_test("rank 30 (1e5..1e-2), gap", 60, 80, B, 1e-9*amax(60, 80, B));
	free(B);

	// graded singular values without gap: the numerical rank is not well defined
	for(ii=0; ii<40; ii++)
		sv[ii] = pow(10.0, -14.0*ii/39);
	B = low_rank(60, 80, 40, sv);
	rank_test("graded 1..1e-14 (no gap)", 60, 80, B, 1e-8*amax(60, 80, B));
	free(B);

	// Kahan matrix: the classic case where pivoting does not reveal the rank
	// (the pivots are the diagonal s^i, but the smallest singular value is much smaller)
	mb = 80; nb = 60;
	double c_k = 0.285, s_k = sqrt(1.0-c_k*c_k);
	B = calloc(mb*nb, sizeof(double));
	for(ii=0; ii<nb; ii++)
		for(jj=ii; jj<nb; jj++)
			B[ii+mb*jj] = pow(s_k, ii) * (ii==jj ? 1.0 : -c_k) * (ii==jj ? (1.0-1e-10*ii) : 1.0);
	double *sk = malloc(nb*sizeof(double));
	svd_values(mb, nb, B, sk);
	// tolerance between the smallest singular value and the smallest pivot
	rank_test("Kahan 60 (+20 zero rows)", mb, nb, B, sqrt(sk[nb-1]*pow(s_k, nb-1)));
	free(sk);
	free(B);

	//
	// 3. numerical stability on challenging matrices
	//

	printf("\n\n3. Numerical stability (tol = 0 unless stated)\n");
	printf("   bwd err: max|P*A*Q - L*U - [0 0; 0 S]| / max|A|, growth: max|U| / max|A|, max|L|: max|L(i,j)|, i>j,\n");
	printf("   max|Ukj|: max|U(k,j)| / |U(k,k)|, j>k, min piv: smallest pivot, nonfin: number of Inf/NaN in LU\n\n");
	printf("%-26s %4s %4s %-7s %4s %9s %9s %7s %8s %9s %6s\n", "matrix", "m", "n", "method", "rank", "bwd err", "growth", "max|L|", "max|Ukj|", "min piv", "nonfin");

	// Wilkinson matrix (extended to rectangular): growth 2^(m-1) for partial pivoting
	mb = 60; nb = 80;
	d_zeros(&B, mb, nb);
	for(jj=0; jj<nb; jj++)
		for(ii=0; ii<mb; ii++)
			B[ii+mb*jj] = ii==jj | jj>=mb-1 ? 1.0 : (ii>jj ? -1.0 : 0.0);
	stability_test("Wilkinson", mb, nb, B, 0.0);
	d_free(B);

	// badly scaled rows and columns: D1 * A * D2, with D1, D2 spanning 1e-8 ... 1e8
	mb = 60; nb = 80;
	d_zeros(&B, mb, nb);
	for(jj=0; jj<nb; jj++)
		for(ii=0; ii<mb; ii++)
			B[ii+mb*jj] = pow(10.0, -8.0+16.0*ii/(mb-1)) * rnd() * pow(10.0, 8.0-16.0*jj/(nb-1));
	stability_test("scaled rows/cols 1e+-8", mb, nb, B, 0.0);
	d_free(B);

	// entries with random magnitude 1e-150 ... 1e150
	mb = 40; nb = 60;
	d_zeros(&B, mb, nb);
	for(ii=0; ii<mb*nb; ii++)
		B[ii] = rnd() * pow(10.0, 300.0*(rnd()));
	stability_test("entries 1e-150..1e150", mb, nb, B, 0.0);
	d_free(B);

	// Hilbert matrix (rectangular), very ill conditioned
	mb = 12; nb = 16;
	d_zeros(&B, mb, nb);
	for(jj=0; jj<nb; jj++)
		for(ii=0; ii<mb; ii++)
			B[ii+mb*jj] = 1.0 / (ii + jj + 1);
	stability_test("Hilbert", mb, nb, B, 0.0);
	d_free(B);

	// Vandermonde matrix, nodes in [0, 1]
	mb = 20; nb = 30;
	d_zeros(&B, mb, nb);
	for(jj=0; jj<nb; jj++)
		for(ii=0; ii<mb; ii++)
			B[ii+mb*jj] = pow((double) jj / (nb-1), ii);
	stability_test("Vandermonde", mb, nb, B, 0.0);
	d_free(B);

	// entries +-1: many ties in the pivot search
	mb = 100; nb = 150;
	d_zeros(&B, mb, nb);
	for(ii=0; ii<mb*nb; ii++)
		B[ii] = rand()%2 ? 1.0 : -1.0;
	stability_test("random +-1 (ties)", mb, nb, B, 0.0);
	d_free(B);

	// rank 5 with tol = 0: the trailing block is at the level of the rounding errors,
	// the factorization continues on pivots of size ~1e-17
	B = low_rank(60, 80, 5, NULL);
	stability_test("rank 5, tol 0", 60, 80, B, 0.0);
	free(B);

	// rank 3 scaled by 1e-310 (denormal entries): the reciprocal of the pivots overflows
	B = low_rank(30, 40, 3, NULL);
	for(ii=0; ii<30*40; ii++)
		B[ii] *= 1e-310;
	stability_test("rank 3 x 1e-310, tol 0", 30, 40, B, 0.0);
	free(B);

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
