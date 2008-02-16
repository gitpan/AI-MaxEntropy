/**************************************************************************
 * XS of AI:MaxEntropy
 * -> by Laye Suen
 **************************************************************************/

#include "EXTERN.h"
#include "perl.h"
#include "XSUB.h"

#include "ppport.h"


/* Macros for debugging */

/* uncomment the line below to enable tracing and timing */
/* #define __ENABLE_TRACING__ */

#ifdef __ENABLE_TRACING__

#include "time.h"

#define TRACE(msg) \
    printf(msg); \
    printf(": %0.10f s\n", 1.0 * (clock() - _c) / CLOCKS_PER_SEC); \
    fflush(stdout); \
    _c = clock()
#define dTRACE clock_t _c = clock()

#else

#define TRACE(msg)
#define dTRACE

#endif

/* Other macros */

#define hvref_fetch(hvref, key) \
    hv_fetch((HV*)SvRV(hvref), key, strlen(key), 0)
#define hvref_exists(hvref, key) \
    hv_exists((HV*)SvRV(hvref), key, strlen(key))

/**************************************************************************
 * EXPORTED XSUBS
 **************************************************************************/
MODULE = AI::MaxEntropy		PACKAGE = AI::MaxEntropy

void
_neg_log_likelihood(lambda, step, self, OUTLIST SV* f, OUTLIST SV* g)
        AV*     lambda
	SV*     step
	SV*     self
    PREINIT:
	AV *samples = (AV*)SvRV(*hvref_fetch(self, "samples"));
	AV *sample, *x, *av_d_log_lh;
	SV *smoother;
	char* smoother_type;
	int i, j, y, x_len, x1, y1, fxy;
	int s_num = av_len(samples) + 1;
        int x_num = SvIV(*hvref_fetch(self, "x_num"));
	int y_num = SvIV(*hvref_fetch(self, "y_num"));
	int f_num = SvIV(*hvref_fetch(self, "f_num"));
        double log_lh, sum_exp_lambda_f, w, sigma, l;
	double *lambda_f = (double*)malloc(sizeof(double) * y_num);
	double *d_log_lh = (double*)malloc(sizeof(double) * f_num);
	double **carr_lambda;
	int *carr_x;
	dTRACE;
    CODE:
        /* initialize */
	log_lh = 0;	
	TRACE("_neg_log_likelihood: enter");
	carr_lambda = (double**)malloc(sizeof(double*) * y_num);
	for (i = 0; i < y_num; i++)
	    carr_lambda[i] = (double*)malloc(sizeof(double) * x_num);
	for (y1 = 0; y1 < y_num; y1++)
	    for (x1 = 0; x1 < x_num; x1++)
	        carr_lambda[y1][x1] = 
		    SvNV(*av_fetch(lambda, x1 + y1 * x_num, 0));
	for (i = 0; i < f_num; i++) d_log_lh[i] = 0;
	TRACE("_neg_log_likelihood: finish initialize carr_lambda");
	/* calculate log likelihood and its gradient */
        for (i = 0; i < s_num; i++) {
	    /* get a sample -> x, y, w */
	    sample = (AV*)SvRV(*av_fetch(samples, i, 0));
	    x = (AV*)SvRV(*av_fetch(sample, 0, 0));
	    x_len = av_len(x) + 1;
	    carr_x = (int*)malloc(sizeof(int) * x_len);
	    for (j = 0; j < x_len; j++)
	        carr_x[j] = SvIV(*av_fetch(x, j, 0));
	    y = SvNV(*av_fetch(sample, 1, 0));
	    w = SvNV(*av_fetch(sample, 2, 0));
	    /* log likelihood */
	    for (y1 = 0; y1 < y_num; y1++) lambda_f[y1] = 0;
	    sum_exp_lambda_f = 0;
	    for (y1 = 0; y1 < y_num; y1++) {
	        for (j = 0; j < x_len; j++) {
		    lambda_f[y1] += carr_lambda[y1][carr_x[j]];
		}
		sum_exp_lambda_f += exp(lambda_f[y1]);
	    }
	    log_lh += w * (lambda_f[y] - log(sum_exp_lambda_f));
	    /* gradient */
	    for (y1 = 0; y1 < y_num; y1++) {
		fxy = (y1 == y ? 1 : 0);
		for (j = 0; j < x_len; j++) {
		    d_log_lh[carr_x[j] + x_num * y1] +=
		        w * (fxy - exp(lambda_f[y1]) / sum_exp_lambda_f);
		}
	    }
	    /* release sample */
	    free(carr_x);
	}
	TRACE("_neg_log_likelihood: finish log likelihood and gradient");
	/* smoothing */
	smoother = *hvref_fetch(self, "smoother");
	if (SvOK(smoother) && hvref_exists(smoother, "type")) {
	    smoother_type = SvPV_nolen(*hvref_fetch(smoother, "type"));
	    if (strcmp(smoother_type, "gaussian") == 0) {
	        sigma = SvOK(*hvref_fetch(smoother, "sigma")) ?
		    SvNV(*hvref_fetch(smoother, "sigma")) : 1.0;
		for (y1 = 0; y1 < y_num; y1++) {
		    for (x1 = 0; x1 < x_num; x1++) {
			l = carr_lambda[y1][x1];
			log_lh -= (l * l) / ( 2 * sigma * sigma);
			d_log_lh[x1 + x_num * y1] -= l / (sigma * sigma);
		    }
		}
	    }
	}
	TRACE("_neg_log_likelihood: finish smoothing");
	/* negate the value and finish */
	log_lh = -log_lh;
        av_d_log_lh = newAV();
	av_extend(av_d_log_lh, f_num - 1);
	for (i = 0; i < f_num; i++)
	    av_store(av_d_log_lh, i, newSVnv(-d_log_lh[i]));
	f = sv_2mortal(newSVnv(log_lh));
	g = sv_2mortal(newRV_noinc((SV*)av_d_log_lh));
	TRACE("_neg_log_likelihood: leave");
    CLEANUP:
	free(lambda_f);
	free(d_log_lh);
	for (i = 0; i< y_num; i++) free(carr_lambda[i]);
	free(carr_lambda);
        
