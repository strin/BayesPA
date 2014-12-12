/*************************************************************************
Copyright (c) 1992-2007 The University of Tennessee.  All rights reserved.

Contributors:
    * Sergey Bochkanov (ALGLIB project). Translation from FORTRAN to
      pseudocode.

See subroutines comments for additional copyrights.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

- Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

- Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer listed
  in this license in the documentation and/or other materials
  provided with the distribution.

- Neither the name of the copyright holders nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*************************************************************************/

#ifndef _spdinverse_h
#define _spdinverse_h

#include "ap.h"

#include "cholesky.h"


/*************************************************************************
Inversion of a symmetric positive definite matrix which is given
by Cholesky decomposition.

Input parameters:
    A       -   Cholesky decomposition of the matrix to be inverted:
                A=U’*U or A = L*L'.
                Output of  CholeskyDecomposition subroutine.
                Array with elements [0..N-1, 0..N-1].
    N       -   size of matrix A.
    IsUpper –   storage format.
                If IsUpper = True, then matrix A is given as A = U'*U
                (matrix contains upper triangle).
                Similarly, if IsUpper = False, then A = L*L'.

Output parameters:
    A       -   upper or lower triangle of symmetric matrix A^-1, depending
                on the value of IsUpper.

Result:
    True, if the inversion succeeded.
    False, if matrix A contains zero elements on its main diagonal.
    Matrix A could not be inverted.

The algorithm is the modification of DPOTRI and DLAUU2 subroutines from
LAPACK library.
*************************************************************************/
bool spdmatrixcholeskyinverse(ap::real_2d_array& a, int n, bool isupper);


/*************************************************************************
Inversion of a symmetric positive definite matrix.

Given an upper or lower triangle of a symmetric positive definite matrix,
the algorithm generates matrix A^-1 and saves the upper or lower triangle
depending on the input.

Input parameters:
    A       -   matrix to be inverted (upper or lower triangle).
                Array with elements [0..N-1,0..N-1].
    N       -   size of matrix A.
    IsUpper -   storage format.
                If IsUpper = True, then the upper triangle of matrix A is
                given, otherwise the lower triangle is given.

Output parameters:
    A       -   inverse of matrix A.
                Array with elements [0..N-1,0..N-1].
                If IsUpper = True, then the upper triangle of matrix A^-1
                is used, and the elements below the main diagonal are not
                used nor changed. The same applies if IsUpper = False.

Result:
    True, if the matrix is positive definite.
    False, if the matrix is not positive definite (and it could not be
    inverted by this algorithm).
*************************************************************************/
bool spdmatrixinverse(ap::real_2d_array& a, int n, bool isupper);


/*************************************************************************
Obsolete subroutine.
*************************************************************************/
bool inversecholesky(ap::real_2d_array& a, int n, bool isupper);


/*************************************************************************
Obsolete subroutine.
*************************************************************************/
bool inversesymmetricpositivedefinite(ap::real_2d_array& a,
     int n,
     bool isupper);


#endif
