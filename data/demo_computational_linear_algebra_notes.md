# Computational Linear Algebra Demo Notes

These notes are original demo content for the hosted RAG assistant. They are not copied from a course handout. They provide enough retrieval signal to demonstrate the project without requiring a visitor to upload private documents first.

## Direct Methods For Linear Systems

A square linear system Ax = b can often be solved by Gaussian elimination or LU factorization. In LU factorization, the matrix A is decomposed into a lower triangular matrix L and an upper triangular matrix U. After factorization, solving Ax = b becomes two triangular solves: first solve Ly = b by forward substitution, then solve Ux = y by backward substitution.

Partial pivoting improves numerical stability by swapping rows so the pivot element is large relative to the entries below it. Pivoting is important when the current pivot is small, because division by a small value can amplify rounding error.

## Sparse And Banded Systems

Sparse matrices contain many zero entries. Algorithms should avoid storing and operating on zeros. A banded matrix has nonzero entries concentrated near the main diagonal. Specialized banded solvers reduce memory use and arithmetic cost compared with dense Gaussian elimination.

For a tridiagonal matrix, each row has nonzeros only on the main diagonal and the first diagonals above and below it. Such systems can be solved efficiently with a specialized elimination method that runs in linear time with respect to the number of unknowns.

## Iterative Methods

Iterative methods start with an initial guess and repeatedly improve it. They are useful for large systems where direct factorization is expensive or memory-intensive.

The Jacobi method updates each component using values from the previous iteration. Gauss-Seidel updates components immediately and uses the newest available values during the same iteration. Successive over-relaxation, or SOR, modifies Gauss-Seidel with a relaxation parameter omega. Choosing omega carefully can accelerate convergence, but a poor value can slow or prevent convergence.

Convergence of basic stationary methods depends on properties such as diagonal dominance and the spectral radius of the iteration matrix.

## Conjugate Gradient

The conjugate gradient method solves Ax = b when A is symmetric positive definite. It constructs search directions that are conjugate with respect to A. For well-conditioned systems, convergence can be fast. Preconditioning can improve convergence by transforming the system into one with more favorable spectral properties.

Conjugate gradient is often preferred for large sparse symmetric positive definite systems because it avoids explicitly factoring A.

## Least Squares And Overdetermined Systems

An overdetermined system has more equations than unknowns. Such a system may not have an exact solution. The least squares solution minimizes the residual norm ||Ax - b||_2.

The normal equations are A^T A x = A^T b, but forming A^T A can worsen conditioning. QR factorization is usually more numerically stable. If A = QR with Q having orthonormal columns, then the least squares problem can be solved by Rx = Q^T b.

The singular value decomposition, or SVD, is especially useful when the matrix is rank deficient or ill-conditioned. It exposes singular values, numerical rank, and the directions in which the data is most or least informative.

## QR Factorization

QR factorization decomposes a matrix A into Q and R, where Q has orthonormal columns and R is upper triangular. QR can be computed using Gram-Schmidt, modified Gram-Schmidt, Householder reflections, or Givens rotations.

Householder QR is commonly preferred for dense numerical linear algebra because it is stable. Givens rotations are useful when selectively zeroing entries, especially for sparse matrices or updating factorizations.

## Eigenvalue Methods

Eigenvalue problems ask for Ax = lambda x, where lambda is an eigenvalue and x is an eigenvector. The power method estimates the dominant eigenvalue when one eigenvalue has largest magnitude and the starting vector has a component in the corresponding eigenvector direction.

Inverse iteration can find an eigenvector associated with an eigenvalue near a chosen shift. The QR algorithm is a general method for computing eigenvalues by repeatedly applying QR factorization and multiplying RQ.

## Choosing A Method

Use LU or Gaussian elimination for moderate dense square systems. Use sparse or banded solvers when the matrix structure allows it. Use Jacobi, Gauss-Seidel, SOR, or conjugate gradient for large systems where iterative methods are cheaper than factorization. Use QR or SVD for least squares problems, especially when the system is overdetermined or ill-conditioned.
