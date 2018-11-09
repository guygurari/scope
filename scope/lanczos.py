"""Lanczos algorithm as implemented in SciPy/ARPACK, but without the need to pass the actual matrix.

Only the product x |-> Ax is passed.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
# from scipy.sparse.linalg.eigen.arpack import ArpackError, \
#     ArpackNoConvergence, ReentrancyLock
from scipy._lib._util import _aligned_zeros
# from scipy._lib._threadsafety import ReentrancyLock
import scipy.sparse.linalg.eigen.arpack._arpack as _arpack

_type_conv = {"f": "s", "d": "d", "F": "c", "D": "z"}

# ARPACK is not threadsafe or reentrant (SAVE variables), so we need a
# lock and a re-entering check.
# _ARPACK_LOCK = ReentrancyLock("Nested calls to eigs/eighs not allowed: "
#                               "ARPACK is not re-entrant")

DSEUPD_ERRORS = {
    0:
        "Normal exit.",
    -1:
        "N must be positive.",
    -2:
        "NEV must be positive.",
    -3:
        "NCV must be greater than NEV and less than or equal to N.",
    -5:
        "WHICH must be one of 'LM', 'SM', 'LA', 'SA' or 'BE'.",
    -6:
        "BMAT must be one of 'I' or 'G'.",
    -7:
        "Length of private work WORKL array is not sufficient.",
    -8: ("Error return from trid. eigenvalue calculation; "
         "Information error from LAPACK routine dsteqr."),
    -9:
        "Starting vector is zero.",
    -10:
        "IPARAM(7) must be 1,2,3,4,5.",
    -11:
        "IPARAM(7) = 1 and BMAT = 'G' are incompatible.",
    -12:
        "NEV and WHICH = 'BE' are incompatible.",
    -14:
        "DSAUPD  did not find any eigenvalues to sufficient accuracy.",
    -15:
        "HOWMNY must be one of 'A' or 'S' if RVEC = .true.",
    -16:
        "HOWMNY = 'S' not yet implemented",
    -17: ("DSEUPD  got a different count of the number of converged "
          "Ritz values than DSAUPD  got.  This indicates the user "
          "probably made an error in passing data from DSAUPD  to "
          "DSEUPD  or that the data was modified before entering  "
          "DSEUPD.")
}

DSAUPD_ERRORS = {
    0: "Normal exit.",
    1: "Maximum number of iterations taken. "
       "All possible eigenvalues of OP has been found.",
    2: "No longer an informational error. Deprecated starting with "
       "release 2 of ARPACK.",
    3: "No shifts could be applied during a cycle of the Implicitly "
       "restarted Arnoldi iteration. One possibility is to increase "
       "the size of NCV relative to NEV. ",
    -1: "N must be positive.",
    -2: "NEV must be positive.",
    -3: "NCV must be greater than NEV and less than or equal to N.",
    -4: "The maximum number of Arnoldi update iterations allowed "
        "must be greater than zero.",
    -5: "WHICH must be one of 'LM', 'SM', 'LA', 'SA' or 'BE'.",
    -6: "BMAT must be one of 'I' or 'G'.",
    -7: "Length of private work array WORKL is not sufficient.",
    -8: "Error return from trid. eigenvalue calculation; "
        "Informational error from LAPACK routine dsteqr .",
    -9: "Starting vector is zero.",
    -10: "IPARAM(7) must be 1,2,3,4,5.",
    -11: "IPARAM(7) = 1 and BMAT = 'G' are incompatible.",
    -12: "IPARAM(1) must be equal to 0 or 1.",
    -13: "NEV and WHICH = 'BE' are incompatible. ",
    -9999: "Could not build an Arnoldi factorization. "
           "IPARAM(5) returns the size of the current Arnoldi "
           "factorization. The user is advised to check that "
           "enough workspace and array storage has been allocated.",
}

DSAUPD_ERRORS = {
    0: "Normal exit.",
    1: "Maximum number of iterations taken. "
       "All possible eigenvalues of OP has been found.",
    2: "No longer an informational error. Deprecated starting with "
       "release 2 of ARPACK.",
    3: "No shifts could be applied during a cycle of the Implicitly "
       "restarted Arnoldi iteration. One possibility is to increase "
       "the size of NCV relative to NEV. ",
    -1: "N must be positive.",
    -2: "NEV must be positive.",
    -3: "NCV must be greater than NEV and less than or equal to N.",
    -4: "The maximum number of Arnoldi update iterations allowed "
        "must be greater than zero.",
    -5: "WHICH must be one of 'LM', 'SM', 'LA', 'SA' or 'BE'.",
    -6: "BMAT must be one of 'I' or 'G'.",
    -7: "Length of private work array WORKL is not sufficient.",
    -8: "Error return from trid. eigenvalue calculation; "
        "Informational error from LAPACK routine dsteqr .",
    -9: "Starting vector is zero.",
    -10: "IPARAM(7) must be 1,2,3,4,5.",
    -11: "IPARAM(7) = 1 and BMAT = 'G' are incompatible.",
    -12: "IPARAM(1) must be equal to 0 or 1.",
    -13: "NEV and WHICH = 'BE' are incompatible. ",
    -9999: "Could not build an Arnoldi factorization. "
           "IPARAM(5) returns the size of the current Arnoldi "
           "factorization. The user is advised to check that "
           "enough workspace and array storage has been allocated.",
}

SSAUPD_ERRORS = DSAUPD_ERRORS
SSEUPD_ERRORS = DSEUPD_ERRORS.copy()

SSAUPD_ERRORS = DSAUPD_ERRORS

_SAUPD_ERRORS = {"d": DSAUPD_ERRORS, "s": SSAUPD_ERRORS}

_SEUPD_ERRORS = {"d": DSEUPD_ERRORS, "s": SSEUPD_ERRORS}

# accepted values of parameter WHICH in _SEUPD
_SEUPD_WHICH = ["LM", "SM", "LA", "SA", "BE"]


def choose_ncv(k):
  """
    Choose number of lanczos vectors based on target number
    of singular/eigen values and vectors to compute, k.
    """
  return max(2 * k + 1, 20)


def eigsh(n,
          dtype,
          matvec,
          k=6,
          which="LM",
          v0=None,
          ncv=None,
          maxiter=None,
          tol=0,
          return_eigenvectors=True,
          mode="normal"):
  """
    Find k eigenvalues and eigenvectors of the real symmetric square matrix
    or complex hermitian matrix A. The matrix is only specified through
    matvec, the matrix-vector product Ax.

    Solves ``A * x[i] = w[i] * x[i]``, the standard eigenvalue problem for
    w[i] eigenvalues with corresponding eigenvectors x[i].

    Parameters
    ----------
    n : int, the dimension of the real-symmetric matrix A.
    dtype: The element type of A
    matvec : A function that takes a vector x and returns Ax.
    k : int, optional
        The number of eigenvalues and eigenvectors desired.
        `k` must be smaller than N. It is not possible to compute all
        eigenvectors of a matrix.

    Returns
    -------
    w : array
        Array of k eigenvalues
    v : array
        An array representing the `k` eigenvectors.  The column ``v[:, i]`` is
        the eigenvector corresponding to the eigenvalue ``w[i]``.

    Other Parameters
    ----------------
    v0 : ndarray, optional
        Starting vector for iteration.
        Default: random
    ncv : int, optional
        The number of Lanczos vectors generated ncv must be greater than k and
        smaller than n; it is recommended that ``ncv > 2*k``.
        Default: ``min(n, max(2*k + 1, 20))``
    which : str ['LM' | 'SM' | 'LA' | 'SA' | 'BE']
        If A is a complex hermitian matrix, 'BE' is invalid.
        Which `k` eigenvectors and eigenvalues to find:

            'LM' : Largest (in magnitude) eigenvalues

            'SM' : Smallest (in magnitude) eigenvalues

            'LA' : Largest (algebraic) eigenvalues

            'SA' : Smallest (algebraic) eigenvalues

            'BE' : Half (k/2) from each end of the spectrum

        When k is odd, return one more (k/2+1) from the high end.
        When sigma != None, 'which' refers to the shifted eigenvalues ``w'[i]``
        (see discussion in 'sigma', above).  ARPACK is generally better
        at finding large values than small values.  If small eigenvalues are
        desired, consider using shift-invert mode for better performance.
    maxiter : int, optional
        Maximum number of Arnoldi update iterations allowed
        Default: ``n*10``
    tol : float
        Relative accuracy for eigenvalues (stopping criterion).
        The default value of 0 implies machine precision.
    return_eigenvectors : bool
        Return eigenvectors (True) in addition to eigenvalues
    mode : string ['normal' | 'buckling' | 'cayley']
        Specify strategy to use for shift-invert mode.  This argument applies
        only for real-valued A and sigma != None.  For shift-invert mode,
        ARPACK internally solves the eigenvalue problem
        ``OP * x'[i] = w'[i] * B * x'[i]``
        and transforms the resulting Ritz vectors x'[i] and Ritz values w'[i]
        into the desired eigenvectors and eigenvalues of the problem
        ``A * x[i] = w[i] * M * x[i]``.
        The modes are as follows:

            'normal' :
                OP = [A - sigma * M]^-1 * M,
                B = M,
                w'[i] = 1 / (w[i] - sigma)

            'buckling' :
                OP = [A - sigma * M]^-1 * A,
                B = A,
                w'[i] = w[i] / (w[i] - sigma)

            'cayley' :
                OP = [A - sigma * M]^-1 * [A + sigma * M],
                B = M,
                w'[i] = (w[i] + sigma) / (w[i] - sigma)

        The choice of mode will affect which eigenvalues are selected by
        the keyword 'which', and can also impact the stability of
        convergence (see [2] for a discussion)

    Raises
    ------
    ArpackNoConvergence
        When the requested convergence is not obtained.

        The currently converged eigenvalues and eigenvectors can be found
        as ``eigenvalues`` and ``eigenvectors`` attributes of the exception
        object.

    See Also
    --------
    eigs : eigenvalues and eigenvectors for a general (nonsymmetric) matrix A
    svds : singular value decomposition for a matrix A

    Notes
    -----
    This function is a wrapper to the ARPACK [1]_ SSEUPD and DSEUPD
    functions which use the Implicitly Restarted Lanczos Method to
    find the eigenvalues and eigenvectors [2]_.

    References
    ----------
    .. [1] ARPACK Software, http://www.caam.rice.edu/software/ARPACK/
    .. [2] R. B. Lehoucq, D. C. Sorensen, and C. Yang,  ARPACK USERS GUIDE:
       Solution of Large Scale Eigenvalue Problems by Implicitly Restarted
       Arnoldi Methods. SIAM, Philadelphia, PA, 1998.

    Examples
    --------
    >>> import scipy.sparse as sparse
    >>> id = np.eye(13)
    >>> vals, vecs = sparse.linalg.eigsh(id, k=6)
    >>> vals
    array([ 1.+0.j,  1.+0.j,  1.+0.j,  1.+0.j,  1.+0.j,  1.+0.j])
    >>> vecs.shape
    (13, 6)

    """
  # complex hermitian matrices should be solved with eigs
  if np.issubdtype(dtype, np.complexfloating):
    raise ValueError("Complex matrix not supported")

  if k <= 0 or k >= n:
    raise ValueError("k must be between 1 and the order of the "
                     "square input matrix.")

  # standard eigenvalue problem
  mode = 1
  M_matvec = None
  Minv_matvec = None
  sigma = None
  params = _SymmetricArpackParams(n, k,
                                  np.dtype(dtype).char, matvec, mode, M_matvec,
                                  Minv_matvec, sigma, ncv, v0, maxiter, which,
                                  tol)

  # with _ARPACK_LOCK:
  while not params.converged:
    params.iterate()

  return params.extract(return_eigenvectors)


class _ArpackParams(object):

  def __init__(self,
               n,
               k,
               tp,
               mode=1,
               sigma=None,
               ncv=None,
               v0=None,
               maxiter=None,
               which="LM",
               tol=0):
    if k <= 0:
      raise ValueError("k must be positive, k=%d" % k)

    if maxiter is None:
      maxiter = n * 10
    if maxiter <= 0:
      raise ValueError("maxiter must be positive, maxiter=%d" % maxiter)

    if tp not in "fdFD":
      raise ValueError("matrix type must be 'f', 'd', 'F', or 'D'")

    if v0 is not None:
      # ARPACK overwrites its initial resid,  make a copy
      self.resid = np.array(v0, copy=True)
      info = 1
    else:
      # ARPACK will use a random initial vector.
      self.resid = np.zeros(n, tp)
      info = 0

    if sigma is None:
      # sigma not used
      self.sigma = 0
    else:
      self.sigma = sigma

    if ncv is None:
      ncv = choose_ncv(k)
    ncv = min(ncv, n)

    self.v = np.zeros((n, ncv), tp)  # holds Ritz vectors
    self.iparam = np.zeros(11, "int")

    # set solver mode and parameters
    ishfts = 1
    self.mode = mode
    self.iparam[0] = ishfts
    self.iparam[2] = maxiter
    self.iparam[3] = 1
    self.iparam[6] = mode

    self.n = n
    self.tol = tol
    self.k = k
    self.maxiter = maxiter
    self.ncv = ncv
    self.which = which
    self.tp = tp
    self.info = info

    self.converged = False
    self.ido = 0

  def _raise_no_convergence(self):
    msg = "No convergence (%d iterations, %d/%d eigenvectors converged)"
    k_ok = self.iparam[4]
    num_iter = self.iparam[2]
    try:
      ev, vec = self.extract(True)
    except ArpackError as err:
      msg = "%s [%s]" % (msg, err)
      ev = np.zeros((0,))
      vec = np.zeros((self.n, 0))
      k_ok = 0
    raise ArpackNoConvergence(msg % (num_iter, k_ok, self.k), ev, vec)


class _SymmetricArpackParams(_ArpackParams):
  """Copied as-is from SciPy"""

  def __init__(self,
               n,
               k,
               tp,
               matvec,
               mode=1,
               M_matvec=None,
               Minv_matvec=None,
               sigma=None,
               ncv=None,
               v0=None,
               maxiter=None,
               which="LM",
               tol=0):
    # The following modes are supported:
    #  mode = 1:
    #    Solve the standard eigenvalue problem:
    #      A*x = lambda*x :
    #       A - symmetric
    #    Arguments should be
    #       matvec      = left multiplication by A
    #       M_matvec    = None [not used]
    #       Minv_matvec = None [not used]
    #
    #  mode = 2:
    #    Solve the general eigenvalue problem:
    #      A*x = lambda*M*x
    #       A - symmetric
    #       M - symmetric positive definite
    #    Arguments should be
    #       matvec      = left multiplication by A
    #       M_matvec    = left multiplication by M
    #       Minv_matvec = left multiplication by M^-1
    #
    #  mode = 3:
    #    Solve the general eigenvalue problem in shift-invert mode:
    #      A*x = lambda*M*x
    #       A - symmetric
    #       M - symmetric positive semi-definite
    #    Arguments should be
    #       matvec      = None [not used]
    #       M_matvec    = left multiplication by M
    #                     or None, if M is the identity
    #       Minv_matvec = left multiplication by [A-sigma*M]^-1
    #
    #  mode = 4:
    #    Solve the general eigenvalue problem in Buckling mode:
    #      A*x = lambda*AG*x
    #       A  - symmetric positive semi-definite
    #       AG - symmetric indefinite
    #    Arguments should be
    #       matvec      = left multiplication by A
    #       M_matvec    = None [not used]
    #       Minv_matvec = left multiplication by [A-sigma*AG]^-1
    #
    #  mode = 5:
    #    Solve the general eigenvalue problem in Cayley-transformed mode:
    #      A*x = lambda*M*x
    #       A - symmetric
    #       M - symmetric positive semi-definite
    #    Arguments should be
    #       matvec      = left multiplication by A
    #       M_matvec    = left multiplication by M
    #                     or None, if M is the identity
    #       Minv_matvec = left multiplication by [A-sigma*M]^-1
    if mode == 1:
      if matvec is None:
        raise ValueError("matvec must be specified for mode=1")
      if M_matvec is not None:
        raise ValueError("M_matvec cannot be specified for mode=1")
      if Minv_matvec is not None:
        raise ValueError("Minv_matvec cannot be specified for mode=1")

      self.OP = matvec
      self.B = lambda x: x
      self.bmat = "I"
    elif mode == 2:
      if matvec is None:
        raise ValueError("matvec must be specified for mode=2")
      if M_matvec is None:
        raise ValueError("M_matvec must be specified for mode=2")
      if Minv_matvec is None:
        raise ValueError("Minv_matvec must be specified for mode=2")

      self.OP = lambda x: Minv_matvec(matvec(x))
      self.OPa = Minv_matvec
      self.OPb = matvec
      self.B = M_matvec
      self.bmat = "G"
    elif mode == 3:
      if matvec is not None:
        raise ValueError("matvec must not be specified for mode=3")
      if Minv_matvec is None:
        raise ValueError("Minv_matvec must be specified for mode=3")

      if M_matvec is None:
        self.OP = Minv_matvec
        self.OPa = Minv_matvec
        self.B = lambda x: x
        self.bmat = "I"
      else:
        self.OP = lambda x: Minv_matvec(M_matvec(x))
        self.OPa = Minv_matvec
        self.B = M_matvec
        self.bmat = "G"
    elif mode == 4:
      if matvec is None:
        raise ValueError("matvec must be specified for mode=4")
      if M_matvec is not None:
        raise ValueError("M_matvec must not be specified for mode=4")
      if Minv_matvec is None:
        raise ValueError("Minv_matvec must be specified for mode=4")
      self.OPa = Minv_matvec
      self.OP = lambda x: self.OPa(matvec(x))
      self.B = matvec
      self.bmat = "G"
    elif mode == 5:
      if matvec is None:
        raise ValueError("matvec must be specified for mode=5")
      if Minv_matvec is None:
        raise ValueError("Minv_matvec must be specified for mode=5")

      self.OPa = Minv_matvec
      self.A_matvec = matvec

      if M_matvec is None:
        self.OP = lambda x: Minv_matvec(matvec(x) + sigma * x)
        self.B = lambda x: x
        self.bmat = "I"
      else:
        self.OP = lambda x: Minv_matvec(matvec(x) + sigma * M_matvec(x))
        self.B = M_matvec
        self.bmat = "G"
    else:
      raise ValueError("mode=%i not implemented" % mode)

    if which not in _SEUPD_WHICH:
      raise ValueError("which must be one of %s" % " ".join(_SEUPD_WHICH))
    if k >= n:
      raise ValueError("k must be less than ndim(A), k=%d" % k)

    _ArpackParams.__init__(self, n, k, tp, mode, sigma, ncv, v0, maxiter, which,
                           tol)

    if self.ncv > n or self.ncv <= k:
      raise ValueError("ncv must be k<ncv<=n, ncv=%s" % self.ncv)

    # Use _aligned_zeros to work around a f2py bug in Numpy 1.9.1
    self.workd = _aligned_zeros(3 * n, self.tp)
    self.workl = _aligned_zeros(self.ncv * (self.ncv + 8), self.tp)

    ltr = _type_conv[self.tp]
    if ltr not in ["s", "d"]:
      raise ValueError("Input matrix is not real-valued.")

    self._arpack_solver = _arpack.__dict__[ltr + "saupd"]
    self._arpack_extract = _arpack.__dict__[ltr + "seupd"]

    self.iterate_infodict = _SAUPD_ERRORS[ltr]
    self.extract_infodict = _SEUPD_ERRORS[ltr]

    self.ipntr = np.zeros(11, "int")

  def iterate(self):
    self.ido, self.tol, self.resid, self.v, self.iparam, self.ipntr, self.info = \
        self._arpack_solver(self.ido, self.bmat, self.which, self.k,
                            self.tol, self.resid, self.v, self.iparam,
                            self.ipntr, self.workd, self.workl, self.info)

    xslice = slice(self.ipntr[0] - 1, self.ipntr[0] - 1 + self.n)
    yslice = slice(self.ipntr[1] - 1, self.ipntr[1] - 1 + self.n)
    if self.ido == -1:
      # initialization
      self.workd[yslice] = self.OP(self.workd[xslice])
    elif self.ido == 1:
      # compute y = Op*x
      if self.mode == 1:
        self.workd[yslice] = self.OP(self.workd[xslice])
      elif self.mode == 2:
        self.workd[xslice] = self.OPb(self.workd[xslice])
        self.workd[yslice] = self.OPa(self.workd[xslice])
      elif self.mode == 5:
        Bxslice = slice(self.ipntr[2] - 1, self.ipntr[2] - 1 + self.n)
        Ax = self.A_matvec(self.workd[xslice])
        self.workd[yslice] = self.OPa(Ax + (self.sigma * self.workd[Bxslice]))
      else:
        Bxslice = slice(self.ipntr[2] - 1, self.ipntr[2] - 1 + self.n)
        self.workd[yslice] = self.OPa(self.workd[Bxslice])
    elif self.ido == 2:
      self.workd[yslice] = self.B(self.workd[xslice])
    elif self.ido == 3:
      raise ValueError("ARPACK requested user shifts.  Assure ISHIFT==0")
    else:
      self.converged = True

      if self.info == 0:
        pass
      elif self.info == 1:
        self._raise_no_convergence()
      else:
        raise ArpackError(self.info, infodict=self.iterate_infodict)

  def extract(self, return_eigenvectors):
    rvec = return_eigenvectors
    ierr = 0
    howmny = "A"  # return all eigenvectors
    sselect = np.zeros(self.ncv, "int")  # unused
    d, z, ierr = self._arpack_extract(
        rvec, howmny, sselect, self.sigma, self.bmat, self.which, self.k,
        self.tol, self.resid, self.v, self.iparam[0:7], self.ipntr,
        self.workd[0:2 * self.n], self.workl, ierr)
    if ierr != 0:
      raise ArpackError(ierr, infodict=self.extract_infodict)
    k_ok = self.iparam[4]
    d = d[:k_ok]
    z = z[:, :k_ok]

    if return_eigenvectors:
      return d, z
    else:
      return d
