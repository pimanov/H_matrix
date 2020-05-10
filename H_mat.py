import numpy as np
import scipy
from scipy import linalg


def rsvd(a, b):
    """
    Produce an SVD for a matrix A = a @ b
    
    Parametrs
    ---------
    a, b : numpy.ndarray
    
    Return 
    ------
    u, s, vh : numpy.ndarray

    u * s @ vh = A
    """
    
    u_a, v_a = np.linalg.qr(a)
    u_bT, v_bT = np.linalg.qr(b.T)

    usv = v_a @ v_bT.T
    u_s, s, v_s = np.linalg.svd(usv, full_matrices=False)
    
    u = u_a @ u_s
    vh = v_s @ u_bT.T
    return u, s, vh


def create(matrix, tol=1e-6, rang=10, diag_f=True):
    """
    Create H-matrix based on the dense matrix 'matrix'.

    Parametrs
    ---------
    matrix : numpy.ndarray - dense matrix 
    tol : float - threashold singula values for low rang blocks to be stored
    rang : threacshold rang for block to be stored factorised
    diag_f : bool - True if this block in diagonal of the matris (It is full matrix).

    Return
    ------
    H_matrix
    """ 
    n, m = matrix.shape

    if n <= 2 * rang or m <= 2 * rang:
        return H_matrix('mat', matrix.copy())

    if not diag_f:
        u, s, vh = np.linalg.svd(matrix, full_matrices=False)

        if s[rang] < tol:
            r = [i for i, s1 in enumerate(s) if s1 < tol][0]
            us = u[:,:r] * s[:r]
            vh = vh[:r]
            return H_matrix('svd', us, vh)

    A11 = create(matrix[:n//2, :m//2], tol, rang, diag_f)
    A21 = create(matrix[n//2:, :m//2], tol, rang, False)
    A12 = create(matrix[:n//2, m//2:], tol, rang, False)
    A22 = create(matrix[n//2:, m//2:], tol, rang, diag_f)
    return H_matrix('split', A11, A12, A21, A22)


def lu_solve(L, U, b):
    """
    lu_solve(L, U, b) solves equation LUx=b
    L - lower triangular H_matrix
    B - upper triangular H_matrix
    b - vector of type numpy.ndarray
    
    Return
    ------
    x : numpy.ndarray
    
    To calculate L and U factors of H_matrix H, use H.lu()
    """
    y = L.lsolve(b)
    x = U.usolve(y)
    return x


class H_matrix:
    def __init__(self, *args):
        """
        Create H_matrix node of one of three type ('mat', 'svd', 'split'). 

        H_matrix('mat', matrix) -- Create H_matrix block with dense matrix 'matrix' of type numpy.ndarray
        H_matrix('svd', us, vh) -- Create H_matrix block with low-rang matrix A = us @ vh. us, vh : numpy.ndarray
        H_matrix('split', H11, H12, H21, H22) -- Create H_matrix node with four subbloks of type of H_matrix
        
        Return
        ------
        H-matrix
        
        """
        if args[0] == 'svd':
            self.type = 'svd'
            self.us = args[1]
            self.vh = args[2]
            return

        if args[0] == 'mat':
            self.type = 'mat'
            self.A = args[1]
            return

        if args[0] == 'split':
            self.type = 'split'
            self.A11 = args[1]
            self.A12 = args[2]
            self.A21 = args[3]
            self.A22 = args[4]
            return
        
    
    def show(self):
        """
        H.show() creates 2D numpy.ndarray, that represents block structure of the matrix H. 
        
        Return matrix contains:
        2.0 for block borders,
        1.0 for stored elements
        0.0 for other
        
        Typical use: plt.imshow(S.show())
        """
        if self.type == 'mat':
            res = np.ones(self.A.shape)
            res[0] = res[-1] = res[:,0] = res[:,-1] = 2
            return res
        
        if self.type == 'svd':
            n, r = self.us.shape
            r, m = self.vh.shape
            res = np.zeros((n, m))
            res[:r] = res[:,:r] = 1
            res[0] = res[-1] = res[:,0] = res[:,-1] = 2
            return res
        
        if self.type == 'split':
            S11 = self.A11.show()
            S22 = self.A22.show()
            S21 = self.A21.show()
            S12 = self.A12.show()
            return np.hstack((np.vstack((S11, S21)), np.vstack((S12, S22))))
            
            
    def mat(self):
        """
        H.mat() creates dense matrix of type numpy.ndarray corresponding for the matrix H
        
        Return 2D numpy.ndarray
        """
        if self.type == 'mat':
            return self.A
        
        if self.type == 'svd':
            return self.us @ self.vh
        
        if self.type == 'split':
            res1 = np.hstack((self.A11.mat(), self.A12.mat()))
            res2 = np.hstack((self.A21.mat(), self.A22.mat()))
            return np.vstack((res1, res2))
        
        
    def dot(self, vec):
        """
        matmul

        B -- H_matrix or vector of type numpy.ndarray
        Return -- H_matrix or vector of type numpy.ndarray
        """
        return self @ vec
    
    
    def __matmul__(self, B):
        """
        matmul

        B -- H_matrix or vector of type numpy.ndarray
        Return -- H_matrix or vector of type numpy.ndarray
        """
        tol = 1e-10
        
        if type(B) == np.ndarray:
            if self.type == 'mat':
                return self.A @ B
            if self.type == 'svd':
                return self.us @ (self.vh @ B)
            if self.type == 'split':
                n = B.shape[0]
                res1 = self.A11 @ B[:n//2] + self.A12 @ B[n//2:]
                res2 = self.A21 @ B[:n//2] + self.A22 @ B[n//2:]
                return np.concatenate((res1, res2))
        
        
        if self.type == 'mat':
            S = H_matrix('mat', self.A @ B.mat())
            
        if self.type == 'svd':
            if B.type == 'split':
                A11, A12, A21, A22 = self.split()
                B11, B12, B21, B22 = B.split()
                
                S11 = A11 @ B11 + A12 @ B21
                S12 = A12 @ B22 + A11 @ B12
                S21 = A21 @ B11 + A22 @ B21
                S22 = A22 @ B22 + A21 @ B12
                
                us1 = np.hstack((S11.us, np.zeros((S11.us.shape[0], S21.us.shape[1])),
                                 S12.us, np.zeros((S12.us.shape[0], S22.us.shape[1]))))
                us2 = np.hstack((np.zeros((S21.us.shape[0], S11.us.shape[1])), S21.us,
                                 np.zeros((S22.us.shape[0], S12.us.shape[1])), S22.us))
                us = np.vstack((us1, us2))
                
                vh1 = np.vstack((S11.vh, S21.vh, np.zeros((S12.vh.shape[0], S11.vh.shape[1])), 
                                 np.zeros((S22.vh.shape[0], S21.vh.shape[1]))))
                vh2 = np.vstack((np.zeros((S11.vh.shape[0], S12.vh.shape[1])), 
                                 np.zeros((S21.vh.shape[0], S22.vh.shape[1])), S12.vh, S22.vh))
                vh = np.hstack((vh1, vh2))

                u, s, vh = rsvd(us, vh)
                
            
            if B.type == 'mat':
                S = self.mat() @ B.mat()
                u, s, vh = np.linalg.svd(S, full_matrices=False)

                
            if B.type == 'svd':
                us = self.us @ (self.vh @ B.us)
                u, s, vh = rsvd(us, B.vh)

                
            tail = [i for i, s0 in enumerate(s) if s0 < tol]
            if len(tail) > 0:
                r = tail[0]
                us = u[:,:r] * s[:r]
                vh = vh[:r].copy()
            else:
                us = u * s
                vh = vh
                
            S = H_matrix('svd', us, vh)
                
        
        if self.type == 'split':
            A11, A12, A21, A22 = self.split()
            B11, B12, B21, B22 = B.split()
            
            S11 = A11 @ B11 + A12 @ B21
            S12 = A12 @ B22 + A11 @ B12
            S21 = A21 @ B11 + A22 @ B21
            S22 = A22 @ B22 + A21 @ B12

            S = H_matrix('split', S11, S12, S21, S22)
            
        return S
        

    def split(self):
        """
        Split H_matrix for four blocks of type H_matrix
        """

        if self.type == 'mat':
            n, m = self.A.shape
            S11 = H_matrix('mat', self.A[:n//2, :m//2])
            S12 = H_matrix('mat', self.A[:n//2, m//2:])
            S21 = H_matrix('mat', self.A[n//2:, :m//2])
            S22 = H_matrix('mat', self.A[n//2:, m//2:])
            
        if self.type == 'svd':
            n, m = self.us.shape[0], self.vh.shape[1]
            S11 = H_matrix('svd', self.us[:n//2], self.vh[:,:m//2])
            S12 = H_matrix('svd', self.us[:n//2], self.vh[:,m//2:])
            S21 = H_matrix('svd', self.us[n//2:], self.vh[:,:m//2])
            S22 = H_matrix('svd', self.us[n//2:], self.vh[:,m//2:])
    
        if self.type == 'split':
            S11, S12, S21, S22 = self.A11, self.A12, self.A21, self.A22
            
        return S11, S12, S21, S22
        
        
    def copy(self):
        """
        Create deep copy of H_matrix.
        """
        if self.type == 'mat':
            return H_matrix('mat', self.A.copy())
        
        if self.type == 'svd':
            return H_matrix('svd', self.us.copy(), self.vh.copy())
        
        if self.type == 'split':
            return H_matrix('split', self.A11.copy(), self.A12.copy(), self.A21.copy(), self.A22.copy())
        
        
    def __add__(self, B, alpha=1):
        """
        Sum
        
        A.__add__(B, alpha) returns H = A + alpha * B
        B - H_matrix
        Return -- H_matrix with the same block structure as that of the matrix A
        """
        
        tol = 1e-10
        
        if self.type == 'mat':
            S1 = H_matrix('mat', self.A + alpha * B.mat())
            
        if self.type == 'svd':
            if B.type == 'svd':
                us = np.hstack((self.us, alpha * B.us))
                vh = np.vstack((self.vh, B.vh))
                u, s, vh = rsvd(us, vh)

            if B.type == 'mat':
                M = self.us @ self.vh + alpha * B.mat()
                u, s, vh = np.linalg.svd(M, full_matrices=False)
                
            if B.type == 'split':
                A11, A12, A21, A22 = self.split()
                B11, B12, B21, B22 = B.split()
                
                S11 = A11 + B11
                S12 = A12 + B12
                S21 = A21 + B21
                S22 = A22 + B22
                
                us1 = np.hstack((S11.us, np.zeros((S11.us.shape[0], S21.us.shape[1])),
                                 S12.us, np.zeros((S12.us.shape[0], S22.us.shape[1]))))
                us2 = np.hstack((np.zeros((S21.us.shape[0], S11.us.shape[1])), S21.us,
                                 np.zeros((S22.us.shape[0], S12.us.shape[1])), S22.us))
                us = np.vstack((us1, us2))
                
                vh1 = np.vstack((S11.vh, S21.vh, np.zeros((S12.vh.shape[0], S11.vh.shape[1])), 
                                 np.zeros((S22.vh.shape[0], S21.vh.shape[1]))))
                vh2 = np.vstack((np.zeros((S11.vh.shape[0], S12.vh.shape[1])), 
                                 np.zeros((S21.vh.shape[0], S22.vh.shape[1])), S12.vh, S22.vh))
                vh = np.hstack((vh1, vh2))

                u, s, vh = rsvd(us, vh)
                
                
            tail = [i for i, s0 in enumerate(s) if s0 < tol]
            if len(tail) > 0:
                r = tail[0]
                us = u[:,:r] * s[:r]
                vh = vh[:r].copy()
            else:
                us = u * s
                vh = vh

            S1 = H_matrix('svd', us, vh)
        
        
        if self.type == 'split':
            S11, S12, S21, S22 = self.split()  
            B11, B12, B21, B22 = B.split()
            S1 = H_matrix('split', S11.__add__(B11, alpha), S12.__add__(B12, alpha), 
                        S21.__add__(B21, alpha), S22.__add__(B22, alpha))
            
        return S1
    
    
    def __sub__(self, B):
        """
        Substraction
        
        B - H_matrix
        Return -- H_matrix with the same block structure as that of the matrix 'self'
        """
        return self.__add__(B, -1)
    
    
    def __neg__(self):
        """
        Create H_matrix H = -A
        """
        if self.type == 'mat':
            return H_matrix('mat', -self.A)
        
        if self.type == 'svd':
            return H_matrix('svd', -self.us, self.vh)
        
        if self.type == 'split':
            return H_matrix('split', -self.A11, -self.A12, -self.A21, -self.A22)
        
        
    def lu(self):
        """
        Produces LU decomposition, 
        L - lower triangular H-matrix, 
        U - upper triangulat H-matrix
        
        Return
        ------
        L, U : H_matrix with the same block structure as that of the 'self' matrix
        """
        if self.type == 'mat':
            l, u = scipy.linalg.lu(self.A, permute_l=True)
            return H_matrix('mat', l), H_matrix('mat', u)
        
        if self.type == 'split':
            L11, U11 = self.A11.lu()
            U12 = L11.lsolve(self.A12)
            L21 = U11.uTsolve(self.A21)
            Tmp = self.A22.copy()
            Tmp.addmatmul(L21, U12, -1)
            L22, U22 = Tmp.lu()
            Z12 = self.A12.zeros_like()
            Z21 = self.A21.zeros_like()
            L = H_matrix('split', L11, Z12, L21, L22)
            U = H_matrix('split', U11, U12, Z21, U22)
            return L, U 
         
    
    def usolve(self, B):
        """
        U.usolve(B) solves equation Ux = B 
        U - upper triangular matrix
        B - H_matrix or vector of type numpy.ndarray
        
        Return
        ------
        numpy.ndarray or H_matrix with the same block structure as that of the matrix B
        """
        tol = 1e-10
        
        if type(B) == np.ndarray:
            if self.type == 'mat':
                return np.linalg.solve(self.A, B)
            
            U11, U12, _, U22 = self.split()
            n = B.shape[0]
            b1, b2 = B[:n//2], B[n//2:]
            
            x2 = U22.usolve(b2)
            x1 = U11.usolve(b1 - U12 @ x2)
            
            return np.concatenate((x1, x2))
            
        if B.type == 'mat':
            return H_matrix('mat', np.linalg.solve(self.mat(), B.A))
        
        if B.type == 'svd' and self.type == 'mat':
            A_inv = np.linalg.inv(self.A)
            return H_matrix('svd', A_inv @ B.us, B.vh)
        
        if self.type == 'split' or B.type == 'split':
            U11, U12, _, U22 = self.split()
            B11, B12, B21, B22 = B.split()

            X21 = U22.usolve(B21)
            Tmp = B11.copy()
            Tmp.addmatmul(U12, X21, -1)
            X11 = U11.usolve(Tmp)
            
            X22 = U22.usolve(B22)
            Tmp = B12.copy()
            Tmp.addmatmul(U12, X22, -1)
            X12 = U11.usolve(Tmp)

        if B.type == 'split':
            return H_matrix('split', X11, X12, X21, X22)
        
        if B.type == 'svd':
            us1 = np.hstack((X11.us, np.zeros((X11.us.shape[0], X21.us.shape[1])),
                             X12.us, np.zeros((X12.us.shape[0], X22.us.shape[1]))))
            us2 = np.hstack((np.zeros((X21.us.shape[0], X11.us.shape[1])), X21.us,
                             np.zeros((X22.us.shape[0], X12.us.shape[1])), X22.us))
            us = np.vstack((us1, us2))

            vh1 = np.vstack((X11.vh, X21.vh, np.zeros((X12.vh.shape[0], X11.vh.shape[1])), 
                             np.zeros((X22.vh.shape[0], X21.vh.shape[1]))))
            vh2 = np.vstack((np.zeros((X11.vh.shape[0], X12.vh.shape[1])), 
                             np.zeros((X21.vh.shape[0], X22.vh.shape[1])), X12.vh, X22.vh))
            vh = np.hstack((vh1, vh2))

            u, s, vh = rsvd(us, vh)

                
            tail = [i for i, s0 in enumerate(s) if s0 < tol]
            if len(tail) > 0:
                r = tail[0]
                us = u[:,:r] * s[:r]
                vh = vh[:r].copy()
            else:
                us = u * s
                vh = vh

            return H_matrix('svd', us, vh)
        
        
    def lsolve(self, B):
        """
        L.lsolve(B) solves equation Lx = B 
        L - lower triangular matrix
        B - H_matrix or vector of type numpy.ndarray
        
        Return
        ------
        numpy.ndarray or H_matrix with the same block structure as that of the matrix B
        """
        tol = 1e-10
        
        if type(B) == np.ndarray:
            if self.type == 'mat':
                return np.linalg.solve(self.A, B)
            
            L11, _, L21, L22 = self.split()
            n = B.shape[0]
            b1, b2 = B[:n//2], B[n//2:]
            
            x1 = L11.lsolve(b1)
            x2 = L22.lsolve(b2 - L21 @ x1)
            
            return np.concatenate((x1, x2))
            
        if B.type == 'mat':
            return H_matrix('mat', np.linalg.solve(self.mat(), B.A))
        
        if B.type == 'svd' and self.type == 'mat':
            A_inv = np.linalg.inv(self.A)
            return H_matrix('svd', A_inv @ B.us, B.vh)
        
        if self.type == 'split' or B.type == 'split':
            L11, _, L21, L22 = self.split()
            B11, B12, B21, B22 = B.split()

            X11 = L11.lsolve(B11)
            Tmp = B21.copy()
            Tmp.addmatmul(L21, X11, -1)
            X21 = L22.lsolve(Tmp)
            X12 = L11.lsolve(B12)
            Tmp = B22.copy()
            Tmp.addmatmul(L21, X12, -1)
            X22 = L22.lsolve(Tmp)

        if B.type == 'split':
            return H_matrix('split', X11, X12, X21, X22)
        
        if B.type == 'svd':
            us1 = np.hstack((X11.us, np.zeros((X11.us.shape[0], X21.us.shape[1])),
                             X12.us, np.zeros((X12.us.shape[0], X22.us.shape[1]))))
            us2 = np.hstack((np.zeros((X21.us.shape[0], X11.us.shape[1])), X21.us,
                             np.zeros((X22.us.shape[0], X12.us.shape[1])), X22.us))
            us = np.vstack((us1, us2))

            vh1 = np.vstack((X11.vh, X21.vh, np.zeros((X12.vh.shape[0], X11.vh.shape[1])), 
                             np.zeros((X22.vh.shape[0], X21.vh.shape[1]))))
            vh2 = np.vstack((np.zeros((X11.vh.shape[0], X12.vh.shape[1])), 
                             np.zeros((X21.vh.shape[0], X22.vh.shape[1])), X12.vh, X22.vh))
            vh = np.hstack((vh1, vh2))

            u, s, vh = rsvd(us, vh)

                
            tail = [i for i, s0 in enumerate(s) if s0 < tol]
            if len(tail) > 0:
                r = tail[0]
                us = u[:,:r] * s[:r]
                vh = vh[:r].copy()
            else:
                us = u * s
                vh = vh

            return H_matrix('svd', us, vh)
    
    def uTsolve(self, B):
        """
        U.uTsolve(B) solves equation xU = b
        U - upper triangular H-matrix
        B - H_matrix or vector of type numpy.ndarray
        
        Return
        ------
        numpy.ndarray or H_matrix with the same block structure as that of matrix B
        """
        
        tol = 1e-10
        
        if type(B) == np.ndarray:
            if self.type == 'mat':
                return np.linalg.solve(self.A.T, B)

            U11, U12, _, U22 = self.split()
            n = B.shape[0]
            b1, b2 = B[:n//2], B[n//2:]
            
            x1 = U11.uTsolve(b1)
            x2 = U22.uTsolve(b2 - U12.T() @ x1)
            
            return np.concatenate((x1, x2))
        
        if B.type == 'mat':
            return H_matrix('mat', np.linalg.solve(self.mat().T, B.A.T).T)

        if B.type == 'svd' and self.type == 'mat':
            A_inv = np.linalg.inv(self.A)
            return H_matrix('svd', B.us, B.vh @ A_inv)
            
        if self.type == 'split' or B.type == 'split':
            U11, U12, _, U22 = self.split()
            B11, B12, B21, B22 = B.split()

            X11 = U11.uTsolve(B11)
            Tmp = B12.copy()
            Tmp.addmatmul(X11, U12, -1)
            X12 = U22.uTsolve(Tmp)
            X21 = U11.uTsolve(B21)
            Tmp = B22.copy()
            Tmp.addmatmul(X21, U12, -1)
            X22 = U22.uTsolve(Tmp)

        if B.type == 'split':
            return H_matrix('split', X11, X12, X21, X22)
        
        if B.type == 'svd':
            us1 = np.hstack((X11.us, np.zeros((X11.us.shape[0], X21.us.shape[1])),
                             X12.us, np.zeros((X12.us.shape[0], X22.us.shape[1]))))
            us2 = np.hstack((np.zeros((X21.us.shape[0], X11.us.shape[1])), X21.us,
                             np.zeros((X22.us.shape[0], X12.us.shape[1])), X22.us))
            us = np.vstack((us1, us2))

            vh1 = np.vstack((X11.vh, X21.vh, np.zeros((X12.vh.shape[0], X11.vh.shape[1])), 
                             np.zeros((X22.vh.shape[0], X21.vh.shape[1]))))
            vh2 = np.vstack((np.zeros((X11.vh.shape[0], X12.vh.shape[1])), 
                             np.zeros((X21.vh.shape[0], X22.vh.shape[1])), X12.vh, X22.vh))
            vh = np.hstack((vh1, vh2))

            u, s, vh = rsvd(us, vh)

                
            tail = [i for i, s0 in enumerate(s) if s0 < tol]
            if len(tail) > 0:
                r = tail[0]
                us = u[:,:r] * s[:r]
                vh = vh[:r].copy()
            else:
                us = u * s
                vh = vh

            return H_matrix('svd', us, vh)
            
    
    def T(self):
        """
        Create transpose H_matrix
        """
        if self.type == 'mat':
            return H_matrix('mat', self.A.T)
        
        if self.type == 'svd':
            return H_matrix('svd', self.vh.T, self.us.T)
        
        if self.type == 'split':
            return H_matrix('split', self.A11.T(), self.A21.T(), self.A12.T(), self.A22.T())
            
            
    def mem(self):
        """
        Returns the number of stored elements of type double
        """
        if self.type == 'mat':
            return self.A.shape[0] * self.A.shape[1]
        
        if self.type == 'svd':
            return self.us.shape[0] * self.us.shape[1] + self.vh.shape[0] * self.vh.shape[1]
        
        if self.type == 'split':
            return self.A11.mem() + self.A21.mem() + self.A12.mem() + self.A22.mem()
        
        
    def zeros_like(self):
        """
        Create zero H-matrix with the same block structure as that of the matrix 'self'
        """
        if self.type == 'mat':
            return H_matrix('mat', np.zeros_like(self.A))
        
        if self.type == 'svd':
            return H_matrix('svd', self.us[:,:0].copy(), self.vh[:0].copy())
        
        if self.type == 'split':
            return H_matrix('split', self.A11.zeros_like(), self.A12.zeros_like(), 
                            self.A21.zeros_like(), self.A22.zeros_like())
        
        
    def addmatmul(self, A, B, alpha=1):
        """
        self += alpha * A @ B
        
        self, A, B : H-matrixs
        alpha : float
        return : None
        """
        tol = 1e-10
            
        if self.type == 'mat':
            self.A += alpha * A.mat() @ B.mat()
        
        if self.type == 'svd':
            if A.type == 'split' or B.type == 'split':
                S11, S12, S21, S22 = self.split()
                A11, A12, A21, A22 = A.split()
                B11, B12, B21, B22 = B.split()
                
                S11.addmatmul(A11, B11, alpha)
                S11.addmatmul(A12, B21, alpha)
                S12.addmatmul(A11, B12, alpha)
                S12.addmatmul(A12, B22, alpha)
                S21.addmatmul(A21, B11, alpha)
                S21.addmatmul(A22, B21, alpha)
                S22.addmatmul(A21, B12, alpha)
                S22.addmatmul(A22, B22, alpha)
                
                us1 = np.hstack((S11.us, np.zeros((S11.us.shape[0], S21.us.shape[1])),
                                 S12.us, np.zeros((S12.us.shape[0], S22.us.shape[1]))))
                us2 = np.hstack((np.zeros((S21.us.shape[0], S11.us.shape[1])), S21.us,
                                 np.zeros((S22.us.shape[0], S12.us.shape[1])), S22.us))
                us = np.vstack((us1, us2))
                
                vh1 = np.vstack((S11.vh, S21.vh, np.zeros((S12.vh.shape[0], S11.vh.shape[1])), 
                                 np.zeros((S22.vh.shape[0], S21.vh.shape[1]))))
                vh2 = np.vstack((np.zeros((S11.vh.shape[0], S12.vh.shape[1])), 
                                 np.zeros((S21.vh.shape[0], S22.vh.shape[1])), S12.vh, S22.vh))
                vh = np.hstack((vh1, vh2))

                u, s, vh = rsvd(us, vh)
                tail = [i for i, s0 in enumerate(s) if s0 < tol]
                if len(tail) > 0:
                    r = tail[0]
                    self.us = u[:,:r] * s[:r]
                    self.vh = vh[:r].copy()
                else:
                    self.us = u * s
                    self.vh = vh
            
            if A.type == 'mat' or B.type == 'mat':
                S = self.mat() + alpha * A.mat() @ B.mat()
                u, s, vh = np.linalg.svd(S, full_matrices=False)

                tail = [i for i, s1 in enumerate(s) if s1 < tol]
                if len(tail) > 0:
                    r = tail[0]
                    self.us = u[:,:r] * s[:r]
                    self.vh = vh[:r].copy()
                else:
                    self.us = u * s
                    self.vh = vh
                    
            if A.type == 'svd' and B.type == 'svd':
                us = np.hstack((self.us, alpha * A.us @ (A.vh @ B.us)))
                vh = np.vstack((self.vh, B.vh))

                u, s, vh = rsvd(us, vh)
                tail = [i for i, s0 in enumerate(s) if s0 < tol]
                if len(tail) > 0:
                    r = tail[0]
                    self.us = u[:,:r] * s[:r]
                    self.vh = vh[:r].copy()
                else:
                    self.us = u * s
                    self.vh = vh
                
        
        if self.type == 'split':
            A11, A12, A21, A22 = A.split()
            B11, B12, B21, B22 = B.split()
            self.A11.addmatmul(A11, B11, alpha)
            self.A11.addmatmul(A12, B21, alpha)
            self.A12.addmatmul(A11, B12, alpha)
            self.A12.addmatmul(A12, B22, alpha)
            self.A21.addmatmul(A21, B11, alpha)
            self.A21.addmatmul(A22, B21, alpha)
            self.A22.addmatmul(A21, B12, alpha)
            self.A22.addmatmul(A22, B22, alpha)
            
        return

        
    def inv(self):
        """
        Create invers H_matrix with the same block structure
        """
        if self.type == 'mat':
            A_inv = np.linalg.inv(self.A)
            return H_matrix('mat', A_inv)

        if self.type == 'split':
            A11_inv = self.A11.inv()

            C = self.A22.copy()
            C.addmatmul(self.A21, A11_inv @ self.A12, -1)
            C_inv = C.inv()

            I12 = self.A12.zeros_like()
            I12.addmatmul(A11_inv @ self.A12, C_inv, -1)
            I21 = self.A21.zeros_like()
            I21.addmatmul(C_inv @ self.A21, A11_inv, -1)

            Tmp = I12 @ self.A21
            I11 = A11_inv.copy()
            I11.addmatmul(Tmp, A11_inv, -1)
            I22 = C_inv

            return H_matrix('split', I11, I12, I21, I22)

