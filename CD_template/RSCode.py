import galois
import numpy as np

class RSCode:
    def __init__(self, m,t,l,m0):
        self.m = m #GF(2^m) field
        self.t = t #Error correction capability
        self.n = 2**m-1 #Code length
        self.k = self.n-2*t #Information length
        self.l = l #Shortened information length (-> shortened code length = l+n-k)
        self.m0 = m0 #m0 of the Reed-Solomon code, determines first root of generator
        
        self.g = self.makeGenerator(m,t,m0) # generator polynomial represented by a galois.Poly variable

    def encode(self,msg):
        # Systematically encodes information words using the Reed-Solomon code
        # Input:
        #  -msg: a 2D array of galois.GF elements, every row corresponds with a GF(2^m) information word of length self.l
        # Output:
        #  -code: a 2D array of galois.GF elements, every row contains a GF(2^m) codeword corresponding to systematic Reed-Solomon coding of the corresponding information word
        assert np.shape(msg)[1] == self.l, 'the number of columns must be equal to self.l'
        assert type(msg) is galois.GF(2**self.m) , 'each element of msg  must be a galois.GF element'

        #insert your code here

        assert np.shape(code)[1] == self.l+self.n-self.k , 'the number of columns must be equal to self.l+self.n-self.k'
        assert type(code) is galois.GF(2**self.m) , 'each element of code  must be a galois.GF element'
        return code

    def decode(self,code):
        # Decode Reed-Solomon codes
        # Input:
        #  -code: a 2D array of galois.GF elements, every row contains a GF(2^m) codeword of length self.l+self.n-self.k
        # Output:
        #  -decoded: a 2D array of galois.GF elements, every row contains a GF(2^m) information word corresponding to decoding of the corresponding Reed-Solomon codeword
        #  -nERR: 1D numpy array containing the number of corrected symbols for every codeword, -1 if error correction failed
        assert np.shape(code)[1] == self.l+self.n-self.k , 'the number of columns must be equal to self.l+self.n-self.k'
        assert type(code) is galois.GF(2**self.m) , 'each element of code  must be a galois.GF element'

        GF = galois.GF(2**self.m)
        alpha = GF.primitive_element
        n_sym = self.n - self.k
        n_short = self.l + n_sym
        n_pad = self.n - n_short

        def poly_eval_asc(coeffs, x):
            # Evaluate polynomial with ascending coefficients: c0 + c1*x + ...
            y = GF(0)
            x_pow = GF(1)
            for c in coeffs:
                y += c * x_pow
                x_pow *= x
            return y

        def compute_syndromes(r_full):
            synd = GF.Zeros(n_sym)
            for j in range(n_sym):
                root = alpha ** (self.m0 + j)
                acc = GF(0)
                for idx, val in enumerate(r_full):
                    acc += val * (root ** (self.n - 1 - idx))
                synd[j] = acc
            return synd

        def berlekamp_massey(synd):
            C = GF.Zeros(n_sym + 1)
            B = GF.Zeros(n_sym + 1)
            C[0] = GF(1)
            B[0] = GF(1)
            L = 0
            m = 1
            b = GF(1)

            for n_idx in range(n_sym):
                d = synd[n_idx]
                for i in range(1, L + 1):
                    if n_idx - i >= 0:
                        d += C[i] * synd[n_idx - i]

                if d == 0:
                    m += 1
                    continue

                T = C.copy()
                coef = d / b
                for i in range(0, n_sym + 1 - m):
                    C[i + m] -= coef * B[i]

                if 2 * L <= n_idx:
                    L = n_idx + 1 - L
                    B = T
                    b = d
                    m = 1
                else:
                    m += 1

            return C, L

        n_rows = np.shape(code)[0]
        decoded = GF.Zeros((n_rows, self.l))
        nERR = np.zeros(n_rows, dtype=int)

        for r in range(n_rows):
            r_short = GF(code[r, :])
            r_full = np.concatenate((GF.Zeros(n_pad), r_short))

            synd = compute_syndromes(r_full)
            if np.all(synd == 0):
                corrected_short = r_full[n_pad:]
                decoded[r, :] = corrected_short[:self.l]
                nERR[r] = 0
                continue

            Lambda, L = berlekamp_massey(synd)
            if L == 0 or L > self.t:
                decoded[r, :] = r_short[:self.l]
                nERR[r] = -1
                continue

            # Chien search: find positions i (from right, i=0 is x^0 term) s.t. Lambda(alpha^-i)=0.
            err_locs = []
            for i in range(n_short):
                if poly_eval_asc(Lambda[:L + 1], alpha ** (-i)) == 0:
                    err_locs.append(i)

            if len(err_locs) != L or len(err_locs) > self.t:
                decoded[r, :] = r_short[:self.l]
                nERR[r] = -1
                continue

            # Error evaluator Omega(x) = (S(x) * Lambda(x)) mod x^(n-k)
            omega = GF.Zeros(n_sym)
            for i in range(n_sym):
                for j in range(min(i, L) + 1):
                    omega[i] += synd[i - j] * Lambda[j]

            # Formal derivative Lambda'(x)
            Lambda_deriv = GF.Zeros(max(L, 1))
            for i in range(1, L + 1):
                if i % 2 == 1:
                    Lambda_deriv[i - 1] = Lambda[i]

            r_corr = r_full.copy()
            decode_failed = False
            for i in err_locs:
                x_inv = alpha ** (-i)
                Xi = alpha ** i
                num = poly_eval_asc(omega, x_inv)
                den = poly_eval_asc(Lambda_deriv, x_inv)
                if den == 0:
                    # In characteristic-2 fields, Lambda'(x) can be identically zero (e.g., repeated roots),
                    # which makes Forney's denominator vanish and decoding with errors-only must fail.
                    decode_failed = True
                    break

                # Forney for consecutive roots alpha^m0 .. alpha^(m0+n-k-1)
                err = (Xi ** (1 - self.m0)) * num / den
                # Descending convention: position x^i maps to r_full index (self.n - 1 - i).
                arr_idx = self.n - 1 - i
                r_corr[arr_idx] -= err

            if decode_failed:
                decoded[r, :] = r_short[:self.l]
                nERR[r] = -1
                continue

            if not np.all(compute_syndromes(r_corr) == 0):
                decoded[r, :] = r_short[:self.l]
                nERR[r] = -1
                continue

            corrected_short = r_corr[n_pad:]
            decoded[r, :] = corrected_short[:self.l]
            nERR[r] = len(err_locs)

        assert np.shape(decoded)[1] == self.l, 'the number of columns must be equal to self.l'
        assert type(decoded) is galois.GF(2**self.m) , 'each element of decoded  must be a galois.GF element'
        assert type(nERR) is np.ndarray and len(np.shape(nERR))==1 , 'nERR must be a 1D numpy array'

        return (decoded,nERR)




    @staticmethod
    def makeGenerator(m, t, m0):
        # Generate the Reed-Solomon generator polynomial with error correcting capability t over GF(2^m)
        # Input:
        #  -m: order of the galois field is 2^m
        #  -t: error correction capability of the Reed-Solomon code
        #  -m0: determines the first root of the generator polynomial
        # Output:
        #  -generator: generator polynomial represented by a galois.Poly variable

        #insert your code here

        assert type(generator) == type(galois.Poly([0],field=galois.GF(2**m))), 'generator must be a galois.Poly object'
        return generator

    @staticmethod
    def test():
        # function that illustrates how the other code of this class can be tested
        m0 = 1 # Also test with other values of m0!
        m=8
        t=5
        l=10
        rs = RSCode(m,t,l,m0) # Construct the RSCode object
        p=2
        prim_poly=galois.primitive_poly(p,m)
        galois_field=galois.GF(p**m,prim_poly)


        msg = galois_field(np.random.randint(0,2**8-1,(5,10))) # Generate a random message of 5 information words

        code = rs.encode(msg) # Encode this message

        # Introduce errors
        code[1,[2, 17]] = code[1,[4, 17]]+galois_field(1)
        code[2,7] = 0;
        code[3,[3, 1, 18, 19, 5]] = np.random.randint(0,2**8-1,(1,5))
        code[4,[3, 1, 18, 19, 5, 12]] = np.random.randint(0,2**8-1,(1,6))


        [decoded,nERR] = rs.decode(code) # Decode


        print(nERR)
        assert((decoded[0:4,:] == msg[0:4,:]).all())
        pass