#!/usr/env/python
#coding: utf-8
import unittest
from math import cos, sin, pi
from galois import GaloisInteger, modinv

p = 0xFFFFFFFF00000001 # 2**64 - 2**32 + 1

# Apply DGT
def dgt(x):
    n = len(x)
    x = [a if isinstance(a, GaloisInteger) else GaloisInteger(a) for a in x]

    # Params
    r = 7 ## Primitive root of p

    assert (p-1)%n == 0
    k = (p-1)/n

    g = pow(r, k, p)
    assert pow(g, n, p) == 1
    #

    X = []

    for k in range(n):
        X.append(
            sum(
                [x[j]*modinv(pow(g, j*k, p), p) for j in range(n)]
                )
            )
    return X

# Apply iDGT
def idgt(X):
    n = len(X)
    invN = modinv(n, p)

    # Params
    r = 7 ## Primitive root of p

    assert (p-1)%n == 0
    k = (p-1)/n

    g = pow(r, k, p)
    assert pow(g, n, p) == 1
    #

    x = []
    for k in range(n):
        x.append(
            invN*sum(
                [X[j]*pow(g, j*k, p) for j in range(n)]
                )
            )
    return x

#############################################################$$$$$$$$$$$$$$$$$$
# Polynomial mul
nthroots = {
    256:GaloisInteger(1100507988529617178, 13061373484646814047),
    512:GaloisInteger(5809945479226292735, 4344400649288295733),
    1024:GaloisInteger(1973388244086427726, 10274180581472164772),
    2048:GaloisInteger(2796647310976247644, 10276259027288899473),
    4096:GaloisInteger(1838446843991, 11906871093314535013)
}

invNthroots = {
    # 256:GaloisInteger(16074635847813643139, 1012656084342873654)
    256:GaloisInteger(1012656084342873654, 2372108221600941182)
}

# Multpĺication inside DGT's domain
def dgt_mul(a, b):
    assert len(a) == len(b)
    N = len(a)

    # Initialize
    a_folded = [GaloisInteger(x, y) for x, y in zip(a[:N/2], a[N/2:])]
    b_folded = [GaloisInteger(x, y) for x, y in zip(b[:N/2], b[N/2:])]

    # Compute h
    assert pow(nthroots[N/2], N/2) == GaloisInteger(0, 1)
    # assert nthroots[N/2] * invNthroots[N/2] == GaloisInteger(0, 1)
    assert nthroots[N/2] * invNthroots[N/2] == GaloisInteger(1)

    # Twist the folded signals
    a_h = [a_folded[j] * pow(nthroots[N/2], j) for j in range(N / 2)]
    b_h = [b_folded[j] * pow(nthroots[N/2], j) for j in range(N / 2)]

    # Compute n/2 DGT
    a_dgt = dgt(a_h)
    assert idgt(a_dgt) == a_h
    b_dgt = dgt(b_h)
    assert idgt(b_dgt) == b_h

    # Point-wise multiplication
    c_dgt = [x * y for x, y in zip(a_dgt, b_dgt)]

    # Compute n/2 IDGT
    c_h = idgt(c_dgt)

    # Remove twisting factors
    c_folded = [c_h[j] * pow(invNthroots[N/2], j) for j in range(N / 2)]

    # Unfold output
    c = [c_folded[j].re for j in range(N/2)] + [c_folded[j].imag for j in range(N/2)]

    return c

# Apply schoolbook polynomial multiplication and reduces by Z_q / <x^N + 1>
def mul(a, b):
    assert len(a) == len(b)
    N = len(a)
    c = [0]*N

    # Mul and reduce
    for i in range(N):
        for j in range(N):
            v = a[i]*b[j]*(-1)**(int((i+j)/float(N)))

            c[(i+j) % N] = (c[(i+j) % N] + v) % p

    return c

#############################################################$$$$$$$$$$$$$$$$$$
# Tests
class TestDGT(unittest.TestCase):

    def test_transformation(self):
        # Verifies if iDGT(DGT(x)) == x
        x = range(512)
        self.assertEqual(idgt(dgt(x)), x)

    def test_mul(self):
        # Verifies multiplication in DGT's domain
        a = range(512)
        b = range(512)
        self.assertEqual(
            dgt_mul(a, b),
            mul(a, b)
            )

if __name__ == '__main__':
    unittest.main()