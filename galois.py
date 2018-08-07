#!/usr/env/python
#coding: utf-8

def egcd(a, b):
    if a == 0:
        return (b, 0, 1)
    else:
        g, y, x = egcd(b % a, a)
        return (g, x - (b // a) * y, y)

def modinv(a, m):
    g, x, y = egcd(a, m)
    if g != 1:
        raise Exception('modular inverse does not exist')
    else:
        return x % m

#############################################################$$$$$$$$$$$$$$$$$$
# Discrete Galois Transform
#

# Galois arithmetic
class GaloisInteger:
    re = 0
    imag = 0
    p = 0xFFFFFFFF00000001 # 2**64 - 2**32 + 1

    def __init__(self, re, imag = 0, p = None):
        assert type(re) in (int, long)
        assert type(imag) in (int, long)

        self.re = re
        self.imag = imag

        if type(p) in (int, long):
            self.p = p
        elif p is not None:
            raise Exception(
                "Weird pq! Received " + str(p)
                )

    def __valid_operand(self, b):
        if type(b) in (int, long):
            b = GaloisInteger(b)
        elif type(b) == float:
            raise Exception(
                "Error! We shouldn't be dealing with floats. Found " + str(b)
                )
        return b

    def __repr__(self):
        return "%d + i%d" % (self.re, self.imag)

    def __eq__(self, b):
        b = self.__valid_operand(b)

        return self.re == b.re and self.imag == b.imag

    # Galois add
    def __add__(self, b):
        b = self.__valid_operand(b)

        return GaloisInteger(
            (self.re + b.re) % self.p,
            (self.imag + b.imag) % self.p
            )

    def __radd__(self, b):
        b = self.__valid_operand(b)

        return GaloisInteger(
            (self.re + b.re) % self.p,
            (self.imag + b.imag) % self.p
            )

    # Galois sub
    def __sub__(self, b):
        b = self.__valid_operand(b)

        return GaloisInteger(
            (self.re - b.re) % self.p,
            (self.imag - b.imag) % self.p
            )

    def __rsub__(self, b):
        b = self.__valid_operand(b)

        return GaloisInteger(
            (self.re - b.re) % self.p,
            (self.imag - b.imag) % self.p
            )
    # Galois mul
    def __mul__(self, b):
        b = self.__valid_operand(b)
        # https://stackoverflow.com/questions/19621686/complex-numbers-product-using-only-three-multiplications
        # 
        # S1=ac,S2=bd, and S3=(a+b)(c+d). Now you can compute the results as 
        # A=S1−S2 and B=S3−S1−S2.
        # 
        s1 = self.re * b.re
        s2 = self.imag * b.imag
        s3 = (self.re + self.imag) * (b.re + b.imag) 

        return GaloisInteger(
            (s1 - s2) % self.p,
            (s3 - s1 - s2) % self.p
            )

    def __rmul__(self, b):
        b = self.__valid_operand(b)

        s1 = self.re * b.re
        s2 = self.imag * b.imag
        s3 = (self.re + self.imag) * (b.re + b.imag) 

        return GaloisInteger(
            (s1 - s2) % self.p,
            (s3 - s1 - s2) % self.p
            )

    def __pow__(self, b):
        # Square and multiply
        if(b == 0):
            return GaloisInteger(1)

        exp = bin(b)[3:]
        value = self

        for i in range(len(exp)):
            value = value * value
            if(exp[i:i+1] == '1'):
                value = value * self
        return value

    def __div__(self, b):
        assert type(b) in [int, long]
        assert b != 0

        return GaloisInteger(
            self.re / b,
            self.imag / b
            )


#############################################################$$$$$$$$$$$$$$$$$$
# nthroot
#
from random import randint

def get_mod_order(p):
    # In number theory, given an integer a and a positive integer n with gcd(a,n) = 1,
    # the multiplicative order of a modulo n is the smallest positive integer k with 
    # a^k \equiv 1 mod n
    # 
    for _ in xrange(10**5):
        a = GaloisInteger(randint(0, p), randint(0, p))
        assert egcd(a, p)[0] == 1
        for i in xrange(1, p):
            if pow(a, i, p) == 1:
                return i
    raise Exception("Couln't find a generator")

def nthroot(n, p): 
    f0 = GaloisInteger(65536, 4294967295)
    f1 = GaloisInteger(65536, -4294967295)

    g0 = get_mod_order(p)
    g1 = get_mod_order(p)