#!/usr/env/python3
#coding: utf-8

def egcd(a, b):
    if a == 0:
        return (b, 0, 1)
    else:
        g, y, x = egcd(b % a, a)
        return (g, x - (b // a) * y, y)

def modinv(a, m):
    a = a % m
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
        assert type(re) == int
        assert type(imag) == int

        self.re = re
        self.imag = imag

        if type(p) == int:
            self.p = p
        elif p is not None:
            raise Exception(
                "Weird p! Received " + str(p)
                )

    def conjugate(self):
        return GaloisInteger(self.re, -self.imag)

    def __valid_operand(self, b):
        if type(b) == int:
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

    # About divisions and remainders:
    # https://math.stackexchange.com/questions/889809/calculating-the-reminder-when-dividing-complex-numbers
    def __floordiv__(self, b):
        if b.imag == 0:
            b = b.re
        if type(b) == int:
            assert b != 0

            return GaloisInteger(
                self.re // b,
                self.imag // b
                )
        else:
            assert isinstance(b, GaloisInteger)

            # We don't want to reduce before the rounding
            s1 = self.re * b.conjugate().re
            s2 = self.imag * b.conjugate().imag
            s3 = (self.re + self.imag) * (b.conjugate().re + b.conjugate().imag) 

            num = GaloisInteger(
                (s1 - s2),
                (s3 - s1 - s2)
                )

            s1 = b.re * b.conjugate().re
            s2 = b.imag * b.conjugate().imag

            den = (s1 - s2)

            return GaloisInteger(
                int(round(num.re / den)),
                int(round(num.imag / den))
                )

    def __mod__(self, b):
        if type(b) == int:
            assert b != 0

            return GaloisInteger(
                int(self.re % b),
                int(self.imag % b)
                )
        else:
            assert isinstance(b, GaloisInteger)

            return self - (self // b) * b


#############################################################$$$$$$$$$$$$$$$$$$
# nthroot
#
from random import randint
def mul(a, b):
    # Non modular multiplication
    # 
    # https://stackoverflow.com/questions/19621686/complex-numbers-product-using-only-three-multiplications
    # 
    # S1=ac,S2=bd, and S3=(a+b)(c+d). Now you can compute the results as 
    # A=S1−S2 and B=S3−S1−S2.
    # 
    s1 = a.re * b.re
    s2 = a.imag * b.imag
    s3 = (a.re + a.imag) * (b.re + b.imag) 

    return GaloisInteger(
        (s1 - s2),
        (s3 - s1 - s2) 
        )

def get_generator(f, p):
    # In number theory, given an integer a and a positive integer n with gcd(a,n) = 1,
    # the multiplicative order of a modulo n is the smallest positive integer k with 
    # a^k \equiv 1 mod n
    # 
    for _ in range(10**5):
        a = GaloisInteger(randint(0, p), randint(0, p)) % f
        # assert egcd(a, p)[0] == 1
        # for i in range(1, p):
        i = p-1
        # print("Testing %s^%d: %s" % (a, i, pow(a, i)))
        if pow(a, i) % f == 1:
            return a
    raise Exception("Couln't find a generator")

def brute_force_modinv(x, f, p):
    for i in range(10**8):
        y = GaloisInteger(randint(0, p), randint(0, p)) % f
        # print("%d) testing %s" % (i, y))
        if x * y % f == 1:
            return y
    print("Failure!")

def gi_modinv(u, p):
    a = u.re
    b = u.imag

    yInv = -b - a * a * modinv(b, p)
    if yInv % p == 0:
        raise Exception("Linear system without a solution: 0 == 1.")
    y = modinv(yInv, p)
    x = modinv(a, p)*(1 + b * y)

    return GaloisInteger(x, y) % p

def nthroot(n, p): 
    f0 = GaloisInteger(65536,  4294967295)
    f1 = GaloisInteger(65536, -4294967295)
    invf0Modf1 = GaloisInteger(0,2147483648)
    invf1Modf0 = GaloisInteger(0,-2147483648)
    assert f0*invf0Modf1 % f1 == 1
    assert f1*invf1Modf0 % f0 == 1

    while 1 == 1:
        g0 = get_generator(f0, p)
        kp0 = pow(g0, (p-1)//(4*n)) % f0

        g1 = get_generator(f1, p)
        kp1 = pow(g1, (p-1)//(4*n)) % f1

        result = (f1 * (invf1Modf0 * kp0 % f0) + f0 * (invf0Modf1 * kp1 % f1)) % p

        # If result == GaloisInteger(0, 1) then we found a nthroot of i
        # Otherwise, try another pair of generators 
        if pow(result, n) == GaloisInteger(0, 1):
            return result

# Use this to generate nthroots and invnthroots
# 
# nthroots = {}
# invnthroots = {}
# for i in range(2, 17):
#     n = 2**i
#     nthroots[n] = nthroot(n, p)
#     invnthroots[n] = gi_modinv(nthroots[n], p)
#     assert nthroots[n]*invnthroots[n] == 1