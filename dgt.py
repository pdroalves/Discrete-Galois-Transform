#!/usr/env/python3
#coding: utf-8
import unittest
from math import cos, sin, pi
from gaussian import GaussianInteger
from multiprocessing import Pool
from math import log

p = 0xFFFFFFFF00000001 # 2**64 - 2**32 + 1

def egcd(a, b):
    # print("a: %s(%d)\nb: %s(%d)\n" % (a, a.norm(), b, b.norm()))
    if a == 0:
        if type(b) == int:
            return (b, 0, 1)
        else:
            assert isinstance(b, GaussianInteger)
            return (b, GaussianInteger(0), GaussianInteger(1))
    else:
        g, x, y = egcd(b % a, a)
        # print(g.norm(), x.norm(), y.norm())
        return (g, y - (b // a) * x, x)

# x = mulinv(b) mod n, (x * b) % n == 1
def modinv(b, n):
    g, x, _ = egcd(b, n)

    if g != 1:
        raise Exception('modular inverse does not exist (found %s)' % g)
    else:
        return x

def is_power2(n):
    n = int(n)
    while n>1:
        if n//2 != n//2.0: #compare integer division to float division
           return False
        n = n//2
    return True

def is_power4(n):
    n = int(n)
    while n>1:
        if n//4 != n//4.0: #compare integer division to float division
           return False
        n = n//4
    return True

if p == 9223372036801560577:
    nthroots = {
        16: GaussianInteger(2711615452560896026, 1702263320480097066),
    }

    invNthroots = {
        16: GaussianInteger(5564493748148731354, 2094114884238307585),
    }
else:
    assert p == 0xFFFFFFFF00000001
    nthroots = {
        4: GaussianInteger(17872535116924321793, 18446744035054843905),
        8: GaussianInteger(18446741870391328801, 18293621682117541889),
        16: GaussianInteger(13835058050988244993, 68719460336),
        32: GaussianInteger(2711615452560896026, 1702263320480097066),
        64: GaussianInteger(5006145799600815370, 13182758589097755684),
        128: GaussianInteger(1139268601170473317, 15214299942841048380),
        256: GaussianInteger(4169533218321950981, 11340503865284752770),
        512: GaussianInteger(1237460157098848423, 590072184190675415),
        1024: GaussianInteger(13631489165933064639, 9250462654091849156),
        2048: GaussianInteger(12452373509881738698, 10493048841432036102),
        4096: GaussianInteger(12694354791372265231, 372075061476485181),
        8192: GaussianInteger(9535633482534078963, 8920239275564743446),
        16384: GaussianInteger(9868966171728904500, 6566969707398269596),
        32768: GaussianInteger(10574165493955537594, 3637150097037633813),
        65536: GaussianInteger(2132094355196445245, 12930307277289315455)
    }

    invNthroots = {
        4: GaussianInteger(34359736320, 17868031517297999873),
        8: GaussianInteger(18311636080627023873, 18446741870391328737),
        16: GaussianInteger(18446739675663041537, 18446462594505048065),
        32: GaussianInteger(9223372049739972605, 9223372049739382781),
        64: GaussianInteger(3985917792403544159, 10871216858344511621),
        128: GaussianInteger(697250266387245748, 7269985899340929883),
        256: GaussianInteger(16440350318199496391, 8259263625103887671),
        512: GaussianInteger(11254465366323603399, 282547220712277683),
        1024: GaussianInteger(4772545667722300316, 8077569763565898552),
        2048: GaussianInteger(13028894352332048345, 9995848711590186809),
        4096: GaussianInteger(11525613835860693, 17335883825168514904),
        8192: GaussianInteger(17414606149056687587, 3916527805974289959),
        16384: GaussianInteger(9801605401783064476, 2749242888134484347),
        32768: GaussianInteger(10469048769509713349, 8715957816394874924),
        65536: GaussianInteger(15132804493885713016, 7997468840100395569)
    }

PROOTS = {
    9223372036801560577:5,
    9223372036746117121:19,
    9223372036743626753:3,
    9223372036737335297:3,
    9223372036731174913:5,
    9223372036717281281:3,
    9223372036703256577:5,
    9223372036696178689:7,
    9223372036668653569:7,
    9223372036639424513:3,
    0xFFFFFFFF00000001:7
}
###############
# Textbook DGT
###############

# Apply DGT
def dgt(x):
    n = len(x)
    x = [a if isinstance(a, GaussianInteger) else GaussianInteger(a) for a in x]

    # Params
    r = PROOTS[p] ## Primitive root of p

    assert (p-1)%n == 0
    k = (p-1)//n

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
    r = PROOTS[p] ## Primitive root of p

    assert (p-1)%n == 0
    k = (p-1)//n

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

##############################
# Gentleman-sande DGT radix-2
##############################

# Apply DGT
def dgt_gentlemansande(x):
    k = len(x)
    assert is_power2(k)
    x = [a if isinstance(a, GaussianInteger) else GaussianInteger(a) for a in x]

    # Params
    r = PROOTS[p] ## Primitive root of p

    assert (p-1)%k == 0
    n = (p-1)//k

    g = pow(r, n, p)
    assert pow(g, k, p) == 1 # k-th primitive root of unity
    gj = [pow(g, j, p) for j in range(k)]

    #
    X = list(x) # Copy because the algorithm is run in-place
    for stride in range(int(log(k,2))):
        m = k // (2<<stride)

        for l in range(k // 2):
            j = l//(k//(2*m))
            a = pow(gj[j], k >> (int(log(k,2)) - stride), p)

            i = j + (l % (k//(2*m)))*2*m
        

            xi = X[i]
            xim = X[i + m]
            # print("m: %d, i: %d, j: %d, jk: %d, a: %d, xi: %s, xim: %s" % (m, i, j, int(log(k,2)) - stride - 1, a, xi, xim))
            X[i] = xi + xim
            X[i + m] = a * (xi - xim)
    return X


def idgt_gentlemansande(x):
    k = len(x)
    assert is_power2(k)
    x = [a if isinstance(a, GaussianInteger) else GaussianInteger(a) for a in x]

    # Params
    r = PROOTS[p] ## Primitive root of p

    assert (p-1)%k == 0
    n = (p-1)//k

    g = pow(r, n, p)
    assert g != 0
    # print "g: %d" % g
    assert pow(g, k, p) == 1 # n-th primitive root of unity
    invgj = [pow(g, (k - j), p) for j in range(k)] # g^-i \equiv g^((k-i) mod k) mod p

    X = list(x) # Copy because the algorithm is run in-place
    m = 1
    for stride in range(int(log(k,2))):
        for l in range(k // 2):
            j = l//(k//(2*m))
            a = pow(invgj[j], k >> (stride + 1), p)
            i = j + (l % (k//(2*m)))*2*m

            xi = X[i]
            xim = X[i + m]
            # print "m: %d, i: %d, j: %d, jk: %d, a: %d, xi: %s, xim: %s" % (m, i, j, int(log(k,2)) - stride - 1, a, xi, xim)

            X[i] = xi + a * xim
            X[i + m] = xi - a * xim
        m = 2 * m
    return [v*modinv(k,p) for v in X]

##############################
# Gentleman-sande DGT radix-4
##############################

# Apply DGT
def dgt_gentlemansande_radix4(x):
    k = len(x)
    assert is_power4(k)
    x = [a if isinstance(a, GaussianInteger) else GaussianInteger(a) for a in x]

    # Params
    r = PROOTS[p] ## Primitive root of p

    assert (p-1)%k == 0
    n = (p-1)//k

    g = pow(r, n, p)
    assert pow(g, k, p) == 1 # k-th primitive root of unity
    gj = [pow(g, j, p) for j in range(k)]

    #
    X = list(x) # Copy because the algorithm is run in-place
    for stride in range(int(log(k,4))):
        m = k // (4<<stride)

        for l in range(k // 4):
            j = l//(k//(4*m))
            a = pow(gj[j], k >> (int(log(k,4)) - stride), p)

            i = j + (l % (k//(4*m)))*4*m

            xim0 = X[i + 0 * m]
            xim1 = X[i + 1 * m]
            xim2 = X[i + 2 * m]
            xim3 = X[i + 3 * m]
            # print("m: %d, i: %d, j: %d, jk: %d, a: %d, xi: %s, xim: %s" % (m, i, j, int(log(k,2)) - stride - 1, a, xi, xim))
            X[i] = xi + xim
            X[i + m] = a * (xi - xim)
    return X


def idgt_gentlemansande_radix4(x):
    k = len(x)
    assert is_power4(k)
    x = [a if isinstance(a, GaussianInteger) else GaussianInteger(a) for a in x]

    # Params
    r = PROOTS[p] ## Primitive root of p

    assert (p-1)%k == 0
    n = (p-1)//k

    g = pow(r, n, p)
    assert g != 0
    # print "g: %d" % g
    assert pow(g, k, p) == 1 # n-th primitive root of unity
    invgj = [pow(g, (k - j), p) for j in range(k)] # g^-i \equiv g^((k-i) mod k) mod p

    X = list(x) # Copy because the algorithm is run in-place
    m = 1
    for stride in range(int(log(k,2))):
        for l in range(k // 2):
            j = l//(k//(2*m))
            a = pow(invgj[j], k >> (stride + 1), p)
            i = j + (l % (k//(2*m)))*2*m

            xi = X[i]
            xim = X[i + m]
            # print "m: %d, i: %d, j: %d, jk: %d, a: %d, xi: %s, xim: %s" % (m, i, j, int(log(k,2)) - stride - 1, a, xi, xim)

            X[i] = xi + a * xim
            X[i + m] = xi - a * xim
        m = 2 * m
    return [v*modinv(k,p) for v in X]

#############################################################$$$$$$$$$$$$$$$$$$
# Polynomial mul

# Multpĺication inside DGT's domain
def dgt_mul(a, b):
    assert len(a) == len(b)
    N = len(a)

    # Initialize
    a_folded = [GaussianInteger(x, y) for x, y in zip(a[:N//2], a[N//2:])]
    b_folded = [GaussianInteger(x, y) for x, y in zip(b[:N//2], b[N//2:])]

    # Compute h
    assert pow(nthroots[N//2], N//2) == GaussianInteger(0, 1)
    assert nthroots[N//2] * invNthroots[N//2] == GaussianInteger(1)

    # Twist the folded signals
    a_h = [a_folded[j] * pow(nthroots[N//2], j) for j in range(N // 2)]
    b_h = [b_folded[j] * pow(nthroots[N//2], j) for j in range(N // 2)]

    # Compute n//2 DGT
    a_dgt = dgt(a_h)
    assert idgt(a_dgt) == a_h
    b_dgt = dgt(b_h)
    assert idgt(b_dgt) == b_h

    # Point-wise multiplication
    c_dgt = [x * y for x, y in zip(a_dgt, b_dgt)]

    # Compute n//2 IDGT
    c_h = idgt(c_dgt)

    # Remove twisting factors
    c_folded = [c_h[j] * pow(invNthroots[N//2], j) for j in range(N // 2)]

    # Unfold output
    c = [c_folded[j].re for j in range(N//2)] + [c_folded[j].imag for j in range(N//2)]

    return c

# Multpĺication inside DGT's domain
def dgt_gentlemansande_mul(a, b):
    assert len(a) == len(b)
    N = len(a)

    # Initialize
    a_folded = [GaussianInteger(x, y) for x, y in zip(a[:N//2], a[N//2:])]
    b_folded = [GaussianInteger(x, y) for x, y in zip(b[:N//2], b[N//2:])]

    # Compute h
    assert pow(nthroots[N//2], N//2) == GaussianInteger(0, 1)
    assert nthroots[N//2] * invNthroots[N//2] == GaussianInteger(1)

    # Twist the folded signals
    a_h = [a_folded[j] * pow(nthroots[N//2], j) for j in range(N // 2)]
    b_h = [b_folded[j] * pow(nthroots[N//2], j) for j in range(N // 2)]

    # Compute n//2 DGT
    a_dgt = dgt_gentlemansande(a_h)
    assert idgt_gentlemansande(a_dgt) == a_h
    b_dgt = dgt_gentlemansande(b_h)
    assert idgt_gentlemansande(b_dgt) == b_h

    # Point-wise multiplication
    c_dgt = [x * y for x, y in zip(a_dgt, b_dgt)]

    # Compute n//2 IDGT
    c_h = idgt_gentlemansande(c_dgt)

    # Remove twisting factors
    c_folded = [c_h[j] * pow(invNthroots[N//2], j) for j in range(N // 2)]

    # Unfold output
    c = [c_folded[j].re for j in range(N//2)] + [c_folded[j].imag for j in range(N//2)]

    return c

# Apply schoolbook polynomial multiplication and reduces by Z_q // <x^N + 1>
def mul(a, b):
    assert len(a) == len(b)
    N = len(a)
    c = [0]*N

    # Mul and reduce
    for i in range(N):
        for j in range(N):
            v = a[i]*b[j]*(-1)**(int((i+j)//float(N)))

            c[(i+j) % N] = (c[(i+j) % N] + v) % p

    return c

# Multpĺication inside DGT's domain by an integer
def dgt_gentlemansande_mulint(a, b):
    assert type(b) == int
    N = len(a)

    # Initialize
    a_folded = [GaussianInteger(x, y) for x, y in zip(a[:N//2], a[N//2:])]

    # Compute h
    assert pow(nthroots[N//2], N//2) == GaussianInteger(0, 1)
    assert nthroots[N//2] * invNthroots[N//2] == GaussianInteger(1)

    # Twist the folded signals
    a_h = [a_folded[j] * pow(nthroots[N//2], j) for j in range(N // 2)]

    # Compute n//2 DGT
    a_dgt = dgt_gentlemansande(a_h)
    assert idgt_gentlemansande(a_dgt) == a_h

    # Point-wise multiplication
    c_dgt = [x * (b % p) for x in a_dgt]

    # Compute n//2 IDGT
    c_h = idgt_gentlemansande(c_dgt)

    # Remove twisting factors
    c_folded = [c_h[j] * pow(invNthroots[N//2], j) for j in range(N // 2)]

    # Unfold output
    c = [c_folded[j].re for j in range(N//2)] + [c_folded[j].imag for j in range(N//2)]

    return c

# Apply schoolbook polynomial multiplication and reduces by Z_q // <x^N + 1>
def mulint(a, b):
    assert type(b) == int
    N = len(a)
    c = [x * b % p for x in a]
    return c



# Multpĺication inside DGT's domain
def dgt_gentlemansande_sub(a, b):
    assert len(a) == len(b)
    N = len(a)

    # Initialize
    a_folded = [GaussianInteger(x, y) for x, y in zip(a[:N//2], a[N//2:])]
    b_folded = [GaussianInteger(x, y) for x, y in zip(b[:N//2], b[N//2:])]

    # Compute h
    assert pow(nthroots[N//2], N//2) == GaussianInteger(0, 1)
    # assert nthroots[N//2] * invNthroots[N//2] == GaussianInteger(0, 1)
    assert nthroots[N//2] * invNthroots[N//2] == GaussianInteger(1)

    # Twist the folded signals
    a_h = [a_folded[j] * pow(nthroots[N//2], j) for j in range(N // 2)]
    b_h = [b_folded[j] * pow(nthroots[N//2], j) for j in range(N // 2)]

    # Compute n//2 DGT
    a_dgt = dgt_gentlemansande(a_h)
    assert idgt_gentlemansande(a_dgt) == a_h
    b_dgt = dgt_gentlemansande(b_h)
    assert idgt_gentlemansande(b_dgt) == b_h

    # Point-wise multiplication
    c_dgt = [(x - y) for x, y in zip(a_dgt, b_dgt)]

    # Compute n//2 IDGT
    c_h = idgt_gentlemansande(c_dgt)

    # Remove twisting factors
    c_folded = [c_h[j] * pow(invNthroots[N//2], j) for j in range(N // 2)]

    # Unfold output
    c = [c_folded[j].re for j in range(N//2)] + [c_folded[j].imag for j in range(N//2)]

    return c

#############################################################$$$$$$$$$$$$$$$$$$
# Tests
class TestDGT(unittest.TestCase):

    def test_transformation(self):
        # Verifies if iDGT(DGT(x)) == x
        x = [x for x in range(32)]
        self.assertEqual(idgt(dgt(x)), x)

    def test_mul(self):
        # Verifies multiplication in DGT's domain
        a = [x for x in range(512)]
        b = [x for x in range(512)]
        self.assertEqual(
            dgt_mul(a, b),
            mul(a, b)
            )

class TestDGTGentlemansande(unittest.TestCase):

    def test_transformation(self):
        # Verifies if iDGT(DGT(x)) == x
        x = [x for x in range(32)]
        # print "\n".join([str(y) for y in dgt_gentlemansande(x)])
        self.assertEqual(idgt_gentlemansande(dgt_gentlemansande(x)), x)

    def test_mul(self):
        # Verifies multiplication in DGT's domain
        a = [x for x in range(512)]
        b = [x for x in range(512)]
        self.assertEqual(
            dgt_gentlemansande_mul(a, b),
            mul(a, b)
            )

    def test_mulint(self):
        # Verifies multiplication in DGT's domain
        a = [x for x in range(512)]
        b = p//3

        self.assertEqual(
            dgt_gentlemansande_mulint(a, b),
            mulint(a, b)
            )

    def test_sub(self):
        # Verifies multiplication in DGT's domain
        a = [2 * x for x in range(512)]
        b = [x for x in range(512)]
        self.assertEqual(
            dgt_gentlemansande_sub(a, b),
            b
            )

if __name__ == '__main__':
    unittest.main()