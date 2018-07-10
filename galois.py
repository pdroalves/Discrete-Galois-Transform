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
                "Weird p! Received " + str(p)
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

        return GaloisInteger(
            (self.re * b.re - self.imag * b.imag) % self.p,
            (self.re * b.imag + self.imag * b.re) % self.p
            )

    def __rmul__(self, b):
        b = self.__valid_operand(b)

        return GaloisInteger(
            (self.re * b.re - self.imag * b.imag) % self.p,
            (self.re * b.imag + self.imag * b.re) % self.p
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