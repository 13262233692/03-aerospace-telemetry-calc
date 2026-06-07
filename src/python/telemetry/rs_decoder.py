import numpy as np
from typing import List, Tuple

try:
    from telemetry_core import ReedSolomon
    HAS_CPP_EXTENSION = True
except ImportError:
    HAS_CPP_EXTENSION = False


class ReedSolomonDecoder:
    def __init__(self):
        self.RS_BLOCK_LENGTH = 255
        self.RS_DATA_LENGTH = 223
        self.RS_ECC_LENGTH = 32
        self.primitive_poly = 0x11D
        self.gf_size = 256

        if HAS_CPP_EXTENSION:
            self._rs = ReedSolomon()
        else:
            self.gf_exp = [0] * (2 * self.gf_size)
            self.gf_log = [0] * self.gf_size
            self._init_gf()
            self.gen_poly = self._build_generator()

    def _init_gf(self):
        x = 1
        for i in range(self.gf_size - 1):
            self.gf_exp[i] = x
            self.gf_log[x] = i
            x <<= 1
            if x & self.gf_size:
                x ^= self.primitive_poly

        for i in range(self.gf_size - 1, 2 * self.gf_size):
            self.gf_exp[i] = self.gf_exp[i - (self.gf_size - 1)]

    def _gf_mul(self, a, b):
        if a == 0 or b == 0:
            return 0
        return self.gf_exp[self.gf_log[a] + self.gf_log[b]]

    def _gf_div(self, a, b):
        if b == 0:
            return 0
        return self.gf_exp[(self.gf_log[a] - self.gf_log[b] + self.gf_size - 1) % (self.gf_size - 1)]

    def _gf_pow(self, a, power):
        if a == 0:
            return 0
        return self.gf_exp[(self.gf_log[a] * power) % (self.gf_size - 1)]

    def _gf_poly_mul(self, p1, p2):
        result = [0] * (len(p1) + len(p2) - 1)
        for i, c1 in enumerate(p1):
            for j, c2 in enumerate(p2):
                result[i + j] ^= self._gf_mul(c1, c2)
        return result

    def _gf_poly_eval(self, poly, x):
        y = poly[0]
        for c in poly[1:]:
            y = self._gf_mul(y, x) ^ c
        return y

    def _build_generator(self):
        gen = [1]
        for i in range(self.RS_ECC_LENGTH):
            gen = self._gf_poly_mul(gen, [1, self.gf_exp[i]])
        return gen

    def _rs_encode_msg(self, msg):
        gen = self.gen_poly
        msg_out = [0] * (len(msg) + self.RS_ECC_LENGTH)
        msg_out[:len(msg)] = msg

        for i in range(len(msg)):
            coef = msg_out[i]
            if coef != 0:
                for j in range(1, len(gen)):
                    msg_out[i + j] ^= self._gf_mul(gen[j], coef)

        msg_out[:len(msg)] = msg
        return msg_out

    def _berlekamp_massey(self, syndromes):
        n = len(syndromes)
        c = [1]
        b = [1]
        l = 0
        m = 1
        b_mis = 1

        for i in range(n):
            d = syndromes[i]
            for j in range(1, l + 1):
                d ^= self._gf_mul(c[j], syndromes[i - j])

            if d == 0:
                m += 1
            else:
                t = c[:]
                coef = self._gf_div(d, b_mis)
                while len(c) < len(b) + m:
                    c.append(0)
                for j in range(len(b)):
                    c[j + m] ^= self._gf_mul(coef, b[j])

                if 2 * l <= i:
                    l = i + 1 - l
                    b = t
                    b_mis = d
                    m = 1
                else:
                    m += 1

        return c, l

    def _find_errors(self, error_locator, length):
        errors = []
        for i in range(self.gf_size - 1):
            if self._gf_poly_eval(error_locator, self.gf_exp[i]) == 0:
                pos = (self.gf_size - 2) - i
                if pos >= length or pos < 0:
                    return None
                errors.append(pos)
        return errors

    def _solve_error_values(self, syndromes, error_poly_indices, num_errors):
        matrix = []
        for i in range(num_errors):
            row = []
            for j in range(num_errors):
                row.append(self._gf_pow(self.gf_exp[error_poly_indices[j]], i))
            matrix.append(row)

        for col in range(num_errors):
            pivot = matrix[col][col]
            if pivot == 0:
                for row in range(col + 1, num_errors):
                    if matrix[row][col] != 0:
                        matrix[col], matrix[row] = matrix[row], matrix[col]
                        syndromes[col], syndromes[row] = syndromes[row], syndromes[col]
                        pivot = matrix[col][col]
                        break

            inv_pivot = self._gf_div(1, pivot)
            for j in range(col, num_errors):
                matrix[col][j] = self._gf_mul(matrix[col][j], inv_pivot)
            syndromes[col] = self._gf_mul(syndromes[col], inv_pivot)

            for row in range(num_errors):
                if row != col and matrix[row][col] != 0:
                    factor = matrix[row][col]
                    for j in range(col, num_errors):
                        matrix[row][j] ^= self._gf_mul(factor, matrix[col][j])
                    syndromes[row] ^= self._gf_mul(factor, syndromes[col])

        return syndromes[:num_errors]

    def _correct_errors(self, msg_in, syndromes, error_pos):
        msg = msg_in[:]
        num_errors = len(error_pos)

        error_positions = sorted(error_pos)
        error_poly_indices = [(self.gf_size - 2) - pos for pos in error_positions]

        s = syndromes[:num_errors]
        error_values = self._solve_error_values(s, error_poly_indices, num_errors)

        for i, pos in enumerate(error_positions):
            msg[pos] ^= error_values[i]

        return msg

    def _rs_decode_msg(self, msg):
        if len(msg) != self.RS_BLOCK_LENGTH:
            return None, 0

        syndromes = [0] * self.RS_ECC_LENGTH
        for i in range(self.RS_ECC_LENGTH):
            syndromes[i] = self._gf_poly_eval(msg, self.gf_exp[i])

        if max(syndromes) == 0:
            return msg[:self.RS_DATA_LENGTH], 0

        e_loc, error_count = self._berlekamp_massey(syndromes)

        if error_count == 0 or error_count > self.RS_ECC_LENGTH // 2:
            return None, 0

        error_pos = self._find_errors(e_loc, len(msg))

        if error_pos is None or len(error_pos) != error_count:
            return None, 0

        corrected = self._correct_errors(msg, syndromes, error_pos)

        return corrected[:self.RS_DATA_LENGTH], len(error_pos)

    def encode(self, data: bytes) -> bytes:
        if HAS_CPP_EXTENSION:
            result = self._rs.encode(list(data))
            return bytes(result)

        if len(data) > self.RS_DATA_LENGTH:
            raise ValueError(f"Data too long, max {self.RS_DATA_LENGTH} bytes")

        msg = list(data) + [0] * (self.RS_DATA_LENGTH - len(data))
        encoded = self._rs_encode_msg(msg)
        return bytes(encoded)

    def decode(self, data_with_ecc: bytes) -> Tuple[bytes, int]:
        if HAS_CPP_EXTENSION:
            result, corrected = self._rs.decode(list(data_with_ecc))
            return bytes(result), corrected

        decoded, count = self._rs_decode_msg(list(data_with_ecc))
        if decoded is None:
            return b'', 0
        return bytes(decoded), count

    def decode_burst(self, blocks: List[bytes]) -> Tuple[List[bytes], List[int]]:
        if HAS_CPP_EXTENSION:
            block_lists = [list(b) for b in blocks]
            results, corrected = self._rs.decode_burst(block_lists)
            return [bytes(r) for r in results], list(corrected)

        decoded_blocks = []
        corrected_counts = []
        for block in blocks:
            decoded, count = self.decode(block)
            decoded_blocks.append(decoded)
            corrected_counts.append(count)
        return decoded_blocks, corrected_counts
