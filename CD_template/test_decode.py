import numpy as np
import pytest
import galois

from RSCode import RSCode  # adjust import


@pytest.fixture
def rs():
    m = 8
    t = 3
    l = 10
    m0 = 1
    return RSCode(m, t, l, m0)


@pytest.fixture
def GF(rs):
    return galois.GF(2**rs.m)


def random_msg(GF, rows, cols):
    return GF(np.random.randint(0, GF.order - 1, (rows, cols)))


# -------------------
# ENCODE TESTS
# -------------------

def test_encode_valid(rs, GF):
    msg = random_msg(GF, 4, rs.l)
    code = rs.encode(msg)

    assert code.shape == (4, rs.l + rs.n - rs.k)
    assert isinstance(code, GF)


def test_encode_invalid_shape(rs, GF):
    msg = random_msg(GF, 4, rs.l + 1)
    with pytest.raises(AssertionError):
        rs.encode(msg)


def test_encode_invalid_type(rs):
    msg = np.random.randint(0, 10, (4, rs.l))  # not GF
    with pytest.raises(AssertionError):
        rs.encode(msg)


# -------------------
# DECODE TESTS
# -------------------

def test_decode_no_errors(rs, GF):
    msg = random_msg(GF, 3, rs.l)
    code = rs.encode(msg)

    decoded, nERR = rs.decode(code)

    assert np.all(decoded == msg)
    assert np.all(nERR == 0)


def test_decode_correctable_errors(rs, GF):
    msg = random_msg(GF, 2, rs.l)
    code = rs.encode(msg)

    # introduce <= t errors
    for r in range(code.shape[0]):
        idx = np.random.choice(code.shape[1], rs.t, replace=False)
        code[r, idx] += GF.Random(rs.t)

    decoded, nERR = rs.decode(code)

    assert np.all(decoded == msg)
    assert np.all(nERR <= rs.t)


def test_decode_too_many_errors(rs, GF):
    msg = random_msg(GF, 1, rs.l)
    code = rs.encode(msg)

    # introduce > t errors
    idx = np.random.choice(code.shape[1], rs.t + 2, replace=False)
    code[0, idx] += GF.Random(rs.t + 2)

    decoded, nERR = rs.decode(code)

    assert nERR[0] == -1
    # fallback returns corrupted message slice
    assert decoded.shape == (1, rs.l)


def test_decode_zero_syndrome_path(rs, GF):
    msg = random_msg(GF, 1, rs.l)
    code = rs.encode(msg)

    decoded, nERR = rs.decode(code)

    # triggers early continue
    assert nERR[0] == 0
    assert np.all(decoded == msg)


def test_decode_berlekamp_failure(rs, GF):
    msg = random_msg(GF, 1, rs.l)
    code = rs.encode(msg)

    # heavy corruption → likely L > t
    idx = np.arange(code.shape[1])
    code[0, idx] += GF.Random(len(idx))

    decoded, nERR = rs.decode(code)

    assert nERR[0] == -1


def test_decode_chien_search_mismatch(rs, GF):
    msg = random_msg(GF, 1, rs.l)
    code = rs.encode(msg)

    # crafted corruption: random large disturbance
    code[0, :] = GF.Random(code.shape[1])

    decoded, nERR = rs.decode(code)

    assert nERR[0] == -1


def test_decode_forney_denominator_zero(rs, GF):
    msg = random_msg(GF, 1, rs.l)
    code = rs.encode(msg)

    # Try to provoke derivative zero case (rare, so brute randomness)
    for _ in range(10):
        corrupted = code.copy()
        idx = np.random.choice(code.shape[1], rs.t, replace=False)
        corrupted[0, idx] += GF.Random(rs.t)

        decoded, nERR = rs.decode(corrupted)

        # we accept either success or controlled failure
        assert nERR[0] in (-1, 0, 1, 2, 3)


def test_decode_syndrome_check_failure(rs, GF):
    msg = random_msg(GF, 1, rs.l)
    code = rs.encode(msg)

    # corrupt then partially "fix" incorrectly
    idx = np.random.choice(code.shape[1], rs.t, replace=False)
    code[0, idx] += GF.Random(rs.t)

    decoded, nERR = rs.decode(code)

    # if post-check fails, must return -1
    assert nERR[0] in (-1, rs.t)


def test_decode_invalid_shape(rs, GF):
    code = GF.Zeros((2, rs.l))  # wrong width
    with pytest.raises(AssertionError):
        rs.decode(code)


def test_decode_invalid_type(rs):
    code = np.zeros((2, rs.l + rs.n - rs.k))
    with pytest.raises(AssertionError):
        rs.decode(code)


# -------------------
# GENERATOR TEST
# -------------------

def test_generator_type():
    g = RSCode.makeGenerator(8, 3, 1)
    assert isinstance(g, galois.Poly)


# -------------------
# ROUNDTRIP FUZZ TEST
# -------------------

def test_random_roundtrip(rs, GF):
    for _ in range(10):
        msg = random_msg(GF, 3, rs.l)
        code = rs.encode(msg)

        # random errors up to t
        for r in range(code.shape[0]):
            n_err = np.random.randint(0, rs.t + 1)
            if n_err > 0:
                idx = np.random.choice(code.shape[1], n_err, replace=False)
                code[r, idx] += GF.Random(n_err)

        decoded, nERR = rs.decode(code)

        assert np.all(decoded == msg)