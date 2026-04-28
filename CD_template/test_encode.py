import numpy as np
import pytest
import galois

from RSCode import RSCode


@pytest.fixture
def rs():
    return RSCode(m=8, t=3, l=10, m0=1)


@pytest.fixture
def GF(rs):
    return galois.GF(2**rs.m)


@pytest.fixture(autouse=True)
def fixed_seed():
    np.random.seed(1234)


def random_msg(GF, rows, cols):
    return GF(np.random.randint(0, GF.order - 1, (rows, cols)))


def test_encode_shape_and_type(rs, GF):
    msg = random_msg(GF, 5, rs.l)
    code = rs.encode(msg)

    assert code.shape == (5, rs.l + rs.n - rs.k)
    assert isinstance(code, GF)


def test_encode_invalid_shape_raises(rs, GF):
    msg = random_msg(GF, 2, rs.l + 1)
    with pytest.raises(AssertionError):
        rs.encode(msg)


def test_encode_invalid_type_raises(rs):
    msg = np.random.randint(0, 255, (2, rs.l))
    with pytest.raises(AssertionError):
        rs.encode(msg)


def test_encode_is_systematic_for_shortened_code(rs, GF):
    msg = random_msg(GF, 4, rs.l)
    code = rs.encode(msg)

    # In this implementation, shortened systematic RS keeps info symbols first.
    assert np.array_equal(code[:, : rs.l], msg)


def test_encode_zero_message_gives_zero_codeword(rs, GF):
    msg = GF.Zeros((3, rs.l))
    code = rs.encode(msg)

    assert np.array_equal(code, GF.Zeros((3, rs.l + rs.n - rs.k)))


def test_encode_linearity(rs, GF):
    a = random_msg(GF, 3, rs.l)
    b = random_msg(GF, 3, rs.l)

    lhs = rs.encode(a + b)
    rhs = rs.encode(a) + rs.encode(b)

    assert np.array_equal(lhs, rhs)


def test_encoded_full_length_word_is_generator_multiple(rs, GF):
    msg = random_msg(GF, 1, rs.l)
    code = rs.encode(msg)

    n_par = rs.n - rs.k
    pad = rs.k - rs.l
    full_word = np.concatenate((GF.Zeros(pad), code[0, :]))
    poly = galois.Poly(full_word, field=GF)

    rem = poly % rs.g
    assert rem == galois.Poly.Zero(GF)
    assert code.shape[1] == rs.l + n_par


@pytest.mark.parametrize("m0", [0, 1, 2, 5, 13])
def test_encode_valid_for_different_m0(m0):
    rs = RSCode(m=8, t=3, l=10, m0=m0)
    GF = galois.GF(2**rs.m)
    msg = GF(np.random.randint(0, GF.order - 1, (2, rs.l)))

    code = rs.encode(msg)

    assert code.shape == (2, rs.l + rs.n - rs.k)
    assert isinstance(code, GF)


def test_encode_deterministic_for_same_input(rs, GF):
    msg = random_msg(GF, 3, rs.l)

    code_1 = rs.encode(msg)
    code_2 = rs.encode(msg)

    assert np.array_equal(code_1, code_2)


def test_encode_fuzz_multiple_batches(rs, GF):
    for _ in range(20):
        rows = np.random.randint(1, 8)
        msg = random_msg(GF, rows, rs.l)
        code = rs.encode(msg)

        assert code.shape == (rows, rs.l + rs.n - rs.k)
        assert np.array_equal(code[:, : rs.l], msg)
