# Question 2 (Decoding):

We consider the C1 Reed-Solomon code used in CD audio, which has 4 parity symbols, so

- $n-k = 4$
- $t = \frac{n-k}{2} = 2$ symbol errors correctable
- symbols are over $\mathrm{GF}(2^8)$, so 1 symbol = 8 bits.

## (a) How many erroneous bits can C1 always correct? What is the maximum number of correctable bit errors?

An RS decoder corrects up to $t=2$ erroneous **symbols**.

- Always correctable bit errors (guaranteed for any error pattern): **2 bits**.
Reason: in the worst case, each erroneous bit is in a different symbol, so $b$ bit errors can affect up to $b$ symbols. To guarantee at most 2 erroneous symbols, we need $b \le 2$.

- Maximum correctable bit errors (best case pattern): **16 bits**.
Reason: if all bit errors are concentrated in only 2 symbols, both symbols can be fully corrupted (8 bits each), and RS still corrects them: $2 \times 8 = 16$ bits.

## (b) Relationship between bit error rate $P_b$ and symbol error rate $P_s$ for $\mathrm{GF}(2^8)$

Assuming independent (uncorrelated) bit errors:

- Probability a symbol is correct: $(1-P_b)^8$
- Probability a symbol is erroneous:

$$
P_s = 1 - (1 - P_b)^8
$$

Equivalent inverse relation:

$$
P_b = 1 - (1 - P_s)^{1/8}
$$

For small $P_b$, a useful approximation is:

$$
P_s \approx 8P_b
$$