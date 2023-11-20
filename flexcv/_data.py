from .synthesizer import generate_regression

X, y, group, random_slopes = generate_regression(
    10, 100, n_slopes=1, noise_level=9.1e-2, random_seed=42
)

DATA_TUPLE_10_100 = (X, y, group, random_slopes)

X, y, group, random_slopes = generate_regression(
    3, 25, n_slopes=1, noise_level=9.1e-2, random_seed=42
)

DATA_TUPLE_3_25 = (X, y, group, random_slopes)

X, y, group, random_slopes = generate_regression(
    3,
    100,
    n_slopes=1,
    noise_level=9.1e-2,
    random_seed=42,
)

DATA_TUPLE_3_100 = (X, y, group, random_slopes)
