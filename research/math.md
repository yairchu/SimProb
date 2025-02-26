<!-- markdownlint-disable MD046 -->

# Math parts

## PartialCovar.solve

    (inf * A_inf + A_real) @ (inf * X_inf + X_real) = inf * B_inf + B_real

Expand left side:

    inf^2 * A_inf @ X_inf + inf * A_inf @ X_real + inf * A_real @ X_inf + A_real @ X_real = inf * B_inf + B_real

Break to separate equations by powers of inf

* `inf^2 * A_inf @ X_inf = 0`
* `inf * A_inf @ X_real + inf * A_real @ X_inf = inf * B_inf`
* `A_real @ X_real ~= B_real` (see note on `~=`)

Divide by the common factors

* Eq 1: `A_inf @ X_inf = 0`
* Eq 2: `A_inf @ X_real + A_real @ X_inf = B_inf`
* Eq 3: `A_real @ X_real ~= B_real`

Note:
The system should be dominated by the infinite components (they take precedence),
therefore X_real should not be solved by the real parts (third component),
but rather it should work in the second factor to get the infinite part correct.

### Using Eq 1 to learn constraint of X_inf

Eq1 tells us that X_inf is constrained by the null-space of A_inf.
Therefore we define `sub_X_inf` such that:

* Def 1: `X_inf = null(A_inf) @ sub_X_inf`

We can expand Def 1 in Eq 1 and verify it works

    A_inf @ X_inf = 0
    A_inf @ null(A_inf) @ sub_X_inf = 0
    0 @ sub_X_inf = 0

### Eq 2.1: Expanding Def 1

    A_inf @ X_real + A_real @ X_inf = B_inf
    A_real @ X_inf = B_inf - A_inf @ X_real
    A_real @ null(A_inf) @ sub_X_inf = B_inf - A_inf @ X_real
    null(A_inf) @ sub_X_inf = inv(A_real) @ (B_inf - A_inf @ X_real)

**TODO**: What if A_real is not invertible?

### Eq 4: Breaking down Eq 2.1 by LHS span

We notice how the right hand side of Eq 2.1 can only span what the left side can, null(A_inf),
getting the equation

    span(A_inf).T @ inv(A_real) @ (B_inf - A_inf @ X_real) = 0
    span(A_inf).T @ inv(A_real) @ A_inf @ X_real = span(A_inf).T @ inv(A_real) @ B_inf

* Def 2: `X_real = span(A_inf) @ X_real_s + null(A_inf) @ X_real_n`

### Helper 1: Reducing A_inf @ X_real

Notice `A_inf @ X_real` from Eq 2.2 and Eq 4.
Using Def 2 and relation `A_inf = A_inf @ span(A_inf) @ span(A_inf).T` we develop it into:

    A_inf @ X_real
    A_inf @ span(A_inf) @ span(A_inf).T @ X_real
    ... @ span(A_inf).T @ X_real
    ... @ (span(A_inf).T @ (span(A_inf) @ X_real_s + null(A_inf) @ X_real_n))
    ... @ (span(A_inf).T @ span(A_inf) @ X_real_s + span(A_inf).T @ null(A_inf) @ X_real_n)
    ... @ (span(A_inf).T @ span(A_inf) @ X_real_s + 0)
    A_inf @ span(A_inf) @ span(A_inf).T @ span(A_inf) @ X_real_s
    A_inf @ span(A_inf) @ X_real_s

### Eq 4.1: Substituting Helper 1 and solving X_real_s

    span(A_inf).T @ inv(A_real) @ A_inf @ X_real = span(A_inf).T @ inv(A_real) @ B_inf
    span(A_inf).T @ inv(A_real) @ A_inf @ span(A_inf) @ X_real_s = span(A_inf).T @ inv(A_real) @ B_inf

X_real_s = inv(span(A_inf).T @ inv(A_real) @ A_inf @ span(A_inf)) @ span(A_inf).T @ inv(A_real) @ B_inf

We can now solve X_real_s

### Eq 2.2: Substitute Helper 1 and solve X_inf

    null(A_inf) @ sub_X_inf = inv(A_real) @ (B_inf - A_inf @ X_real)
    null(A_inf) @ sub_X_inf = inv(A_real) @ (B_inf - A_inf @ span(A_inf) @ X_real_s)
    null(A_inf) @ sub_X_inf = inv(A_real) @ (B_inf - A_inf @ span(A_inf) @ X_real_s)
    sub_X_inf = null(A_inf).T @ inv(A_real) @ (B_inf - A_inf @ span(A_inf) @ X_real_s)

Using Def 1

    X_inf = null(A_inf) @ sub_X_inf
    X_inf = null(A_inf) @ null(A_inf).T @ inv(A_real) @ (B_inf - A_inf @ span(A_inf) @ X_real_s)

#### Possibly simplifying further

We look at the last part `inv(A_real) @ (B_inf - A_inf @ span(A_inf) @ X_real_s)`,
and ask whether its rows span null(A_inf).
Emprically, it looks like they do!
If that indeed always holds then we could remove the `null(A_inf) @ null(A_inf).T` part.

### Eq 3.1: Towards solving X_real_n

When A_inf is full-ranked this step is not necessary (X_real_n is an empty matrix), otherwise:

    A_real @ X_real ~= B_real
    A_real @ (span(A_inf) @ X_real_s + null(A_inf) @ X_real_n) ~= B_real
    A_real @ span(A_inf) @ X_real_s + A_real @ null(A_inf) @ X_real_n ~= B_real
    A_real @ null(A_inf) @ X_real_n ~= B_real - A_real @ span(A_inf) @ X_real_s
    null(A_inf) @ X_real_n ~= pinv(A_real) @ (B_real - A_real @ span(A_inf) @ X_real_s)

We can try getting the closest solution using least squares.
Is this a good idea? We'll check by simulation.

```Python
for inf_sim in [1e5, 1e10]:
    a_sim = inf_sim * A_inf + A_real
    b_sim = inf_sim * B_inf + B_real
    print(inf_sim)
    print(inf_sim * X_inf + X_real)
    print(np.linalg.solve(a_sim, b_sim))
```

When B_inf is full-ranked, X_real_n's contribution is fully masked by X_inf, so we can't tell if it's correct!
Only when B_inf is rank-deficient, X_real_n's contribution to the simulation is exposed,
and we can see that least squares inversion didn't do the right thing.

In fact, it looks like it needs to not show through, zeroing it seems to make the simulation work!
What if we constrain it according to B_inf's span?

    X_real_n ~= pinv(A_real @ null(A_inf)) @ (B_real - A_real @ span(A_inf) @ X_real_s) @ span(B_inf) @ span(B_inf).T

### Exploring special cases: B_inf = 0

#### Eq 4.1

    X_real_s = inv(span(A_inf).T @ inv(A_real) @ A_inf @ span(A_inf)) @ span(A_inf).T @ inv(A_real) @ B_inf
    X_real_s = inv(span(A_inf).T @ inv(A_real) @ A_inf @ span(A_inf)) @ span(A_inf).T @ inv(A_real) @ 0
    X_real_s = 0

#### Eq 2.2

    sub_X_inf = null(A_inf).T @ inv(A_real) @ (B_inf - A_inf @ span(A_inf) @ X_real_s)
    sub_X_inf = null(A_inf).T @ inv(A_real) @ (0 - A_inf @ span(A_inf) @ 0)
    sub_X_inf = null(A_inf).T @ inv(A_real) @ 0
    sub_X_inf = 0

    X_inf = null(A_inf) @ sub_X_inf
    X_inf = null(A_inf) @ 0
    X_inf = 0

#### Eq 3.1

    null(A_inf) @ X_real_n ~= pinv(A_real) @ (B_real - A_real @ span(A_inf) @ X_real_s)
    null(A_inf) @ X_real_n ~= pinv(A_real) @ (B_real - A_real @ span(A_inf) @ 0)
    null(A_inf) @ X_real_n ~= pinv(A_real) @ B_real

#### Substituting in initial expression

    (inf * A_inf + A_real) @ (inf * X_inf + X_real) = inf * B_inf + B_real
    (inf * A_inf + A_real) @ X_real = B_real
    (inf * A_inf + A_real) @ (span(A_inf) @ X_real_s + null(A_inf) @ X_real_n) = B_real
    (inf * A_inf + A_real) @ null(A_inf) @ X_real_n = B_real
