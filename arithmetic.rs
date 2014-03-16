use std::num::{One, Zero};
use std_num = std::num;
use num;

/// P31 (**) Determine whether a given integer number is prime.
/// Example:
/// * (is-prime 7)
/// T

fn is_prime<I:num::Integer+Clone>(n:I) -> bool {
    // This is a slow and naive primeness test.  But it should be
    // clearly correct, and plus, it "handles" bignums (although any
    // actual prime bignum would take too long to validate under this
    // approach).
    let one : I = One::one();
    let two = one + one;
    let mut i = two.clone();
    let n_div_2 = n/two;
    while i <= n_div_2 {
        if (n/i)*i == n {
            return false;
        }
        i = i + one;
    }
    return true;
}

#[test]
fn test_is_prime() {
    use num::bigint::BigUint;
    use std::from_str::FromStr;
    assert_eq!(false, is_prime(4));
    assert_eq!(true,  is_prime(7));
    assert_eq!(false, is_prime(10));
    assert_eq!(true,  is_prime(23));
    let big1e21  : BigUint = FromStr::from_str("1000000000000000000000").unwrap();
    let one      : BigUint = One::one();
    assert_eq!(false, is_prime(big1e21 + one));

    // is_prime is too slow to do this for now.
    // let bigprime : BigUint = FromStr::from_str("18446744082299486207").unwrap();
    // assert_eq!(true, is_prime(bigprime));
}

/// P32 (**) Determine the greatest common divisor of two positive integer numbers.
///     Use Euclid's algorithm.
///     Example:
///     * (gcd 36 63)
///     9
fn gcd<I:num::Integer>(a:I, b:I) -> I {
    fn low_high<I:num::Integer>(a:I,b:I) -> (I,I) { if a < b { (a,b) } else { (b,a) } }
    let (mut low, high) = low_high(a,b);
    let mut delta = high - low;
    while low != delta {
        let (l, h) = low_high(low, delta);
        delta = h - l;
        low = l;
    }
    return delta;
}

#[test]
fn test_gcd() {
    assert_eq!(9, gcd(36, 63));
}

/// P33 (*) Determine whether two positive integer numbers are coprime.
///     Two numbers are coprime if their greatest common divisor equals 1.
///     Example:
///     * (coprime 35 64)
///     T

fn coprime<I:num::Integer>(a:I, b:I) -> bool {
    gcd(a,b) == One::one()
}

#[test]
fn test_coprime() {
    assert_eq!(true, coprime(35, 64));
}

/// P34 (**) Calculate Euler's totient function phi(m).
///     Euler's so-called totient function phi(m) is defined as the
///     number of positive integers r (1 <= r < m) that are coprime to
///     m.
/// 
///     Example: m = 10: r = 1,3,7,9; thus phi(m) = 4. Note the special case: phi(1) = 1.
/// 
///     * (totient-phi 10)
///     4
/// 
///     Find out what the value of phi(m) is if m is a prime
///     number. Euler's totient function plays an important role in
///     one of the most widely used public key cryptography methods
///     (RSA). In this exercise you should use the most primitive
///     method to calculate this function (there are smarter ways that
///     we shall discuss later).

fn totient_phi<I:num::Integer+Clone>(m:I) -> I {
    let one = One::one();
    if m == one {
        one
    } else {
        let mut count : I = Zero::zero();
        let mut r = one.clone();
        while r < m {
            if coprime(r.clone(), m.clone()) {
                count = count + one;
            }
            r = r + one;
        }
        count
    }
}

#[test]
fn test_totient_phi() {
    assert_eq!(4, totient_phi(10));
}

/// P35 (**) Determine the prime factors of a given positive integer.
///     Construct a flat list containing the prime factors in ascending order.
///     Example:
///     * (prime-factors 315)
///     (3 3 5 7)

fn prime_factors<I:num::Integer+Clone>(mut m:I) -> ~[I] {
    let one : I = One::one();
    let two = one + one;
    let mut result = ~[];
    let mut cand = two.clone();
    let mut m_div_2 = m / two;
    while cand < m_div_2 {
        let m_div_cand = m / cand;
        if (m_div_cand * cand) == m {
            result.push(cand.clone());
            m = m_div_cand;
            m_div_2 = m / two;
        } else {
            cand = cand + one;
        }
    }
    // could not divide further; remainder m must be prime.
    result.push(m);
    result
}

#[test]
fn test_prime_factors() {
    assert_eq!(~[3, 3, 5, 7], prime_factors(315));
}

// P36 (**) Determine the prime factors of a given positive integer (2).
//     Construct a list containing the prime factors and their multiplicity.
//     Example:
//     * (prime-factors-mult 315)
//     ((3 2) (5 1) (7 1))
// 
//     Hint: The problem is similar to problem P13.
fn prime_factors_mult<I:num::Integer+Clone>(mut m:I) -> ~[(I,uint)] {
    // (could multiplicity actually overflow?  Sticking with uint for clarity for now.)
    let one : I = One::one();
    let two = one + one;
    let mut result = ~[];
    let mut cand = two.clone();
    let mut m_div_2 = m / two;
    let mut count = 0;
    while cand < m_div_2 {
        let m_div_cand = m / cand;
        if (m_div_cand * cand) == m {
            count += 1;
            m = m_div_cand;
            m_div_2 = m / two;
        } else {
            if count > 0 {
                result.push((cand.clone(), count));
            }
            cand = cand + one;
            count = 0;
        }
    }
    // could not divide further; remainder m must be prime.
    if count > 0 {
        result.push((cand.clone(), count));
    }
    result.push((m, 1));
    result
}

#[test]
fn test_prime_factors_mult() {
    assert_eq!(~[(3, 2), (5, 1), (7, 1)], prime_factors_mult(315));
}

/// P37 (**) Calculate Euler's totient function phi(m) (improved).
///     See problem P34 for the definition of Euler's totient
///     function. If the list of the prime factors of a number m is
///     known in the form of problem P36 then the function phi(m) can
///     be efficiently calculated as follows: Let ((p1 m1) (p2 m2) (p3
///     m3) ...) be the list of prime factors (and their
///     multiplicities) of a given number m. Then phi(m) can be
///     calculated with the following formula (sic):
///
///     phi(m) = (p1 - 1) * p1 ** (m1 - 1) + (p2 - 1) * p2 ** (m2 - 1) +
///              (p3 - 1) * p3 ** (m3 - 1) + ...
/// 
///     Note that a ** b stands for the b'th power of a.
///
///     Note that the "a + b" above stands for a product of a and b.
///     (i.e. there is a typo in the original problem statement,
///      thus my "sic" annotation).
fn totient_phi_improved<I:num::Integer+Clone>(m:I) -> I {
    let one : I = One::one();
    let f = prime_factors_mult(m);
    f.move_iter().fold(one.clone(), |b, (factor, count)| {
        b * (factor - one) * std_num::pow(factor, count - 1)
    })
}

#[test]
fn test_totient_phi_improved() {
    assert_eq!(4, totient_phi_improved(10));
}

// 
// P38 (*) Compare the two methods of calculating Euler's totient function.
//     Use the solutions of problems P34 and P37 to compare the algorithms. Take the number of logical inferences as a measure for efficiency. Try to calculate phi(10090) as an example.
mod bench_P38 {
    extern crate test;
    use self::test::BenchHarness;
    use super::{totient_phi_improved, totient_phi};

    #[bench]
    fn bench_totient_phi(bh: &mut BenchHarness) {
        bh.iter(|| {
            totient_phi(10090);
        });
    }

    #[bench]
    fn bench_totient_phi_improved(bh: &mut BenchHarness) {
        bh.iter(|| {
            totient_phi_improved(10090);
        });
    }
}

// P39 (*) A list of prime numbers.
//     Given a range of integers by its lower and upper limit, construct a list of all prime numbers in that range.

fn list_primes<I:num::Integer+Clone>(low_incl: I, high_incl: I) -> ~[I] {
    // Another naive (very slow) approach.  Would be much better to
    // investigate using e.g. Sieve of Erasthones
    let mut result = ~[];
    let one : I = One::one();
    let zed : I = Zero::zero();
    let mut n = low_incl.clone();
    let mut count = high_incl - low_incl + one;
    while count > zed {
        if is_prime(n.clone()) {
            result.push(n.clone());
        }
        n = n + one;
        count = count - one;
    }
    return result;
}

#[test]
fn test_list_primes() {
    assert_eq!(~[3, 5, 7, 11], list_primes(3, 12));
}

// P40 (**) Goldbach's conjecture.
//     Goldbach's conjecture says that every positive even number greater than 2 is the sum of two prime numbers. Example: 28 = 5 + 23. It is one of the most famous facts in number theory that has not been proved to be correct in the general case. It has been numerically confirmed up to very large numbers (much larger than we can go with our Prolog system). Write a predicate to find the two prime numbers that sum up to a given even integer.
// 
//     Example:
//     * (goldbach 28)
//     (5 23)
pub fn goldbach<I:num::Integer+Clone+::std::fmt::Show>(n:I) -> (I,I) {
    let one : I = One::one();
    let two : I = one + one;
    let mut i = two.clone();
    if n % two == one || n <= two {
        fail!("Can only test goldbach conjecture on even numbers > 2");
    }
    let n_div_2 = n / two;
    while i <= n_div_2 {
        let n_sub_i = n - i;
        debug!("goldbach trying: {} {}", i, n_sub_i);
        if is_prime(i.clone()) && is_prime(n_sub_i.clone()) {
            return (i, n_sub_i);
        }
        i = i + one;
    }
    fail!("goldbach was wrong! {}", n);
}

#[test]
fn test_goldbach() {
    assert_eq!((5, 23), goldbach(28));
    assert!({ println!("{}", goldbach(100)); true });
}


// P41 (**) A list of Goldbach compositions.
//     Given a range of integers by its lower and upper limit, print a list of all even numbers and their Goldbach composition.
// 
//     Example:
//     * (goldbach-list 9 20)
//     10 = 3 + 7
//     12 = 5 + 7
//     14 = 3 + 11
//     16 = 3 + 13
//     18 = 5 + 13
//     20 = 3 + 17
// 
//     In most cases, if an even number is written as the sum of two prime numbers, one of them is very small. Very rarely, the primes are both bigger than say 50. Try to find out how many such cases there are in the range 2..3000.
// 
//     Example (for a print limit of 50):
//     * (goldbach-list 1 2000 50)
//     992 = 73 + 919
//     1382 = 61 + 1321
//     1856 = 67 + 1789
//     1928 = 61 + 1867

#[deriving(Eq,Show)]
struct GoldbachComposition<I> {
    num: I,
    equals: (I, I)
}

fn goldbach_list<I:num::Integer+Clone+::std::fmt::Show>(
    low_incl: I, high_incl: I, threshold: Option<I>) -> ~[GoldbachComposition<I>] {
    let one : I = One::one();
    let two : I = one + one;
    let mut result = ~[];
    let mut curr = if low_incl <= two {
        two + two
    } else if low_incl.is_even() {
        low_incl
    } else {
        low_incl + one
    };
    let threshold = threshold.unwrap_or(Zero::zero());
    while curr <= high_incl {
        let (lft, rgt) = goldbach(curr.clone());
        if lft > threshold && rgt > threshold {
            result.push(GoldbachComposition{ num: curr.clone(), equals: (lft, rgt) });
        }
        curr = curr + two;
    }
    result
}

#[test]
fn test_goldbach_list() {
    assert_eq!(goldbach_list(9, 20, None),
               ~[GoldbachComposition{ num: 10, equals: (3,  7) },
                 GoldbachComposition{ num: 12, equals: (5,  7) },
                 GoldbachComposition{ num: 14, equals: (3, 11) },
                 GoldbachComposition{ num: 16, equals: (3, 13) },
                 GoldbachComposition{ num: 18, equals: (5, 13) },
                 GoldbachComposition{ num: 20, equals: (3, 17) }]);

    assert_eq!(goldbach_list(1, 2000, Some(50)),
               ~[GoldbachComposition{ num:  992, equals: (73,  919) },
                 GoldbachComposition{ num: 1382, equals: (61, 1321) },
                 GoldbachComposition{ num: 1856, equals: (67, 1789) },
                 GoldbachComposition{ num: 1928, equals: (61, 1867) }]);
}
