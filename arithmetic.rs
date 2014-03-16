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
    use std::num::One;
    let one : I = One::one();
    let two = one + one;
    let mut i = two.clone();
    let n_div_2 = n/two;
    while i < n_div_2 {
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
    use std::num::One;
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

// P32 (**) Determine the greatest common divisor of two positive integer numbers.
//     Use Euclid's algorithm.
//     Example:
//     * (gcd 36 63)
//     9
// 
// P33 (*) Determine whether two positive integer numbers are coprime.
//     Two numbers are coprime if their greatest common divisor equals 1.
//     Example:
//     * (coprime 35 64)
//     T
// 
// P34 (**) Calculate Euler's totient function phi(m).
//     Euler's so-called totient function phi(m) is defined as the number of positive integers r (1 <= r < m) that are coprime to m.
// 
//     Example: m = 10: r = 1,3,7,9; thus phi(m) = 4. Note the special case: phi(1) = 1.
// 
//     * (totient-phi 10)
//     4
// 
//     Find out what the value of phi(m) is if m is a prime number. Euler's totient function plays an important role in one of the most widely used public key cryptography methods (RSA). In this exercise you should use the most primitive method to calculate this function (there are smarter ways that we shall discuss later).
// 
// P35 (**) Determine the prime factors of a given positive integer.
//     Construct a flat list containing the prime factors in ascending order.
//     Example:
//     * (prime-factors 315)
//     (3 3 5 7)
// 
// P36 (**) Determine the prime factors of a given positive integer (2).
//     Construct a list containing the prime factors and their multiplicity.
//     Example:
//     * (prime-factors-mult 315)
//     ((3 2) (5 1) (7 1))
// 
//     Hint: The problem is similar to problem P13.
// 
// P37 (**) Calculate Euler's totient function phi(m) (improved).
//     See problem P34 for the definition of Euler's totient function. If the list of the prime factors of a number m is known in the form of problem P36 then the function phi(m) can be efficiently calculated as follows: Let ((p1 m1) (p2 m2) (p3 m3) ...) be the list of prime factors (and their multiplicities) of a given number m. Then phi(m) can be calculated with the following formula:
// 
//     phi(m) = (p1 - 1) * p1 ** (m1 - 1) + (p2 - 1) * p2 ** (m2 - 1) + (p3 - 1) * p3 ** (m3 - 1) + ...
// 
//     Note that a ** b stands for the b'th power of a.
// 
// P38 (*) Compare the two methods of calculating Euler's totient function.
//     Use the solutions of problems P34 and P37 to compare the algorithms. Take the number of logical inferences as a measure for efficiency. Try to calculate phi(10090) as an example.
// 
// P39 (*) A list of prime numbers.
//     Given a range of integers by its lower and upper limit, construct a list of all prime numbers in that range.
// 
// P40 (**) Goldbach's conjecture.
//     Goldbach's conjecture says that every positive even number greater than 2 is the sum of two prime numbers. Example: 28 = 5 + 23. It is one of the most famous facts in number theory that has not been proved to be correct in the general case. It has been numerically confirmed up to very large numbers (much larger than we can go with our Prolog system). Write a predicate to find the two prime numbers that sum up to a given even integer.
// 
//     Example:
//     * (goldbach 28)
//     (5 23)
// 
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
