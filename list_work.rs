pub mod sexp {
    pub enum Sexp<X> {
        A(X),
        L(~[Sexp<X>])
    }

    trait SexpLike<A> {
        fn to_sexp(self) -> Sexp<A>;
    }

    impl<A> SexpLike<A> for ~[A] {
        fn to_sexp(self) -> Sexp<A> {
            L(self.move_iter().map(|v|A(v)).collect())
        }
    }
}

mod vec_work {
    use super::sexp::{Sexp,A,L};

    static a : &'static str = "a";
    static b : &'static str = "b";
    static c : &'static str = "c";
    static d : &'static str = "d";
    static e : &'static str = "e";

    /// P01 Find the last *box* of a "list."
    /// Example:
    /// * (my-last '(a b c d))
    /// (D)
    pub fn my_last<'a, A>(x: &'a [A]) -> &'a [A] { x.slice_from(x.len()-1) }

    #[test]
    fn test_my_last() { assert!([d] == my_last([a, b, c, d])); }

    /// P02 Find the last but one box of a list.
    /// Example:
    /// * (my-but-last '(a b c d))
    /// (C D)
    pub fn my_but_last<'a, A>(x: &'a [A]) -> &'a [A] { x.slice_from(x.len() - 2) }

    #[test]
    fn test_my_but_last() { assert!([c, d] == my_but_last([a, b, c, d])); }

    /// P03 (*) Find the K'th element of a list.
    /// The first element in the list is number 1.
    /// Example:
    /// * (element-at '(a b c d e) 3)
    /// C
    pub fn element_at<'a, A>(x: &'a [A], i:uint) -> &'a A { &x[i - 1] }

    #[test]
    fn test_element_at() {
        assert!(c == *element_at([a, b, c, d, e], 3))
    }

    /// P04 Find the number of elements of a list.
    #[test]
    fn test_num_elems() {
        assert!(5 == [a, b, c, d, e].len())
}

    /// P05 Reverse a list.
    #[test]
    fn test_rev_list() {
        fn rev<A>(v: ~[A]) -> ~[A] { v.move_rev_iter().collect() }
        assert!(~[c, b, a] == rev(~[a, b, c]));
    }

    /// P06 Find out whether a list is a palindrome.
    /// A palindrome can be read forward or backward; e.g. (x a m a x)
    fn is_palindrome<A:Eq>(v: &[A]) -> bool {
        v.iter().zip(v.rev_iter()).all(|(lft, rgt)| lft == rgt)
    }
    #[test]
    fn test_is_palindrome() {
        assert!(is_palindrome(["x", a, "m", a, "x"]));
        assert!(is_palindrome(["x", a, a, "x"]));
        assert!(!is_palindrome(["x", a, "m", a]));
    }

    /// P07 (**) Flatten a nested list structure.
    /// Transform a list, possibly holding lists as elements into a
    /// `flat' list by replacing each list with its elements
    /// (recursively).
    ///
    /// Example:
    /// * (my-flatten '(a (b (c d) e)))
    /// (A B C D E)
    ///
    /// Hint: Use the predefined functions list and append.
    fn flatten<X>(s:Sexp<X>) -> ~[X] {
        match s {
            A(x) => ~[x],
            L(l) => l.move_iter().flat_map(|x|flatten(x).move_iter()).collect(),
        }
    }

    #[test]
    fn test_my_flatten() {
        assert!(~[a, b, c, d, e] ==
                flatten(L(~[A(a), L(~[A(b), L(~[A(c)]), A(d)]), A(e)])));
    }

    /// P08 (**) Eliminate consecutive duplicates of list elements.
    /// If a list contains repeated elements they should be replaced
    /// with a single copy of the element. The order of the elements
    /// should not be changed.
    ///
    /// Example:
    /// * (compress '(a a a a b c c a a d e e e e))
    /// (A B C A D E)
    #[cfg(dead_code)]
    fn compress_easy<X:Eq>(l: ~[X]) -> ~[X] {
        let mut l = l; l.dedup(); l
    }

    fn compress<X:Eq>(l: ~[X]) -> ~[X] {
        let mut l = l;
        let mut result = ~[];
        while !l.is_empty() {
            result.push(l.pop().unwrap());
            let last = result.last().unwrap();
            while !l.is_empty() && l.last().unwrap() == last {
                l.pop();
            }
        }
        result.reverse();
        result
    }

    #[test]
    fn test_compress() {
        assert!(~[a, b, c, a, d, e] ==
                compress(~[a, a, a, a, b, c, c,
                           a, a, d, e, e, e, e]))
    }

    /// P09 (**) Pack consecutive duplicates of list elements into sublists.
    /// If a list contains repeated elements they should be placed in separate sublists.
    ///
    /// Example:
    /// * (pack '(a a a a b c c a a d e e e e))
    /// ((A A A A) (B) (C C) (A A) (D) (E E E E))
    #[test]
    #[cfg(off)]
    fn test_pack() {
        assert!(~[((a a a a) (b) (c c) (a a) (d) (e e e e))] ==
                pack(~[a a a a b c c a a d e e e e]));
    }
}
