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

macro_rules! static_vec {
    ( $($e:expr)* ) => { [$($e),*] }
}

macro_rules! owned_vec {
    ( $($e:expr)* ) => { ~[$($e),*] }
}

// These do not work (and Felix does not yet know if they can be made to work).
//
// macro_rules! sexp {
//     (   $i:ident )    => { A($i) };
//     ( ( $e:tt )  )    => { L(sexp_list!(($e), ())) };
// }
// 
// macro_rules! sexp_list {
//     ( ( $i1:ident  $e2:tt ), ($s:tt)      ) => { sexp_list!(($e2), (       A($i1), $s ) ) };
//     ( ( ( $e1:tt ) $e2:tt ), ($s:tt)      ) => { sexp_list!(($e2), ( sexp!(($e1)), $s ) ) };
//     (                    (), $($s:expr),* ) => { ~[$($s),*] };
// }

pub mod sym {
    macro_rules! def_sym {
        ($i:ident) => { pub static $i : &'static str = stringify!($i); }
    }

    macro_rules! def_syms {
        ( $($i:ident)* ) => { $(def_sym!($i))* }
    }

    def_syms!(a b c d e m x)
}

mod vec_work {
    use super::sexp::{Sexp,A,L};
    use super::sym::*;
    use std::mem::swap;
    use std::vec;

    /// P01 Find the last *box* of a "list."
    /// Example:
    /// * (my-last '(a b c d))
    /// (D)
    pub fn my_last<'a, A>(v: &'a [A]) -> &'a [A] { v.slice_from(v.len()-1) }

    #[test]
    fn test_my_last() { assert!([d] == my_last(static_vec!(a b c d))); }

    /// P02 Find the last but one box of a list.
    /// Example:
    /// * (my-but-last '(a b c d))
    /// (C D)
    pub fn my_but_last<'a, A>(v: &'a [A]) -> &'a [A] { v.slice_from(v.len() - 2) }

    #[test]
    fn test_my_but_last() { assert!([c, d] == my_but_last(static_vec!(a b c d))); }

    /// P03 (*) Find the K'th element of a list.
    /// The first element in the list is number 1.
    /// Example:
    /// * (element-at '(a b c d e) 3)
    /// C
    pub fn element_at<'a, A>(v: &'a [A], i:uint) -> &'a A { &v[i - 1] }

    #[test]
    fn test_element_at() {
        assert!(c == *element_at(static_vec!(a b c d e), 3))
    }

    /// P04 Find the number of elements of a list.
    #[test]
    fn test_num_elems() {
        assert!(5 == static_vec!(a b c d e).len())
}

    /// P05 Reverse a list.
    #[test]
    fn test_rev_list() {
        fn rev<A>(v: ~[A]) -> ~[A] { v.move_rev_iter().collect() }
        assert_eq!(owned_vec!(c b a), rev(owned_vec!(a b c)));
    }

    /// P06 Find out whether a list is a palindrome.
    /// A palindrome can be read forward or backward; e.g. (x a m a x)
    fn is_palindrome<A:Eq>(v: &[A]) -> bool {
        v.iter().zip(v.rev_iter()).all(|(lft, rgt)| lft == rgt)
    }
    #[test]
    fn test_is_palindrome() {
        assert!(is_palindrome([x, a, m, a, x]));
        assert!(is_palindrome([x, a, a, x]));
        assert!(!is_palindrome([x, a, m, a]));
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
            A(atom) => ~[atom],
            L(l) => l.move_iter().flat_map(|v|flatten(v).move_iter()).collect(),
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

    #[cfg(dead_code)]
    fn compress_vec<X:Eq>(l: ~[X]) -> ~[X] {
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

    fn compress_iter<X:Eq,I:Iterator<X>>(mut it: I) -> ~[X] {
        let mut result = ~[];
        let mut v = it.next();
        loop {
            match v {
                None => return result,
                Some(val) => {
                    loop {
                        let next = it.next();
                        match next {
                            Some(ref next) if next == &val => {}
                            _ => { v = next; break; }
                        }
                    }
                    result.push(val);
                }
            }
        }
    }

    fn compress<X:Eq>(l: ~[X]) -> ~[X] { compress_iter(l.move_iter()) }

    #[test]
    fn test_compress() {
        assert_eq!(~[a, b, c, a, d, e],
                compress(owned_vec!(a a a a b c c a a d e e e e)))
    }

    /// P09 (**) Pack consecutive duplicates of list elements into sublists.
    /// If a list contains repeated elements they should be placed in separate sublists.
    ///
    /// Example:
    /// * (pack '(a a a a b c c a a d e e e e))
    /// ((A A A A) (B) (C C) (A A) (D) (E E E E))
    fn pack_iter<X:Eq,I:Iterator<X>>(mut it: I) -> ~[~[X]] {
        let mut lists = ~[];
        let mut v = it.next();
        loop {
            match v {
                None => return lists,
                Some(mut val) => {
                    let mut curr = ~[];
                    loop {
                        let next = it.next();
                        match next {
                            None => {
                                curr.push(val);
                                lists.push(curr);
                                v = None;
                                break;
                            }
                            Some(mut n) => {
                                if n == val {
                                    swap(&mut val, &mut n);
                                    curr.push(n);
                                    continue;
                                } else {
                                    curr.push(val);
                                    lists.push(curr);
                                    v = Some(n);
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    fn pack<A:Eq>(l: ~[A]) -> ~[~[A]] { pack_iter(l.move_iter()) }

    #[test]
    fn test_pack() {
        assert!(~[            ~[a,a,a,a], ~[b], ~[c,c], ~[a,a], ~[d], ~[e,e,e,e] ] ==
                pack(owned_vec!(a a a a     b     c c     a a     d     e e e e)));
    }

    /// Wrapper struct where value V has been tagged with meta-data M.  Note that
    /// only V is considered when doing operations like Eq (and Ord, etc).
    struct Meta<M,V> { meta: M, value: V }

    impl<M,V:Eq> Eq for Meta<M,V> {
        fn eq(&self, other: &Meta<M,V>) -> bool {
            self.value == other.value
        }
    }

    #[test]
    fn test_pack_meta() {
        fn m<M,V>(m:M, v:V) -> Meta<M,V> { Meta { meta: m, value: v } }
        assert!(~[~[m( 1,a), m( 2,a), m( 3,a), m( 4,a)],
                  ~[m( 5,b)],
                  ~[m( 6,c), m( 7,c)],
                  ~[m( 8,a), m( 9,a)],
                  ~[m(10,d)],
                  ~[m(11,e), m(12,e), m(13,e), m(14,e)]]

                ==

                pack(owned_vec!(m( 1,a) m( 2,a) m( 3,a) m( 4,a)
                                m( 5,b)
                                m( 6,c) m( 7,c)
                                m( 8,a) m( 9,a)
                                m(10,d)
                                m(11,e) m(12,e) m(13,e) m(14,e))));
    }


    /// P10 (*) Run-length encoding of a list.
    /// Use the result of problem P09 to implement the so-called
    /// run-length encoding data compression method. Consecutive
    /// duplicates of elements are encoded as lists (N E) where N is
    /// the number of duplicates of the element E.
    ///
    /// Example:
    /// * (encode '(a a a a b c c a a d e e e e))
    /// ((4 A) (1 B) (2 C) (2 A) (1 D)(4 E))

    fn encode<A:Eq>(l: ~[A]) -> ~[(uint, A)] {
        pack(l).move_iter().map(|l| (l.len(), l[0])).collect()
    }

    #[test]
    fn test_encode() {
        assert_eq!(~[(4, a), (1, b), (2, c), (2, a), (1, d), (4, e)],
                   encode(owned_vec!(a a a a b c c a a d e e e e)));
    }

    /// N.B. this particular "optimized" representation is silly,
    /// because sizeof(J(A)) == sizeof(C(uint,A)), which are both
    /// strictly larger than sizeof((uint, A)).  The only way it could
    /// make sense would be if we used `C(~(uint,A))` instead, but
    /// that adds little to the pedagogy here (and pedagogy is the
    /// only reason I am bothering with this representation).
    #[deriving(Eq,Show)]
    enum ModRLE<A> { C(uint, A), J(A) }

    fn encode_modified<A:Eq>(l: ~[A]) -> ~[ModRLE<A>] {
        pack(l).move_iter().map(|l| {
            if l.len() == 1 {
                J(l[0])
            } else {
                C(l.len(), l[0])
            }
        }).collect()
    }

    #[test]
    fn test_encode_modified() {
        assert_eq!(~[C(4, a), J(b), C(2, c), C(2, a), J(d), C(4, e)],
                   encode_modified(owned_vec!(a a a a b c c a a d e e e e)));
    }

    impl<A:Clone> ModRLE<A> {
        fn expand(self) -> ~[A] {
            match self {
                J(elem) => ~[elem],
                C(count, elem) => vec::build(Some(count), |push| {
                    for _ in range(0, count) {
                        push(elem.clone())
                    }
                })
            }
        }
    }

    /// P12 (**) Decode a run-length encoded list.
    /// Given a run-length code list generated as specified in problem
    /// P11. Construct its uncompressed version.
    fn decode_modified<A:Eq+Clone>(l: ~[ModRLE<A>]) -> ~[A] {
        l.move_iter().flat_map(|entry| entry.expand().move_iter()).collect()
    }

    #[test]
    fn test_decode() {
        assert_eq!(owned_vec!(a a a a b c c a a d e e e e),
                   decode_modified(~[C(4, a), J(b), C(2, c), C(2, a), J(d), C(4, e)]))
    }

    /// P13 (**) Run-length encoding of a list (direct solution).
    ///    Implement the so-called run-length encoding data
    ///    compression method directly. I.e. don't explicitly create
    ///    the sublists containing the duplicates, as in problem P09,
    ///    but only count them. As in problem P11, simplify the result
    ///    list by replacing the singleton lists (1 X) by X.
    ///
    ///    Example:
    ///    * (encode-direct '(a a a a b c c a a d e e e e))
    ///    ((4 A) B (2 C) (2 A) D (4 E))

    fn encode_direct_iter<A:Eq,I:Iterator<A>>(mut it: I) -> ~[ModRLE<A>] {
        fn entry<A>(count: uint, elem: A) -> ModRLE<A> {
            if count == 1 { J(elem) } else { C(count, elem) }
        }

        let mut entries = ~[];
        match it.next() {
            None => return entries,
            Some(mut val) => {
                let mut count = 1;
                loop {
                    match it.next() {
                        None => {
                            entries.push(entry(count, val));
                            return entries;
                        }
                        Some(mut n) => {
                            if n == val {
                                count += 1;
                                continue;
                            } else {
                                swap(&mut val, &mut n);
                                entries.push(entry(count, n));
                                count = 1;
                            }
                        }
                    }
                }
            }
        }
    }

    fn encode_direct<A:Eq>(l: ~[A]) -> ~[ModRLE<A>] {
        encode_direct_iter(l.move_iter())
    }

    #[test]
    fn test_encode_direct() {
        assert_eq!(~[C(4, a), J(b), C(2, c), C(2, a), J(d), C(4, e)],
                   encode_direct(owned_vec!(a a a a b c c a a d e e e e)));
    }

}
