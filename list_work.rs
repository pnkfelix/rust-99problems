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

    def_syms!(a b c d e f g h i
              j k l m n o p q r s t u w y x z)

    def_sym!(alfa)
}

mod vec_work {
    use super::sexp::{Sexp,A,L};
    use super::sym::*;
    use std::iter;
    use std::mem::swap;
    use std::vec;

    /// P01 Find the last *box* of a "list."
    /// Example:
    /// * (my-last '(a b c d))
    /// (D)
    pub fn my_last<'a, A>(vec: &'a [A]) -> &'a [A] { vec.slice_from(vec.len()-1) }

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
    pub fn element_at<'a, A>(v: &'a [A], idx:uint) -> &'a A { &v[idx - 1] }

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
    fn flatten<X>(sexp:Sexp<X>) -> ~[X] {
        match sexp {
            A(atom) => ~[atom],
            L(vec) => vec.move_iter().flat_map(|v|flatten(v).move_iter()).collect(),
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

    fn compress<X:Eq>(vec: ~[X]) -> ~[X] { compress_iter(vec.move_iter()) }

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
        #[allow(unnecessary_parens)]; // in below case, parens clarify,
        let mut lists = ~[];
        match it.next() {
            None => return lists,
            Some(mut val) => {
                let mut curr = ~[];
                loop {
                    match it.next() {
                        None => {
                            curr.push(val);
                            lists.push(curr);
                            return lists;
                        }
                        Some(mut next) => {
                            let same = (next == val);
                            swap(&mut next, &mut val);
                            curr.push(next);
                            if same {
                                continue;
                            } else {
                                lists.push(curr);
                                curr = ~[];
                            }
                        }
                    }
                }
            }
        }
    }

    fn pack<A:Eq>(vec: ~[A]) -> ~[~[A]] { pack_iter(vec.move_iter()) }

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

    fn encode<A:Eq>(vec: ~[A]) -> ~[(uint, A)] {
        pack(vec).move_iter().map(|subvec| (subvec.len(), subvec[0])).collect()
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

    fn encode_modified<A:Eq>(vec: ~[A]) -> ~[ModRLE<A>] {
        pack(vec).move_iter().map(|subvec| {
            if subvec.len() == 1 {
                J(subvec[0])
            } else {
                C(subvec.len(), subvec[0])
            }
        }).collect()
    }

    #[test]
    fn test_encode_modified() {
        assert_eq!(~[C(4, a), J(b), C(2, c), C(2, a), J(d), C(4, e)],
                   encode_modified(owned_vec!(a a a a b c c a a d e e e e)));
    }

    fn repeated<A:Clone>(elem: A, count: uint) -> ~[A] {
        let mut v = vec::build(Some(count), |push| {
            for _ in iter::range(0, count-1) {
                push(elem.clone())
            }
        });
        if count > 0 { v.push(elem) }
        v
    }

    type IotaIter<'a, A> = iter::Unfold<'a, A, (uint, A)>;
    fn repeat_iter<A:Clone>(elem: A, count: uint) -> IotaIter<A> {
        iter::Unfold::new((count, elem), |&(ref mut count, ref elem)| {
            if *count == 0 {
                None
            } else {
                *count -= 1;
                Some(elem.clone())
            }
        })
    }


    impl<A:Clone> ModRLE<A> {
        fn expand(self) -> ~[A] {
            match self {
                J(elem) => ~[elem],
                C(count, elem) => repeated(elem, count)
            }
        }
    }

    /// P12 (**) Decode a run-length encoded list.
    /// Given a run-length code list generated as specified in problem
    /// P11. Construct its uncompressed version.
    fn decode_modified<A:Eq+Clone>(vec: ~[ModRLE<A>]) -> ~[A] {
        vec.move_iter().flat_map(|entry| entry.expand().move_iter()).collect()
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
                        Some(mut next) => {
                            if next == val {
                                count += 1;
                                continue;
                            } else {
                                swap(&mut val, &mut next);
                                entries.push(entry(count, next));
                                count = 1;
                            }
                        }
                    }
                }
            }
        }
    }

    fn encode_direct<A:Eq>(vec: ~[A]) -> ~[ModRLE<A>] {
        encode_direct_iter(vec.move_iter())
    }

    #[test]
    fn test_encode_direct() {
        assert_eq!(~[C(4, a), J(b), C(2, c), C(2, a), J(d), C(4, e)],
                   encode_direct(owned_vec!(a a a a b c c a a d e e e e)));
    }

    /// P14 (*) Duplicate the elements of a list.
    /// Example:
    /// * (dupli '(a b c c d))
    /// (A A B B C C C C D D)
    fn dupli<A:Clone>(vec: ~[A]) -> ~[A] {
        vec.move_iter().flat_map(|elem| (~[elem.clone(), elem]).move_iter()).collect()
    }
    #[test]
    fn test_dupli() {
        assert_eq!(owned_vec!(a a b b c c c c d d),
                   dupli(owned_vec!(a b c c d)))
    }

    /// P15 (**) Replicate the elements of a list a given number of times.
    /// Example:
    /// * (repli '(a b c) 3)
    /// (A A A B B B C C C)
    fn repli<A:Clone>(vec: ~[A], count: uint) -> ~[A] {
        vec.move_iter().flat_map(|elem| repeat_iter(elem, count)).collect()
    }

    #[test]
    fn test_repli() {
        assert_eq!(owned_vec!(a a a b b b c c c),
                   repli(~[a, b, c], 3))
    }


    /// P16 (**) Drop every N'th element from a list.
    /// Example:
    /// * (drop '(a b c d e f g h i k) 3)
    /// (A B D E G H K)
    fn drop_iter<'a, A,I:Iterator<A>>(iter:I, skip:uint) -> iter::Unfold<'a, A, (uint,uint,I)> {
        iter::Unfold::new((skip, skip, iter), |&(skip, ref mut countdown, ref mut iter)| {
            if *countdown == 1 {
                iter.next();
                *countdown = skip;
            }
            *countdown -= 1;
            iter.next()
        })
    }

    fn drop<A>(vec: ~[A], skip: uint) -> ~[A] {
        drop_iter(vec.move_iter(), skip).collect()
    }

    #[test]
    fn test_drop() {
        assert_eq!(     owned_vec!(a b   d e   g h   k),
                   drop(owned_vec!(a b c d e f g h i k), 3))
    }

    /// P17 (*) Split a list into two parts; the length of the first part is given.
    /// Do not use any predefined predicates.
    ///
    /// Example:
    /// * (split '(a b c d e f g h i k) 3)
    /// ( (A B C) (D E F G H I K))
    fn split_sliced<'a, A>(vec: &'a mut [A], count: uint) -> (&'a mut [A], &'a mut [A]) {
        vec.mut_split_at(count)
    }

    fn split_cloning<A:Clone>(mut vec: ~[A], count: uint) -> (~[A], ~[A]) {
        let (lft, rgt) = split_sliced(vec, count);
        (lft.to_owned(), rgt.to_owned())
    }

    fn split<A>(mut vec: ~[A], count: uint) -> (~[A], ~[A]) {
        let (mut lft, mut rgt) = (vec::with_capacity(count), vec::with_capacity(vec.len() - count));
        for _ in iter::range(count, vec.len()).rev() {
            rgt.push(vec.pop().unwrap());
        }
        for _ in iter::range(0, count) {
            lft.push(vec.pop().unwrap());
        }
        lft.reverse();
        rgt.reverse();
        (lft, rgt)
    }

    #[test]
    fn test_split() {
        assert_eq!((owned_vec!(a b c),
                    owned_vec!(d e f g h i k)),
                   split_cloning(owned_vec!(a b c d e f g h i k), 3))
        assert_eq!((owned_vec!(a b c),
                    owned_vec!(d e f g h i k)),
                   split(owned_vec!(a b c d e f g h i k), 3))
    }

    fn slice<A:Clone>(vec: ~[A], i_: uint, k_: uint) -> ~[A] {
        vec.slice(i_-1, k_).to_owned()
    }

    /// P18 (**) Extract a slice from a list.
    /// Given two indices, I and K, the slice is the list containing
    /// the elements between the I'th and K'th element of the original
    /// list (both limits included). Start counting the elements with
    /// 1.
    ///
    /// Example:
    /// * (slice '(a b c d e f g h i k) 3 7)
    /// (C D E F G)
    #[test]
    fn test_slice() {
        assert_eq!(owned_vec!(c d e f g),
                   slice(owned_vec!(a b c d e f g h i k), 3, 7));
    }

    /// P19 (**) Rotate a list N places to the left.
    /// Examples:
    /// * (rotate '(a b c d e f g h) 3)
    /// (D E F G H A B C)
    ///
    /// * (rotate '(a b c d e f g h) -2)
    /// (G H A B C D E F)
    ///
    /// Hint: Use the predefined functions length and append, as well as the result of problem P17.
    fn rotate<A>(vec:~[A], shift: int) -> ~[A] {
        let prefix = if shift >= 0 { shift } else { vec.len() as int + shift } as uint;
        let (lft,mut rgt) = split(vec, prefix);
        rgt.push_all_move(lft);
        rgt
    }

    #[test]
    fn test_rotate() {
        assert_eq!(owned_vec!(d e f g h a b c),
                   rotate(owned_vec!(a b c d e f g h), 3));
        assert_eq!(owned_vec!(g h a b c d e f),
                   rotate(owned_vec!(a b c d e f g h), -2));
    }

    /// P20 (*) Remove the K'th element from a list.
    /// Example:
    /// * (remove-at '(a b c d) 2)
    /// (A C D)
    fn remove_at<A>(mut vec: ~[A], idx: uint) -> ~[A] {
        vec.remove(idx-1);
        vec
    }

    #[test]
    fn test_remove_at() {
        assert_eq!(owned_vec!(a c d),
                   remove_at(owned_vec!(a b c d), 2))
    }

    /// P21 (*) Insert an element at a given position into a list.
    /// Example:
    /// * (insert-at 'alfa '(a b c d) 2)
    /// (A ALFA B C D)
    fn insert_at<A>(elem: A, mut vec: ~[A], idx: uint) -> ~[A] {
        vec.insert(idx-1, elem);
        vec
    }

    #[test]
    fn test_insert_at() {
        assert_eq!(owned_vec!(a alfa b c d),
                   insert_at(alfa, owned_vec!(a b c d), 2))
    }

    /// P22 (*) Create a list containing all integers within a given range.
    /// If first argument is smaller than second, produce a list in decreasing order.
    /// Example:
    /// * (range 4 9)
    /// (4 5 6 7 8 9)
    fn range(low: int, high: int) -> ~[int] {
        iter::range(low, high+1).collect()
    }

    #[test]
    fn test_range() {
        assert_eq!(owned_vec!(4 5 6 7 8 9),
                   range(4, 9))
    }

    /// P23 (**) Extract a given number of randomly selected elements from a list.
    /// The selected items shall be returned in a list.
    /// Example:
    /// * (rnd-select '(a b c d e f g h) 3)
    /// (E D A)
    ///
    /// Hint: Use the built-in random number generator and the result of problem P20.
    fn rnd_select<A>(mut vec: ~[A], count: uint) -> ~[A] {
        use rand;
        use rand::distributions::{IndependentSample, Range};
        let mut result = vec::with_capacity(count);
        let mut rng = rand::task_rng();
        for _ in iter::range(0, count) {
            let range = Range::new(0, vec.len());
            let idx = range.ind_sample(&mut rng);
            match vec.remove(idx) {
                None => break, // or could fail!
                Some(elem) => result.push(elem)
            }
        }
        result
    }

    /// Random functions are hard to test properly, but we can at least sanity check.
    #[test]
    fn test_rnd_select() {
        let domain = owned_vec!(a b c d e f g h);
        let result = rnd_select(domain.clone(), 3);
        assert_eq!(result.len(), 3);
        assert!(result.iter().all(|elem| domain.contains(elem)));
    }

    /// P24 (*) Lotto: Draw N different random numbers from the set 1..M.
    /// The selected numbers shall be returned in a list.
    /// Example:
    /// * (lotto-select 6 49)
    /// (23 1 17 33 21 37)
    ///
    /// Hint: Combine the solutions of problems P22 and P23.
    fn lotto_select(count: uint, max: int) -> ~[int] {
        rnd_select(range(1, max), count)
    }

    fn sorted<A:TotalOrd>(mut vec: ~[A]) -> ~[A] { vec.sort(); vec }
    fn deduped<A:Eq>(mut vec: ~[A]) -> ~[A] { vec.dedup(); vec }
    #[test]
    fn test_lotto_select() {
        let result = lotto_select(6, 49);
        assert_eq!(result.len(), 6);
        assert!(result.iter().all(|&num| 1 <= num && num <= 49));
        assert!(deduped(sorted(result)).len() == 6);
    }

    /// P25 (*) Generate a random permutation of the elements of a list.
    /// Example:
    /// * (rnd-permu '(a b c d e f))
    /// (B A D C E F)
    ///
    /// Hint: Use the solution of problem P23.
    fn rnd_permu<A>(vec: ~[A]) -> ~[A] {
        let len = vec.len();
        rnd_select(vec, len)
    }

    #[test]
    fn test_rnd_permu() {
        let domain = owned_vec!(a b c d e f);
        let result = rnd_permu(domain.clone());
        assert_eq!(sorted(domain), sorted(result));
    }

    /// P26 (**) Generate the combinations of K distinct objects
    /// chosen from the N elements of a list
    ///
    /// In how many ways can a committee of 3 be chosen from a group
    /// of 12 people? We all know that there are C(12,3) = 220
    /// possibilities (C(N,K) denotes the well-known binomial
    /// coefficients). For pure mathematicians, this result may be
    /// great. But we want to really generate all the possibilities in
    /// a list.
    ///
    /// Example:
    /// * (combination 3 '(a b c d e f))
    /// ((A B C) (A B D) (A B E) ... ) 
    fn combination<A:Clone>(count: uint, vec: &[A]) -> ~[~[A]] {
        if vec.len() < count {
            println!("combination A: {} {:?}", count, vec);
            return ~[];
        } else if count == 0 {
            return ~[~[]];
        } else if vec.len() == count {
            println!("combination B: {} {:?}", count, vec);
            return ~[vec.to_owned()];
        } else {
            println!("combination C: {} {:?}", count, vec);
            let mut soln = ~[];
            for idx in iter::range(0u, vec.len()) {
                let mut one_less = vec.to_owned();
                let elem = one_less.remove(idx).unwrap();
                let subprob = combination(count - 1, one_less);
                for readded in subprob.move_iter().map(|mut v| { v.push(elem.clone()); v }) {
                    soln.push(readded);
                }
            }
            println!("combination: {} {:?} soln: {:?}", count, vec, soln);

            return soln;
        }
    }

    #[test]
    fn test_combination() {
        assert_eq!(combination(3, static_vec!(a b c d)),
                   owned_vec!(~[a,b,c] ~[a,b,d] ~[a,c,d] ~[b,c,d]));
    }
}
