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

    def_syms!(alfa aldo beat carla david evi flip gary hugo ida)
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
        let soln;
        if vec.len() < count {
            soln = ~[];
        } else if count == 0 {
            soln = ~[~[]];
        } else if vec.len() == count {
            soln = ~[vec.to_owned()];
        } else {
            println!("combination E: {} {:?}", count, vec);
            let mut to_soln = ~[];
            let mut one_less = vec.to_owned();
            let elem = one_less.pop().unwrap();
            let subprob_readd = combination(count - 1, one_less);
            for readded in subprob_readd.move_iter().map(|mut v| { v.push(elem.clone()); v }) {
                to_soln.push(readded);
            }
            let subprob_noadd = combination(count, one_less);
            for stands_alone in subprob_noadd.move_iter() {
                to_soln.push(stands_alone);
            }
            soln = to_soln;
        }
        println!("combination: {} {:?} soln: {:?}", count, vec, soln);
        return soln;
    }

    #[test]
    fn test_combination() {
        assert_eq!(sorted(combination(3, static_vec!(a b c d))),
                   sorted(owned_vec!(~[a,b,c] ~[a,b,d] ~[a,c,d] ~[b,c,d])));
        assert_eq!(sorted(combination(2, static_vec!(a b c d))),
                   sorted(owned_vec!(~[a,b] ~[a,c] ~[a,d] ~[b,c] ~[b,d] ~[c,d])));
        assert_eq!(combination(5, static_vec!(a b c d)),
                   owned_vec!());
    }

    /// P27 (**) Group the elements of a set into disjoint subsets.
    ///
    /// a) In how many ways can a group of 9 people work in 3 disjoint
    /// subgroups of 2, 3 and 4 persons? Write a function that
    /// generates all the possibilities and returns them in a list.
    ///
    /// Example:
    /// * (group3 '(aldo beat carla david evi flip gary hugo ida))
    /// ( ( (ALDO BEAT) (CARLA DAVID EVI) (FLIP GARY HUGO IDA) ) ... )
    ///
    /// b) Generalize the above predicate in a way that we can specify a list of group sizes and the predicate will return a list of groups.
    ///
    /// Example:
    /// * (group '(aldo beat carla david evi flip gary hugo ida) '(2 2 5))
    /// ( ( (ALDO BEAT) (CARLA DAVID) (EVI FLIP GARY HUGO IDA) ) ... )
    ///
    /// Note that we do not want permutations of the group members;
    /// i.e. ((ALDO BEAT) ...) is the same solution as ((BEAT ALDO)
    /// ...). However, we make a difference between ((ALDO BEAT)
    /// (CARLA DAVID) ...) and ((CARLA DAVID) (ALDO BEAT) ...).
    ///
    /// You may find more about this combinatorial problem in a good
    /// book on discrete mathematics under the term "multinomial
    /// coefficients".
    fn group3<A:Clone>(group: [A, ..9]) -> ~[(~[A], ~[A], ~[A])] {
        type Soln<A> = (~[A], ~[A], ~[A]);
        let mut soln = ~[];
        let mut partials : ~[Soln<A>] = ~[(~[], ~[], ~[])];

        fn soln_complete<A>(&(ref two, ref three, ref four): &(~[A], ~[A], ~[A])) -> bool {
            two.len() == 2 && three.len() == 3 && four.len() == 4
        }

        let new_solns = |elem:A, partials: ~[Soln<A>]| {
            let mut new = ~[];
            for (two, three, four) in partials.move_iter() {
                if   two.len() < 2 { new.push( (two + ~[elem.clone()], three.clone(), four.clone()) ); }
                if three.len() < 3 { new.push( (two.clone(), three + ~[elem.clone()], four.clone()) ); }
                if  four.len() < 4 { new.push( (two.clone(), three.clone(), four + ~[elem.clone()]) ); }
            }
            new
        };

        for elem in group.iter() {
            let new = new_solns(elem.clone(), partials.clone());
            let (full, part) = new.partitioned(soln_complete);
            soln.push_all(full);
            partials.push_all(part);
        }

        return soln;
    }

    fn group<A:Clone>(group: &[A], sizes: &[uint]) -> ~[~[~[A]]] {
        let mut soln = ~[];
        let mut partials : ~[~[~[A]]] = ~[ sizes.iter().map(|_| ~[]).collect() ];

        fn soln_complete<A>(pot_soln: &~[~[A]], sizes: &[uint]) -> bool {
            pot_soln.iter().zip(sizes.iter()).all(|(grp, &len)| grp.len() == len)
        }

        fn new_solns<A:Clone>(elem: A, partials: ~[~[~[A]]], sizes: &[uint]) -> ~[~[~[A]]] {
            let mut new = ~[];
            for soln in partials.move_iter() {
                assert_eq!(soln.len(), sizes.len());
                for idx in iter::range(0, sizes.len()) {
                    if soln[idx].len() < sizes[idx] {
                        let mut new_soln = soln.clone();
                        new_soln[idx].push(elem.clone());
                        new.push(new_soln);
                    }
                }
            }
            new
        };

        for elem in group.iter() {
            let new = new_solns(elem.clone(), partials.clone(), sizes);
            let (full, part) = new.partitioned(|pot_soln|soln_complete(pot_soln, sizes));
            soln.push_all(full);
            partials.push_all(part);
        }

        return soln;
    }

    #[cfg(slow_tests)]
    #[test]
    fn test_group3() {
        // Naive algorithm: generate all possible solutions via
        // permutations of the input, filtering out duplicates by
        // throwing out the ones that have any entry that is not a
        // canonicalized representative for the set (i.e., the ones
        // with any unsorted entry)
        fn slow_group3<A:Clone+TotalOrd>(group: [A, ..9]) -> ~[(~[A], ~[A], ~[A])] {
            let mut solns = ~[];
            for perm in group.permutations() {
                let potential_soln : (~[A], ~[A], ~[A]) = perm_to_pot_soln(perm);
                if is_soln(&potential_soln) {
                    solns.push(potential_soln);
                }
            }
            return solns;
        }

        fn perm_to_pot_soln<A:Clone>(perm: ~[A]) -> (~[A], ~[A], ~[A]) {
            (perm.slice(0,2).to_owned(), perm.slice(2,5).to_owned(), perm.slice(5,9).to_owned())
        }
        fn is_sorted<A:TotalOrd>(vec: &[A]) -> bool {
            for pair in vec.windows(2) {
                if pair[0] > pair[1] {
                    return false;
                }
            }
            return true;
        }
        fn is_soln<A:TotalOrd>(&(ref two, ref three, ref four): &(~[A], ~[A], ~[A])) -> bool {
            two.len() == 2 && three.len() == 3 && four.len() == 4 &&
                [two, three, four].iter().all(|vec|is_sorted(vec.as_slice()))
        }

        // Running this test takes about 2.1 seconds unoptimized (0.38
        // seconds optimized), due to true slowness of slow_group3 to
        // generate the expected output. So I have inlined the
        // slow-generated test data below.
        assert_eq!(sorted(group3(static_vec!(aldo beat carla david evi flip gary hugo ida))),
                   sorted(slow_group3(static_vec!(aldo beat carla david evi flip gary hugo ida))));
    }

    #[test]
    fn test_group() {
        assert_eq!(sorted(group(static_vec!(aldo), [1])),
                   owned_vec!(~[~[aldo]]));

        assert_eq!(sorted(group(static_vec!(aldo beat), [1, 1])),
                   owned_vec!(~[~[aldo], ~[beat]]
                              ~[~[beat], ~[aldo]]));

        assert_eq!(sorted(group(static_vec!(aldo beat carla), [1, 2])),
                   owned_vec!(~[~[aldo], ~[beat, carla]]
                              ~[~[beat], ~[aldo, carla]]
                              ~[~[carla], ~[aldo, beat]]));

        assert_eq!(sorted(group(static_vec!(aldo beat carla david), [1, 3])),
                   owned_vec!(~[~[aldo], ~[beat, carla, david]]
                              ~[~[beat], ~[aldo, carla, david]]
                              ~[~[carla], ~[aldo, beat, david]]
                              ~[~[david], ~[aldo, beat, carla]]));

        assert_eq!(sorted(group(static_vec!(aldo beat carla david evi), [2, 3])),
                   owned_vec!(~[~[aldo, beat], ~[carla, david, evi]]
                              ~[~[aldo, carla], ~[beat, david, evi]]
                              ~[~[aldo, david], ~[beat, carla, evi]]
                              ~[~[aldo, evi], ~[beat, carla, david]]
                              ~[~[beat, carla], ~[aldo, david, evi]]
                              ~[~[beat, david], ~[aldo, carla, evi]]
                              ~[~[beat, evi], ~[aldo, carla, david]]
                              ~[~[carla, david], ~[aldo, beat, evi]]
                              ~[~[carla, evi], ~[aldo, beat, david]]
                              ~[~[david, evi], ~[aldo, beat, carla]]));
    }

    /// P28 (**) Sorting a list of lists according to length of sublists
    ///
    /// a) We suppose that a list contains elements that are lists
    /// themselves. The objective is to sort the elements of this list
    /// according to their length. E.g. short lists first, longer lists
    /// later, or vice versa.
    ///
    /// Example:
    /// * (lsort '((a b c) (d e) (f g h) (d e) (i j k l) (m n) (o)))
    /// ((O) (D E) (D E) (M N) (A B C) (F G H) (I J K L))
    ///
    /// b) Again, we suppose that a list contains elements that are
    /// lists themselves. But this time the objective is to sort the
    /// elements of this list according to their length frequency;
    /// i.e., in the default, where sorting is done ascendingly, lists
    /// with rare lengths are placed first, others with a more
    /// frequent length come later.

    /// Example:
    /// * (lfsort '((a b c) (d e) (f g h) (d e) (i j k l) (m n) (o)))
    /// ((i j k l) (o) (a b c) (f g h) (d e) (d e) (m n))
    ///
    /// Note that in the above example, the first two lists in the
    /// result have length 4 and 1, both lengths appear just once. The
    /// third and forth list have length 3 which appears twice (there
    /// are two list of this length). And finally, the last three
    /// lists have length 2. This is the most frequent length.

    fn lsort<'a, 'b, A>(vec: &'a mut [&'b [A]]) -> &'a mut [&'b [A]] {
        vec.sort_by(|lft, rgt| lft.len().cmp(&rgt.len()));
        vec
    }

    #[test]
    fn test_lsort() {
        let expect = [&[o], &[d, e], &[d, e], &[m, n], &[a, b, c], &[f, g, h], &[i, j, k, l]];
        let mut input  = [&[a, b, c], &[d, e], &[f, g, h], &[d, e], &[i, j, k, l], &[m, n], &[o]];
        assert_eq!(expect.to_owned(),
                   lsort(input).to_owned());
    }

    fn lfsort<'a, 'b, A>(vec: &'a mut [&'b [A]]) -> &'a mut [&'b [A]] {
        use std::cmp;
        let freq = {
            let max_len = vec.iter().map(|elem|elem.len()).fold(0, cmp::max);
            let mut freq = vec::from_elem(max_len+1, 0);
            for elem in vec.iter() {
                freq[elem.len()] += 1;
            }
            freq
        };
        vec.sort_by(|lft, rgt| freq[lft.len()].cmp(&freq[rgt.len()]));
        vec
    }

    #[test]
    fn test_lfsort() {
        let expect = [&[i, j, k, l], &[o], &[a, b, c], &[f, g, h], &[d, e], &[d, e], &[m, n]];
        let mut input = [&[a, b, c], &[d, e], &[f, g, h], &[d, e], &[i, j, k, l], &[m, n], &[o]];
        assert_eq!(expect.to_owned(), lfsort(input).to_owned());
    }

    #[cfg(not(slow_tests))]
    #[test]
    fn test_group3() {
        // With this version, running the test suite takes about 0.54
        // seconds unoptimized (0.19 seconds optimized).
        assert_eq!(sorted(group3(static_vec!(aldo beat carla david evi flip gary hugo ida))),
                   actual_soln.iter().map(static_soln_to_owned).collect());

        fn static_soln_to_owned<A:Clone>(ss: &([A, ..2], [A, ..3], [A, ..4])) -> (~[A],~[A],~[A]) {
            let &(ref two, ref three, ref four) = ss;
            (two.to_owned(), three.to_owned(), four.to_owned())
        }

        static actual_soln : [([&'static str, ..2],
                               [&'static str, ..3],
                               [&'static str, ..4]), ..1260] =
            // Below was generated via above:
            // sorted(slow_group3(static_vec!(aldo beat carla david evi flip gary hugo ida))));
            [([aldo, beat], [carla, david, evi], [flip, gary, hugo, ida]),
             ([aldo, beat], [carla, david, flip], [evi, gary, hugo, ida]),
             ([aldo, beat], [carla, david, gary], [evi, flip, hugo, ida]),
             ([aldo, beat], [carla, david, hugo], [evi, flip, gary, ida]),
             ([aldo, beat], [carla, david, ida], [evi, flip, gary, hugo]),
             ([aldo, beat], [carla, evi, flip], [david, gary, hugo, ida]),
             ([aldo, beat], [carla, evi, gary], [david, flip, hugo, ida]),
             ([aldo, beat], [carla, evi, hugo], [david, flip, gary, ida]),
             ([aldo, beat], [carla, evi, ida], [david, flip, gary, hugo]),
             ([aldo, beat], [carla, flip, gary], [david, evi, hugo, ida]),
             ([aldo, beat], [carla, flip, hugo], [david, evi, gary, ida]),
             ([aldo, beat], [carla, flip, ida], [david, evi, gary, hugo]),
             ([aldo, beat], [carla, gary, hugo], [david, evi, flip, ida]),
             ([aldo, beat], [carla, gary, ida], [david, evi, flip, hugo]),
             ([aldo, beat], [carla, hugo, ida], [david, evi, flip, gary]),
             ([aldo, beat], [david, evi, flip], [carla, gary, hugo, ida]),
             ([aldo, beat], [david, evi, gary], [carla, flip, hugo, ida]),
             ([aldo, beat], [david, evi, hugo], [carla, flip, gary, ida]),
             ([aldo, beat], [david, evi, ida], [carla, flip, gary, hugo]),
             ([aldo, beat], [david, flip, gary], [carla, evi, hugo, ida]),
             ([aldo, beat], [david, flip, hugo], [carla, evi, gary, ida]),
             ([aldo, beat], [david, flip, ida], [carla, evi, gary, hugo]),
             ([aldo, beat], [david, gary, hugo], [carla, evi, flip, ida]),
             ([aldo, beat], [david, gary, ida], [carla, evi, flip, hugo]),
             ([aldo, beat], [david, hugo, ida], [carla, evi, flip, gary]),
             ([aldo, beat], [evi, flip, gary], [carla, david, hugo, ida]),
             ([aldo, beat], [evi, flip, hugo], [carla, david, gary, ida]),
             ([aldo, beat], [evi, flip, ida], [carla, david, gary, hugo]),
             ([aldo, beat], [evi, gary, hugo], [carla, david, flip, ida]),
             ([aldo, beat], [evi, gary, ida], [carla, david, flip, hugo]),
             ([aldo, beat], [evi, hugo, ida], [carla, david, flip, gary]),
             ([aldo, beat], [flip, gary, hugo], [carla, david, evi, ida]),
             ([aldo, beat], [flip, gary, ida], [carla, david, evi, hugo]),
             ([aldo, beat], [flip, hugo, ida], [carla, david, evi, gary]),
             ([aldo, beat], [gary, hugo, ida], [carla, david, evi, flip]),
             ([aldo, carla], [beat, david, evi], [flip, gary, hugo, ida]),
             ([aldo, carla], [beat, david, flip], [evi, gary, hugo, ida]),
             ([aldo, carla], [beat, david, gary], [evi, flip, hugo, ida]),
             ([aldo, carla], [beat, david, hugo], [evi, flip, gary, ida]),
             ([aldo, carla], [beat, david, ida], [evi, flip, gary, hugo]),
             ([aldo, carla], [beat, evi, flip], [david, gary, hugo, ida]),
             ([aldo, carla], [beat, evi, gary], [david, flip, hugo, ida]),
             ([aldo, carla], [beat, evi, hugo], [david, flip, gary, ida]),
             ([aldo, carla], [beat, evi, ida], [david, flip, gary, hugo]),
             ([aldo, carla], [beat, flip, gary], [david, evi, hugo, ida]),
             ([aldo, carla], [beat, flip, hugo], [david, evi, gary, ida]),
             ([aldo, carla], [beat, flip, ida], [david, evi, gary, hugo]),
             ([aldo, carla], [beat, gary, hugo], [david, evi, flip, ida]),
             ([aldo, carla], [beat, gary, ida], [david, evi, flip, hugo]),
             ([aldo, carla], [beat, hugo, ida], [david, evi, flip, gary]),
             ([aldo, carla], [david, evi, flip], [beat, gary, hugo, ida]),
             ([aldo, carla], [david, evi, gary], [beat, flip, hugo, ida]),
             ([aldo, carla], [david, evi, hugo], [beat, flip, gary, ida]),
             ([aldo, carla], [david, evi, ida], [beat, flip, gary, hugo]),
             ([aldo, carla], [david, flip, gary], [beat, evi, hugo, ida]),
             ([aldo, carla], [david, flip, hugo], [beat, evi, gary, ida]),
             ([aldo, carla], [david, flip, ida], [beat, evi, gary, hugo]),
             ([aldo, carla], [david, gary, hugo], [beat, evi, flip, ida]),
             ([aldo, carla], [david, gary, ida], [beat, evi, flip, hugo]),
             ([aldo, carla], [david, hugo, ida], [beat, evi, flip, gary]),
             ([aldo, carla], [evi, flip, gary], [beat, david, hugo, ida]),
             ([aldo, carla], [evi, flip, hugo], [beat, david, gary, ida]),
             ([aldo, carla], [evi, flip, ida], [beat, david, gary, hugo]),
             ([aldo, carla], [evi, gary, hugo], [beat, david, flip, ida]),
             ([aldo, carla], [evi, gary, ida], [beat, david, flip, hugo]),
             ([aldo, carla], [evi, hugo, ida], [beat, david, flip, gary]),
             ([aldo, carla], [flip, gary, hugo], [beat, david, evi, ida]),
             ([aldo, carla], [flip, gary, ida], [beat, david, evi, hugo]),
             ([aldo, carla], [flip, hugo, ida], [beat, david, evi, gary]),
             ([aldo, carla], [gary, hugo, ida], [beat, david, evi, flip]),
             ([aldo, david], [beat, carla, evi], [flip, gary, hugo, ida]),
             ([aldo, david], [beat, carla, flip], [evi, gary, hugo, ida]),
             ([aldo, david], [beat, carla, gary], [evi, flip, hugo, ida]),
             ([aldo, david], [beat, carla, hugo], [evi, flip, gary, ida]),
             ([aldo, david], [beat, carla, ida], [evi, flip, gary, hugo]),
             ([aldo, david], [beat, evi, flip], [carla, gary, hugo, ida]),
             ([aldo, david], [beat, evi, gary], [carla, flip, hugo, ida]),
             ([aldo, david], [beat, evi, hugo], [carla, flip, gary, ida]),
             ([aldo, david], [beat, evi, ida], [carla, flip, gary, hugo]),
             ([aldo, david], [beat, flip, gary], [carla, evi, hugo, ida]),
             ([aldo, david], [beat, flip, hugo], [carla, evi, gary, ida]),
             ([aldo, david], [beat, flip, ida], [carla, evi, gary, hugo]),
             ([aldo, david], [beat, gary, hugo], [carla, evi, flip, ida]),
             ([aldo, david], [beat, gary, ida], [carla, evi, flip, hugo]),
             ([aldo, david], [beat, hugo, ida], [carla, evi, flip, gary]),
             ([aldo, david], [carla, evi, flip], [beat, gary, hugo, ida]),
             ([aldo, david], [carla, evi, gary], [beat, flip, hugo, ida]),
             ([aldo, david], [carla, evi, hugo], [beat, flip, gary, ida]),
             ([aldo, david], [carla, evi, ida], [beat, flip, gary, hugo]),
             ([aldo, david], [carla, flip, gary], [beat, evi, hugo, ida]),
             ([aldo, david], [carla, flip, hugo], [beat, evi, gary, ida]),
             ([aldo, david], [carla, flip, ida], [beat, evi, gary, hugo]),
             ([aldo, david], [carla, gary, hugo], [beat, evi, flip, ida]),
             ([aldo, david], [carla, gary, ida], [beat, evi, flip, hugo]),
             ([aldo, david], [carla, hugo, ida], [beat, evi, flip, gary]),
             ([aldo, david], [evi, flip, gary], [beat, carla, hugo, ida]),
             ([aldo, david], [evi, flip, hugo], [beat, carla, gary, ida]),
             ([aldo, david], [evi, flip, ida], [beat, carla, gary, hugo]),
             ([aldo, david], [evi, gary, hugo], [beat, carla, flip, ida]),
             ([aldo, david], [evi, gary, ida], [beat, carla, flip, hugo]),
             ([aldo, david], [evi, hugo, ida], [beat, carla, flip, gary]),
             ([aldo, david], [flip, gary, hugo], [beat, carla, evi, ida]),
             ([aldo, david], [flip, gary, ida], [beat, carla, evi, hugo]),
             ([aldo, david], [flip, hugo, ida], [beat, carla, evi, gary]),
             ([aldo, david], [gary, hugo, ida], [beat, carla, evi, flip]),
             ([aldo, evi], [beat, carla, david], [flip, gary, hugo, ida]),
             ([aldo, evi], [beat, carla, flip], [david, gary, hugo, ida]),
             ([aldo, evi], [beat, carla, gary], [david, flip, hugo, ida]),
             ([aldo, evi], [beat, carla, hugo], [david, flip, gary, ida]),
             ([aldo, evi], [beat, carla, ida], [david, flip, gary, hugo]),
             ([aldo, evi], [beat, david, flip], [carla, gary, hugo, ida]),
             ([aldo, evi], [beat, david, gary], [carla, flip, hugo, ida]),
             ([aldo, evi], [beat, david, hugo], [carla, flip, gary, ida]),
             ([aldo, evi], [beat, david, ida], [carla, flip, gary, hugo]),
             ([aldo, evi], [beat, flip, gary], [carla, david, hugo, ida]),
             ([aldo, evi], [beat, flip, hugo], [carla, david, gary, ida]),
             ([aldo, evi], [beat, flip, ida], [carla, david, gary, hugo]),
             ([aldo, evi], [beat, gary, hugo], [carla, david, flip, ida]),
             ([aldo, evi], [beat, gary, ida], [carla, david, flip, hugo]),
             ([aldo, evi], [beat, hugo, ida], [carla, david, flip, gary]),
             ([aldo, evi], [carla, david, flip], [beat, gary, hugo, ida]),
             ([aldo, evi], [carla, david, gary], [beat, flip, hugo, ida]),
             ([aldo, evi], [carla, david, hugo], [beat, flip, gary, ida]),
             ([aldo, evi], [carla, david, ida], [beat, flip, gary, hugo]),
             ([aldo, evi], [carla, flip, gary], [beat, david, hugo, ida]),
             ([aldo, evi], [carla, flip, hugo], [beat, david, gary, ida]),
             ([aldo, evi], [carla, flip, ida], [beat, david, gary, hugo]),
             ([aldo, evi], [carla, gary, hugo], [beat, david, flip, ida]),
             ([aldo, evi], [carla, gary, ida], [beat, david, flip, hugo]),
             ([aldo, evi], [carla, hugo, ida], [beat, david, flip, gary]),
             ([aldo, evi], [david, flip, gary], [beat, carla, hugo, ida]),
             ([aldo, evi], [david, flip, hugo], [beat, carla, gary, ida]),
             ([aldo, evi], [david, flip, ida], [beat, carla, gary, hugo]),
             ([aldo, evi], [david, gary, hugo], [beat, carla, flip, ida]),
             ([aldo, evi], [david, gary, ida], [beat, carla, flip, hugo]),
             ([aldo, evi], [david, hugo, ida], [beat, carla, flip, gary]),
             ([aldo, evi], [flip, gary, hugo], [beat, carla, david, ida]),
             ([aldo, evi], [flip, gary, ida], [beat, carla, david, hugo]),
             ([aldo, evi], [flip, hugo, ida], [beat, carla, david, gary]),
             ([aldo, evi], [gary, hugo, ida], [beat, carla, david, flip]),
             ([aldo, flip], [beat, carla, david], [evi, gary, hugo, ida]),
             ([aldo, flip], [beat, carla, evi], [david, gary, hugo, ida]),
             ([aldo, flip], [beat, carla, gary], [david, evi, hugo, ida]),
             ([aldo, flip], [beat, carla, hugo], [david, evi, gary, ida]),
             ([aldo, flip], [beat, carla, ida], [david, evi, gary, hugo]),
             ([aldo, flip], [beat, david, evi], [carla, gary, hugo, ida]),
             ([aldo, flip], [beat, david, gary], [carla, evi, hugo, ida]),
             ([aldo, flip], [beat, david, hugo], [carla, evi, gary, ida]),
             ([aldo, flip], [beat, david, ida], [carla, evi, gary, hugo]),
             ([aldo, flip], [beat, evi, gary], [carla, david, hugo, ida]),
             ([aldo, flip], [beat, evi, hugo], [carla, david, gary, ida]),
             ([aldo, flip], [beat, evi, ida], [carla, david, gary, hugo]),
             ([aldo, flip], [beat, gary, hugo], [carla, david, evi, ida]),
             ([aldo, flip], [beat, gary, ida], [carla, david, evi, hugo]),
             ([aldo, flip], [beat, hugo, ida], [carla, david, evi, gary]),
             ([aldo, flip], [carla, david, evi], [beat, gary, hugo, ida]),
             ([aldo, flip], [carla, david, gary], [beat, evi, hugo, ida]),
             ([aldo, flip], [carla, david, hugo], [beat, evi, gary, ida]),
             ([aldo, flip], [carla, david, ida], [beat, evi, gary, hugo]),
             ([aldo, flip], [carla, evi, gary], [beat, david, hugo, ida]),
             ([aldo, flip], [carla, evi, hugo], [beat, david, gary, ida]),
             ([aldo, flip], [carla, evi, ida], [beat, david, gary, hugo]),
             ([aldo, flip], [carla, gary, hugo], [beat, david, evi, ida]),
             ([aldo, flip], [carla, gary, ida], [beat, david, evi, hugo]),
             ([aldo, flip], [carla, hugo, ida], [beat, david, evi, gary]),
             ([aldo, flip], [david, evi, gary], [beat, carla, hugo, ida]),
             ([aldo, flip], [david, evi, hugo], [beat, carla, gary, ida]),
             ([aldo, flip], [david, evi, ida], [beat, carla, gary, hugo]),
             ([aldo, flip], [david, gary, hugo], [beat, carla, evi, ida]),
             ([aldo, flip], [david, gary, ida], [beat, carla, evi, hugo]),
             ([aldo, flip], [david, hugo, ida], [beat, carla, evi, gary]),
             ([aldo, flip], [evi, gary, hugo], [beat, carla, david, ida]),
             ([aldo, flip], [evi, gary, ida], [beat, carla, david, hugo]),
             ([aldo, flip], [evi, hugo, ida], [beat, carla, david, gary]),
             ([aldo, flip], [gary, hugo, ida], [beat, carla, david, evi]),
             ([aldo, gary], [beat, carla, david], [evi, flip, hugo, ida]),
             ([aldo, gary], [beat, carla, evi], [david, flip, hugo, ida]),
             ([aldo, gary], [beat, carla, flip], [david, evi, hugo, ida]),
             ([aldo, gary], [beat, carla, hugo], [david, evi, flip, ida]),
             ([aldo, gary], [beat, carla, ida], [david, evi, flip, hugo]),
             ([aldo, gary], [beat, david, evi], [carla, flip, hugo, ida]),
             ([aldo, gary], [beat, david, flip], [carla, evi, hugo, ida]),
             ([aldo, gary], [beat, david, hugo], [carla, evi, flip, ida]),
             ([aldo, gary], [beat, david, ida], [carla, evi, flip, hugo]),
             ([aldo, gary], [beat, evi, flip], [carla, david, hugo, ida]),
             ([aldo, gary], [beat, evi, hugo], [carla, david, flip, ida]),
             ([aldo, gary], [beat, evi, ida], [carla, david, flip, hugo]),
             ([aldo, gary], [beat, flip, hugo], [carla, david, evi, ida]),
             ([aldo, gary], [beat, flip, ida], [carla, david, evi, hugo]),
             ([aldo, gary], [beat, hugo, ida], [carla, david, evi, flip]),
             ([aldo, gary], [carla, david, evi], [beat, flip, hugo, ida]),
             ([aldo, gary], [carla, david, flip], [beat, evi, hugo, ida]),
             ([aldo, gary], [carla, david, hugo], [beat, evi, flip, ida]),
             ([aldo, gary], [carla, david, ida], [beat, evi, flip, hugo]),
             ([aldo, gary], [carla, evi, flip], [beat, david, hugo, ida]),
             ([aldo, gary], [carla, evi, hugo], [beat, david, flip, ida]),
             ([aldo, gary], [carla, evi, ida], [beat, david, flip, hugo]),
             ([aldo, gary], [carla, flip, hugo], [beat, david, evi, ida]),
             ([aldo, gary], [carla, flip, ida], [beat, david, evi, hugo]),
             ([aldo, gary], [carla, hugo, ida], [beat, david, evi, flip]),
             ([aldo, gary], [david, evi, flip], [beat, carla, hugo, ida]),
             ([aldo, gary], [david, evi, hugo], [beat, carla, flip, ida]),
             ([aldo, gary], [david, evi, ida], [beat, carla, flip, hugo]),
             ([aldo, gary], [david, flip, hugo], [beat, carla, evi, ida]),
             ([aldo, gary], [david, flip, ida], [beat, carla, evi, hugo]),
             ([aldo, gary], [david, hugo, ida], [beat, carla, evi, flip]),
             ([aldo, gary], [evi, flip, hugo], [beat, carla, david, ida]),
             ([aldo, gary], [evi, flip, ida], [beat, carla, david, hugo]),
             ([aldo, gary], [evi, hugo, ida], [beat, carla, david, flip]),
             ([aldo, gary], [flip, hugo, ida], [beat, carla, david, evi]),
             ([aldo, hugo], [beat, carla, david], [evi, flip, gary, ida]),
             ([aldo, hugo], [beat, carla, evi], [david, flip, gary, ida]),
             ([aldo, hugo], [beat, carla, flip], [david, evi, gary, ida]),
             ([aldo, hugo], [beat, carla, gary], [david, evi, flip, ida]),
             ([aldo, hugo], [beat, carla, ida], [david, evi, flip, gary]),
             ([aldo, hugo], [beat, david, evi], [carla, flip, gary, ida]),
             ([aldo, hugo], [beat, david, flip], [carla, evi, gary, ida]),
             ([aldo, hugo], [beat, david, gary], [carla, evi, flip, ida]),
             ([aldo, hugo], [beat, david, ida], [carla, evi, flip, gary]),
             ([aldo, hugo], [beat, evi, flip], [carla, david, gary, ida]),
             ([aldo, hugo], [beat, evi, gary], [carla, david, flip, ida]),
             ([aldo, hugo], [beat, evi, ida], [carla, david, flip, gary]),
             ([aldo, hugo], [beat, flip, gary], [carla, david, evi, ida]),
             ([aldo, hugo], [beat, flip, ida], [carla, david, evi, gary]),
             ([aldo, hugo], [beat, gary, ida], [carla, david, evi, flip]),
             ([aldo, hugo], [carla, david, evi], [beat, flip, gary, ida]),
             ([aldo, hugo], [carla, david, flip], [beat, evi, gary, ida]),
             ([aldo, hugo], [carla, david, gary], [beat, evi, flip, ida]),
             ([aldo, hugo], [carla, david, ida], [beat, evi, flip, gary]),
             ([aldo, hugo], [carla, evi, flip], [beat, david, gary, ida]),
             ([aldo, hugo], [carla, evi, gary], [beat, david, flip, ida]),
             ([aldo, hugo], [carla, evi, ida], [beat, david, flip, gary]),
             ([aldo, hugo], [carla, flip, gary], [beat, david, evi, ida]),
             ([aldo, hugo], [carla, flip, ida], [beat, david, evi, gary]),
             ([aldo, hugo], [carla, gary, ida], [beat, david, evi, flip]),
             ([aldo, hugo], [david, evi, flip], [beat, carla, gary, ida]),
             ([aldo, hugo], [david, evi, gary], [beat, carla, flip, ida]),
             ([aldo, hugo], [david, evi, ida], [beat, carla, flip, gary]),
             ([aldo, hugo], [david, flip, gary], [beat, carla, evi, ida]),
             ([aldo, hugo], [david, flip, ida], [beat, carla, evi, gary]),
             ([aldo, hugo], [david, gary, ida], [beat, carla, evi, flip]),
             ([aldo, hugo], [evi, flip, gary], [beat, carla, david, ida]),
             ([aldo, hugo], [evi, flip, ida], [beat, carla, david, gary]),
             ([aldo, hugo], [evi, gary, ida], [beat, carla, david, flip]),
             ([aldo, hugo], [flip, gary, ida], [beat, carla, david, evi]),
             ([aldo, ida], [beat, carla, david], [evi, flip, gary, hugo]),
             ([aldo, ida], [beat, carla, evi], [david, flip, gary, hugo]),
             ([aldo, ida], [beat, carla, flip], [david, evi, gary, hugo]),
             ([aldo, ida], [beat, carla, gary], [david, evi, flip, hugo]),
             ([aldo, ida], [beat, carla, hugo], [david, evi, flip, gary]),
             ([aldo, ida], [beat, david, evi], [carla, flip, gary, hugo]),
             ([aldo, ida], [beat, david, flip], [carla, evi, gary, hugo]),
             ([aldo, ida], [beat, david, gary], [carla, evi, flip, hugo]),
             ([aldo, ida], [beat, david, hugo], [carla, evi, flip, gary]),
             ([aldo, ida], [beat, evi, flip], [carla, david, gary, hugo]),
             ([aldo, ida], [beat, evi, gary], [carla, david, flip, hugo]),
             ([aldo, ida], [beat, evi, hugo], [carla, david, flip, gary]),
             ([aldo, ida], [beat, flip, gary], [carla, david, evi, hugo]),
             ([aldo, ida], [beat, flip, hugo], [carla, david, evi, gary]),
             ([aldo, ida], [beat, gary, hugo], [carla, david, evi, flip]),
             ([aldo, ida], [carla, david, evi], [beat, flip, gary, hugo]),
             ([aldo, ida], [carla, david, flip], [beat, evi, gary, hugo]),
             ([aldo, ida], [carla, david, gary], [beat, evi, flip, hugo]),
             ([aldo, ida], [carla, david, hugo], [beat, evi, flip, gary]),
             ([aldo, ida], [carla, evi, flip], [beat, david, gary, hugo]),
             ([aldo, ida], [carla, evi, gary], [beat, david, flip, hugo]),
             ([aldo, ida], [carla, evi, hugo], [beat, david, flip, gary]),
             ([aldo, ida], [carla, flip, gary], [beat, david, evi, hugo]),
             ([aldo, ida], [carla, flip, hugo], [beat, david, evi, gary]),
             ([aldo, ida], [carla, gary, hugo], [beat, david, evi, flip]),
             ([aldo, ida], [david, evi, flip], [beat, carla, gary, hugo]),
             ([aldo, ida], [david, evi, gary], [beat, carla, flip, hugo]),
             ([aldo, ida], [david, evi, hugo], [beat, carla, flip, gary]),
             ([aldo, ida], [david, flip, gary], [beat, carla, evi, hugo]),
             ([aldo, ida], [david, flip, hugo], [beat, carla, evi, gary]),
             ([aldo, ida], [david, gary, hugo], [beat, carla, evi, flip]),
             ([aldo, ida], [evi, flip, gary], [beat, carla, david, hugo]),
             ([aldo, ida], [evi, flip, hugo], [beat, carla, david, gary]),
             ([aldo, ida], [evi, gary, hugo], [beat, carla, david, flip]),
             ([aldo, ida], [flip, gary, hugo], [beat, carla, david, evi]),
             ([beat, carla], [aldo, david, evi], [flip, gary, hugo, ida]),
             ([beat, carla], [aldo, david, flip], [evi, gary, hugo, ida]),
             ([beat, carla], [aldo, david, gary], [evi, flip, hugo, ida]),
             ([beat, carla], [aldo, david, hugo], [evi, flip, gary, ida]),
             ([beat, carla], [aldo, david, ida], [evi, flip, gary, hugo]),
             ([beat, carla], [aldo, evi, flip], [david, gary, hugo, ida]),
             ([beat, carla], [aldo, evi, gary], [david, flip, hugo, ida]),
             ([beat, carla], [aldo, evi, hugo], [david, flip, gary, ida]),
             ([beat, carla], [aldo, evi, ida], [david, flip, gary, hugo]),
             ([beat, carla], [aldo, flip, gary], [david, evi, hugo, ida]),
             ([beat, carla], [aldo, flip, hugo], [david, evi, gary, ida]),
             ([beat, carla], [aldo, flip, ida], [david, evi, gary, hugo]),
             ([beat, carla], [aldo, gary, hugo], [david, evi, flip, ida]),
             ([beat, carla], [aldo, gary, ida], [david, evi, flip, hugo]),
             ([beat, carla], [aldo, hugo, ida], [david, evi, flip, gary]),
             ([beat, carla], [david, evi, flip], [aldo, gary, hugo, ida]),
             ([beat, carla], [david, evi, gary], [aldo, flip, hugo, ida]),
             ([beat, carla], [david, evi, hugo], [aldo, flip, gary, ida]),
             ([beat, carla], [david, evi, ida], [aldo, flip, gary, hugo]),
             ([beat, carla], [david, flip, gary], [aldo, evi, hugo, ida]),
             ([beat, carla], [david, flip, hugo], [aldo, evi, gary, ida]),
             ([beat, carla], [david, flip, ida], [aldo, evi, gary, hugo]),
             ([beat, carla], [david, gary, hugo], [aldo, evi, flip, ida]),
             ([beat, carla], [david, gary, ida], [aldo, evi, flip, hugo]),
             ([beat, carla], [david, hugo, ida], [aldo, evi, flip, gary]),
             ([beat, carla], [evi, flip, gary], [aldo, david, hugo, ida]),
             ([beat, carla], [evi, flip, hugo], [aldo, david, gary, ida]),
             ([beat, carla], [evi, flip, ida], [aldo, david, gary, hugo]),
             ([beat, carla], [evi, gary, hugo], [aldo, david, flip, ida]),
             ([beat, carla], [evi, gary, ida], [aldo, david, flip, hugo]),
             ([beat, carla], [evi, hugo, ida], [aldo, david, flip, gary]),
             ([beat, carla], [flip, gary, hugo], [aldo, david, evi, ida]),
             ([beat, carla], [flip, gary, ida], [aldo, david, evi, hugo]),
             ([beat, carla], [flip, hugo, ida], [aldo, david, evi, gary]),
             ([beat, carla], [gary, hugo, ida], [aldo, david, evi, flip]),
             ([beat, david], [aldo, carla, evi], [flip, gary, hugo, ida]),
             ([beat, david], [aldo, carla, flip], [evi, gary, hugo, ida]),
             ([beat, david], [aldo, carla, gary], [evi, flip, hugo, ida]),
             ([beat, david], [aldo, carla, hugo], [evi, flip, gary, ida]),
             ([beat, david], [aldo, carla, ida], [evi, flip, gary, hugo]),
             ([beat, david], [aldo, evi, flip], [carla, gary, hugo, ida]),
             ([beat, david], [aldo, evi, gary], [carla, flip, hugo, ida]),
             ([beat, david], [aldo, evi, hugo], [carla, flip, gary, ida]),
             ([beat, david], [aldo, evi, ida], [carla, flip, gary, hugo]),
             ([beat, david], [aldo, flip, gary], [carla, evi, hugo, ida]),
             ([beat, david], [aldo, flip, hugo], [carla, evi, gary, ida]),
             ([beat, david], [aldo, flip, ida], [carla, evi, gary, hugo]),
             ([beat, david], [aldo, gary, hugo], [carla, evi, flip, ida]),
             ([beat, david], [aldo, gary, ida], [carla, evi, flip, hugo]),
             ([beat, david], [aldo, hugo, ida], [carla, evi, flip, gary]),
             ([beat, david], [carla, evi, flip], [aldo, gary, hugo, ida]),
             ([beat, david], [carla, evi, gary], [aldo, flip, hugo, ida]),
             ([beat, david], [carla, evi, hugo], [aldo, flip, gary, ida]),
             ([beat, david], [carla, evi, ida], [aldo, flip, gary, hugo]),
             ([beat, david], [carla, flip, gary], [aldo, evi, hugo, ida]),
             ([beat, david], [carla, flip, hugo], [aldo, evi, gary, ida]),
             ([beat, david], [carla, flip, ida], [aldo, evi, gary, hugo]),
             ([beat, david], [carla, gary, hugo], [aldo, evi, flip, ida]),
             ([beat, david], [carla, gary, ida], [aldo, evi, flip, hugo]),
             ([beat, david], [carla, hugo, ida], [aldo, evi, flip, gary]),
             ([beat, david], [evi, flip, gary], [aldo, carla, hugo, ida]),
             ([beat, david], [evi, flip, hugo], [aldo, carla, gary, ida]),
             ([beat, david], [evi, flip, ida], [aldo, carla, gary, hugo]),
             ([beat, david], [evi, gary, hugo], [aldo, carla, flip, ida]),
             ([beat, david], [evi, gary, ida], [aldo, carla, flip, hugo]),
             ([beat, david], [evi, hugo, ida], [aldo, carla, flip, gary]),
             ([beat, david], [flip, gary, hugo], [aldo, carla, evi, ida]),
             ([beat, david], [flip, gary, ida], [aldo, carla, evi, hugo]),
             ([beat, david], [flip, hugo, ida], [aldo, carla, evi, gary]),
             ([beat, david], [gary, hugo, ida], [aldo, carla, evi, flip]),
             ([beat, evi], [aldo, carla, david], [flip, gary, hugo, ida]),
             ([beat, evi], [aldo, carla, flip], [david, gary, hugo, ida]),
             ([beat, evi], [aldo, carla, gary], [david, flip, hugo, ida]),
             ([beat, evi], [aldo, carla, hugo], [david, flip, gary, ida]),
             ([beat, evi], [aldo, carla, ida], [david, flip, gary, hugo]),
             ([beat, evi], [aldo, david, flip], [carla, gary, hugo, ida]),
             ([beat, evi], [aldo, david, gary], [carla, flip, hugo, ida]),
             ([beat, evi], [aldo, david, hugo], [carla, flip, gary, ida]),
             ([beat, evi], [aldo, david, ida], [carla, flip, gary, hugo]),
             ([beat, evi], [aldo, flip, gary], [carla, david, hugo, ida]),
             ([beat, evi], [aldo, flip, hugo], [carla, david, gary, ida]),
             ([beat, evi], [aldo, flip, ida], [carla, david, gary, hugo]),
             ([beat, evi], [aldo, gary, hugo], [carla, david, flip, ida]),
             ([beat, evi], [aldo, gary, ida], [carla, david, flip, hugo]),
             ([beat, evi], [aldo, hugo, ida], [carla, david, flip, gary]),
             ([beat, evi], [carla, david, flip], [aldo, gary, hugo, ida]),
             ([beat, evi], [carla, david, gary], [aldo, flip, hugo, ida]),
             ([beat, evi], [carla, david, hugo], [aldo, flip, gary, ida]),
             ([beat, evi], [carla, david, ida], [aldo, flip, gary, hugo]),
             ([beat, evi], [carla, flip, gary], [aldo, david, hugo, ida]),
             ([beat, evi], [carla, flip, hugo], [aldo, david, gary, ida]),
             ([beat, evi], [carla, flip, ida], [aldo, david, gary, hugo]),
             ([beat, evi], [carla, gary, hugo], [aldo, david, flip, ida]),
             ([beat, evi], [carla, gary, ida], [aldo, david, flip, hugo]),
             ([beat, evi], [carla, hugo, ida], [aldo, david, flip, gary]),
             ([beat, evi], [david, flip, gary], [aldo, carla, hugo, ida]),
             ([beat, evi], [david, flip, hugo], [aldo, carla, gary, ida]),
             ([beat, evi], [david, flip, ida], [aldo, carla, gary, hugo]),
             ([beat, evi], [david, gary, hugo], [aldo, carla, flip, ida]),
             ([beat, evi], [david, gary, ida], [aldo, carla, flip, hugo]),
             ([beat, evi], [david, hugo, ida], [aldo, carla, flip, gary]),
             ([beat, evi], [flip, gary, hugo], [aldo, carla, david, ida]),
             ([beat, evi], [flip, gary, ida], [aldo, carla, david, hugo]),
             ([beat, evi], [flip, hugo, ida], [aldo, carla, david, gary]),
             ([beat, evi], [gary, hugo, ida], [aldo, carla, david, flip]),
             ([beat, flip], [aldo, carla, david], [evi, gary, hugo, ida]),
             ([beat, flip], [aldo, carla, evi], [david, gary, hugo, ida]),
             ([beat, flip], [aldo, carla, gary], [david, evi, hugo, ida]),
             ([beat, flip], [aldo, carla, hugo], [david, evi, gary, ida]),
             ([beat, flip], [aldo, carla, ida], [david, evi, gary, hugo]),
             ([beat, flip], [aldo, david, evi], [carla, gary, hugo, ida]),
             ([beat, flip], [aldo, david, gary], [carla, evi, hugo, ida]),
             ([beat, flip], [aldo, david, hugo], [carla, evi, gary, ida]),
             ([beat, flip], [aldo, david, ida], [carla, evi, gary, hugo]),
             ([beat, flip], [aldo, evi, gary], [carla, david, hugo, ida]),
             ([beat, flip], [aldo, evi, hugo], [carla, david, gary, ida]),
             ([beat, flip], [aldo, evi, ida], [carla, david, gary, hugo]),
             ([beat, flip], [aldo, gary, hugo], [carla, david, evi, ida]),
             ([beat, flip], [aldo, gary, ida], [carla, david, evi, hugo]),
             ([beat, flip], [aldo, hugo, ida], [carla, david, evi, gary]),
             ([beat, flip], [carla, david, evi], [aldo, gary, hugo, ida]),
             ([beat, flip], [carla, david, gary], [aldo, evi, hugo, ida]),
             ([beat, flip], [carla, david, hugo], [aldo, evi, gary, ida]),
             ([beat, flip], [carla, david, ida], [aldo, evi, gary, hugo]),
             ([beat, flip], [carla, evi, gary], [aldo, david, hugo, ida]),
             ([beat, flip], [carla, evi, hugo], [aldo, david, gary, ida]),
             ([beat, flip], [carla, evi, ida], [aldo, david, gary, hugo]),
             ([beat, flip], [carla, gary, hugo], [aldo, david, evi, ida]),
             ([beat, flip], [carla, gary, ida], [aldo, david, evi, hugo]),
             ([beat, flip], [carla, hugo, ida], [aldo, david, evi, gary]),
             ([beat, flip], [david, evi, gary], [aldo, carla, hugo, ida]),
             ([beat, flip], [david, evi, hugo], [aldo, carla, gary, ida]),
             ([beat, flip], [david, evi, ida], [aldo, carla, gary, hugo]),
             ([beat, flip], [david, gary, hugo], [aldo, carla, evi, ida]),
             ([beat, flip], [david, gary, ida], [aldo, carla, evi, hugo]),
             ([beat, flip], [david, hugo, ida], [aldo, carla, evi, gary]),
             ([beat, flip], [evi, gary, hugo], [aldo, carla, david, ida]),
             ([beat, flip], [evi, gary, ida], [aldo, carla, david, hugo]),
             ([beat, flip], [evi, hugo, ida], [aldo, carla, david, gary]),
             ([beat, flip], [gary, hugo, ida], [aldo, carla, david, evi]),
             ([beat, gary], [aldo, carla, david], [evi, flip, hugo, ida]),
             ([beat, gary], [aldo, carla, evi], [david, flip, hugo, ida]),
             ([beat, gary], [aldo, carla, flip], [david, evi, hugo, ida]),
             ([beat, gary], [aldo, carla, hugo], [david, evi, flip, ida]),
             ([beat, gary], [aldo, carla, ida], [david, evi, flip, hugo]),
             ([beat, gary], [aldo, david, evi], [carla, flip, hugo, ida]),
             ([beat, gary], [aldo, david, flip], [carla, evi, hugo, ida]),
             ([beat, gary], [aldo, david, hugo], [carla, evi, flip, ida]),
             ([beat, gary], [aldo, david, ida], [carla, evi, flip, hugo]),
             ([beat, gary], [aldo, evi, flip], [carla, david, hugo, ida]),
             ([beat, gary], [aldo, evi, hugo], [carla, david, flip, ida]),
             ([beat, gary], [aldo, evi, ida], [carla, david, flip, hugo]),
             ([beat, gary], [aldo, flip, hugo], [carla, david, evi, ida]),
             ([beat, gary], [aldo, flip, ida], [carla, david, evi, hugo]),
             ([beat, gary], [aldo, hugo, ida], [carla, david, evi, flip]),
             ([beat, gary], [carla, david, evi], [aldo, flip, hugo, ida]),
             ([beat, gary], [carla, david, flip], [aldo, evi, hugo, ida]),
             ([beat, gary], [carla, david, hugo], [aldo, evi, flip, ida]),
             ([beat, gary], [carla, david, ida], [aldo, evi, flip, hugo]),
             ([beat, gary], [carla, evi, flip], [aldo, david, hugo, ida]),
             ([beat, gary], [carla, evi, hugo], [aldo, david, flip, ida]),
             ([beat, gary], [carla, evi, ida], [aldo, david, flip, hugo]),
             ([beat, gary], [carla, flip, hugo], [aldo, david, evi, ida]),
             ([beat, gary], [carla, flip, ida], [aldo, david, evi, hugo]),
             ([beat, gary], [carla, hugo, ida], [aldo, david, evi, flip]),
             ([beat, gary], [david, evi, flip], [aldo, carla, hugo, ida]),
             ([beat, gary], [david, evi, hugo], [aldo, carla, flip, ida]),
             ([beat, gary], [david, evi, ida], [aldo, carla, flip, hugo]),
             ([beat, gary], [david, flip, hugo], [aldo, carla, evi, ida]),
             ([beat, gary], [david, flip, ida], [aldo, carla, evi, hugo]),
             ([beat, gary], [david, hugo, ida], [aldo, carla, evi, flip]),
             ([beat, gary], [evi, flip, hugo], [aldo, carla, david, ida]),
             ([beat, gary], [evi, flip, ida], [aldo, carla, david, hugo]),
             ([beat, gary], [evi, hugo, ida], [aldo, carla, david, flip]),
             ([beat, gary], [flip, hugo, ida], [aldo, carla, david, evi]),
             ([beat, hugo], [aldo, carla, david], [evi, flip, gary, ida]),
             ([beat, hugo], [aldo, carla, evi], [david, flip, gary, ida]),
             ([beat, hugo], [aldo, carla, flip], [david, evi, gary, ida]),
             ([beat, hugo], [aldo, carla, gary], [david, evi, flip, ida]),
             ([beat, hugo], [aldo, carla, ida], [david, evi, flip, gary]),
             ([beat, hugo], [aldo, david, evi], [carla, flip, gary, ida]),
             ([beat, hugo], [aldo, david, flip], [carla, evi, gary, ida]),
             ([beat, hugo], [aldo, david, gary], [carla, evi, flip, ida]),
             ([beat, hugo], [aldo, david, ida], [carla, evi, flip, gary]),
             ([beat, hugo], [aldo, evi, flip], [carla, david, gary, ida]),
             ([beat, hugo], [aldo, evi, gary], [carla, david, flip, ida]),
             ([beat, hugo], [aldo, evi, ida], [carla, david, flip, gary]),
             ([beat, hugo], [aldo, flip, gary], [carla, david, evi, ida]),
             ([beat, hugo], [aldo, flip, ida], [carla, david, evi, gary]),
             ([beat, hugo], [aldo, gary, ida], [carla, david, evi, flip]),
             ([beat, hugo], [carla, david, evi], [aldo, flip, gary, ida]),
             ([beat, hugo], [carla, david, flip], [aldo, evi, gary, ida]),
             ([beat, hugo], [carla, david, gary], [aldo, evi, flip, ida]),
             ([beat, hugo], [carla, david, ida], [aldo, evi, flip, gary]),
             ([beat, hugo], [carla, evi, flip], [aldo, david, gary, ida]),
             ([beat, hugo], [carla, evi, gary], [aldo, david, flip, ida]),
             ([beat, hugo], [carla, evi, ida], [aldo, david, flip, gary]),
             ([beat, hugo], [carla, flip, gary], [aldo, david, evi, ida]),
             ([beat, hugo], [carla, flip, ida], [aldo, david, evi, gary]),
             ([beat, hugo], [carla, gary, ida], [aldo, david, evi, flip]),
             ([beat, hugo], [david, evi, flip], [aldo, carla, gary, ida]),
             ([beat, hugo], [david, evi, gary], [aldo, carla, flip, ida]),
             ([beat, hugo], [david, evi, ida], [aldo, carla, flip, gary]),
             ([beat, hugo], [david, flip, gary], [aldo, carla, evi, ida]),
             ([beat, hugo], [david, flip, ida], [aldo, carla, evi, gary]),
             ([beat, hugo], [david, gary, ida], [aldo, carla, evi, flip]),
             ([beat, hugo], [evi, flip, gary], [aldo, carla, david, ida]),
             ([beat, hugo], [evi, flip, ida], [aldo, carla, david, gary]),
             ([beat, hugo], [evi, gary, ida], [aldo, carla, david, flip]),
             ([beat, hugo], [flip, gary, ida], [aldo, carla, david, evi]),
             ([beat, ida], [aldo, carla, david], [evi, flip, gary, hugo]),
             ([beat, ida], [aldo, carla, evi], [david, flip, gary, hugo]),
             ([beat, ida], [aldo, carla, flip], [david, evi, gary, hugo]),
             ([beat, ida], [aldo, carla, gary], [david, evi, flip, hugo]),
             ([beat, ida], [aldo, carla, hugo], [david, evi, flip, gary]),
             ([beat, ida], [aldo, david, evi], [carla, flip, gary, hugo]),
             ([beat, ida], [aldo, david, flip], [carla, evi, gary, hugo]),
             ([beat, ida], [aldo, david, gary], [carla, evi, flip, hugo]),
             ([beat, ida], [aldo, david, hugo], [carla, evi, flip, gary]),
             ([beat, ida], [aldo, evi, flip], [carla, david, gary, hugo]),
             ([beat, ida], [aldo, evi, gary], [carla, david, flip, hugo]),
             ([beat, ida], [aldo, evi, hugo], [carla, david, flip, gary]),
             ([beat, ida], [aldo, flip, gary], [carla, david, evi, hugo]),
             ([beat, ida], [aldo, flip, hugo], [carla, david, evi, gary]),
             ([beat, ida], [aldo, gary, hugo], [carla, david, evi, flip]),
             ([beat, ida], [carla, david, evi], [aldo, flip, gary, hugo]),
             ([beat, ida], [carla, david, flip], [aldo, evi, gary, hugo]),
             ([beat, ida], [carla, david, gary], [aldo, evi, flip, hugo]),
             ([beat, ida], [carla, david, hugo], [aldo, evi, flip, gary]),
             ([beat, ida], [carla, evi, flip], [aldo, david, gary, hugo]),
             ([beat, ida], [carla, evi, gary], [aldo, david, flip, hugo]),
             ([beat, ida], [carla, evi, hugo], [aldo, david, flip, gary]),
             ([beat, ida], [carla, flip, gary], [aldo, david, evi, hugo]),
             ([beat, ida], [carla, flip, hugo], [aldo, david, evi, gary]),
             ([beat, ida], [carla, gary, hugo], [aldo, david, evi, flip]),
             ([beat, ida], [david, evi, flip], [aldo, carla, gary, hugo]),
             ([beat, ida], [david, evi, gary], [aldo, carla, flip, hugo]),
             ([beat, ida], [david, evi, hugo], [aldo, carla, flip, gary]),
             ([beat, ida], [david, flip, gary], [aldo, carla, evi, hugo]),
             ([beat, ida], [david, flip, hugo], [aldo, carla, evi, gary]),
             ([beat, ida], [david, gary, hugo], [aldo, carla, evi, flip]),
             ([beat, ida], [evi, flip, gary], [aldo, carla, david, hugo]),
             ([beat, ida], [evi, flip, hugo], [aldo, carla, david, gary]),
             ([beat, ida], [evi, gary, hugo], [aldo, carla, david, flip]),
             ([beat, ida], [flip, gary, hugo], [aldo, carla, david, evi]),
             ([carla, david], [aldo, beat, evi], [flip, gary, hugo, ida]),
             ([carla, david], [aldo, beat, flip], [evi, gary, hugo, ida]),
             ([carla, david], [aldo, beat, gary], [evi, flip, hugo, ida]),
             ([carla, david], [aldo, beat, hugo], [evi, flip, gary, ida]),
             ([carla, david], [aldo, beat, ida], [evi, flip, gary, hugo]),
             ([carla, david], [aldo, evi, flip], [beat, gary, hugo, ida]),
             ([carla, david], [aldo, evi, gary], [beat, flip, hugo, ida]),
             ([carla, david], [aldo, evi, hugo], [beat, flip, gary, ida]),
             ([carla, david], [aldo, evi, ida], [beat, flip, gary, hugo]),
             ([carla, david], [aldo, flip, gary], [beat, evi, hugo, ida]),
             ([carla, david], [aldo, flip, hugo], [beat, evi, gary, ida]),
             ([carla, david], [aldo, flip, ida], [beat, evi, gary, hugo]),
             ([carla, david], [aldo, gary, hugo], [beat, evi, flip, ida]),
             ([carla, david], [aldo, gary, ida], [beat, evi, flip, hugo]),
             ([carla, david], [aldo, hugo, ida], [beat, evi, flip, gary]),
             ([carla, david], [beat, evi, flip], [aldo, gary, hugo, ida]),
             ([carla, david], [beat, evi, gary], [aldo, flip, hugo, ida]),
             ([carla, david], [beat, evi, hugo], [aldo, flip, gary, ida]),
             ([carla, david], [beat, evi, ida], [aldo, flip, gary, hugo]),
             ([carla, david], [beat, flip, gary], [aldo, evi, hugo, ida]),
             ([carla, david], [beat, flip, hugo], [aldo, evi, gary, ida]),
             ([carla, david], [beat, flip, ida], [aldo, evi, gary, hugo]),
             ([carla, david], [beat, gary, hugo], [aldo, evi, flip, ida]),
             ([carla, david], [beat, gary, ida], [aldo, evi, flip, hugo]),
             ([carla, david], [beat, hugo, ida], [aldo, evi, flip, gary]),
             ([carla, david], [evi, flip, gary], [aldo, beat, hugo, ida]),
             ([carla, david], [evi, flip, hugo], [aldo, beat, gary, ida]),
             ([carla, david], [evi, flip, ida], [aldo, beat, gary, hugo]),
             ([carla, david], [evi, gary, hugo], [aldo, beat, flip, ida]),
             ([carla, david], [evi, gary, ida], [aldo, beat, flip, hugo]),
             ([carla, david], [evi, hugo, ida], [aldo, beat, flip, gary]),
             ([carla, david], [flip, gary, hugo], [aldo, beat, evi, ida]),
             ([carla, david], [flip, gary, ida], [aldo, beat, evi, hugo]),
             ([carla, david], [flip, hugo, ida], [aldo, beat, evi, gary]),
             ([carla, david], [gary, hugo, ida], [aldo, beat, evi, flip]),
             ([carla, evi], [aldo, beat, david], [flip, gary, hugo, ida]),
             ([carla, evi], [aldo, beat, flip], [david, gary, hugo, ida]),
             ([carla, evi], [aldo, beat, gary], [david, flip, hugo, ida]),
             ([carla, evi], [aldo, beat, hugo], [david, flip, gary, ida]),
             ([carla, evi], [aldo, beat, ida], [david, flip, gary, hugo]),
             ([carla, evi], [aldo, david, flip], [beat, gary, hugo, ida]),
             ([carla, evi], [aldo, david, gary], [beat, flip, hugo, ida]),
             ([carla, evi], [aldo, david, hugo], [beat, flip, gary, ida]),
             ([carla, evi], [aldo, david, ida], [beat, flip, gary, hugo]),
             ([carla, evi], [aldo, flip, gary], [beat, david, hugo, ida]),
             ([carla, evi], [aldo, flip, hugo], [beat, david, gary, ida]),
             ([carla, evi], [aldo, flip, ida], [beat, david, gary, hugo]),
             ([carla, evi], [aldo, gary, hugo], [beat, david, flip, ida]),
             ([carla, evi], [aldo, gary, ida], [beat, david, flip, hugo]),
             ([carla, evi], [aldo, hugo, ida], [beat, david, flip, gary]),
             ([carla, evi], [beat, david, flip], [aldo, gary, hugo, ida]),
             ([carla, evi], [beat, david, gary], [aldo, flip, hugo, ida]),
             ([carla, evi], [beat, david, hugo], [aldo, flip, gary, ida]),
             ([carla, evi], [beat, david, ida], [aldo, flip, gary, hugo]),
             ([carla, evi], [beat, flip, gary], [aldo, david, hugo, ida]),
             ([carla, evi], [beat, flip, hugo], [aldo, david, gary, ida]),
             ([carla, evi], [beat, flip, ida], [aldo, david, gary, hugo]),
             ([carla, evi], [beat, gary, hugo], [aldo, david, flip, ida]),
             ([carla, evi], [beat, gary, ida], [aldo, david, flip, hugo]),
             ([carla, evi], [beat, hugo, ida], [aldo, david, flip, gary]),
             ([carla, evi], [david, flip, gary], [aldo, beat, hugo, ida]),
             ([carla, evi], [david, flip, hugo], [aldo, beat, gary, ida]),
             ([carla, evi], [david, flip, ida], [aldo, beat, gary, hugo]),
             ([carla, evi], [david, gary, hugo], [aldo, beat, flip, ida]),
             ([carla, evi], [david, gary, ida], [aldo, beat, flip, hugo]),
             ([carla, evi], [david, hugo, ida], [aldo, beat, flip, gary]),
             ([carla, evi], [flip, gary, hugo], [aldo, beat, david, ida]),
             ([carla, evi], [flip, gary, ida], [aldo, beat, david, hugo]),
             ([carla, evi], [flip, hugo, ida], [aldo, beat, david, gary]),
             ([carla, evi], [gary, hugo, ida], [aldo, beat, david, flip]),
             ([carla, flip], [aldo, beat, david], [evi, gary, hugo, ida]),
             ([carla, flip], [aldo, beat, evi], [david, gary, hugo, ida]),
             ([carla, flip], [aldo, beat, gary], [david, evi, hugo, ida]),
             ([carla, flip], [aldo, beat, hugo], [david, evi, gary, ida]),
             ([carla, flip], [aldo, beat, ida], [david, evi, gary, hugo]),
             ([carla, flip], [aldo, david, evi], [beat, gary, hugo, ida]),
             ([carla, flip], [aldo, david, gary], [beat, evi, hugo, ida]),
             ([carla, flip], [aldo, david, hugo], [beat, evi, gary, ida]),
             ([carla, flip], [aldo, david, ida], [beat, evi, gary, hugo]),
             ([carla, flip], [aldo, evi, gary], [beat, david, hugo, ida]),
             ([carla, flip], [aldo, evi, hugo], [beat, david, gary, ida]),
             ([carla, flip], [aldo, evi, ida], [beat, david, gary, hugo]),
             ([carla, flip], [aldo, gary, hugo], [beat, david, evi, ida]),
             ([carla, flip], [aldo, gary, ida], [beat, david, evi, hugo]),
             ([carla, flip], [aldo, hugo, ida], [beat, david, evi, gary]),
             ([carla, flip], [beat, david, evi], [aldo, gary, hugo, ida]),
             ([carla, flip], [beat, david, gary], [aldo, evi, hugo, ida]),
             ([carla, flip], [beat, david, hugo], [aldo, evi, gary, ida]),
             ([carla, flip], [beat, david, ida], [aldo, evi, gary, hugo]),
             ([carla, flip], [beat, evi, gary], [aldo, david, hugo, ida]),
             ([carla, flip], [beat, evi, hugo], [aldo, david, gary, ida]),
             ([carla, flip], [beat, evi, ida], [aldo, david, gary, hugo]),
             ([carla, flip], [beat, gary, hugo], [aldo, david, evi, ida]),
             ([carla, flip], [beat, gary, ida], [aldo, david, evi, hugo]),
             ([carla, flip], [beat, hugo, ida], [aldo, david, evi, gary]),
             ([carla, flip], [david, evi, gary], [aldo, beat, hugo, ida]),
             ([carla, flip], [david, evi, hugo], [aldo, beat, gary, ida]),
             ([carla, flip], [david, evi, ida], [aldo, beat, gary, hugo]),
             ([carla, flip], [david, gary, hugo], [aldo, beat, evi, ida]),
             ([carla, flip], [david, gary, ida], [aldo, beat, evi, hugo]),
             ([carla, flip], [david, hugo, ida], [aldo, beat, evi, gary]),
             ([carla, flip], [evi, gary, hugo], [aldo, beat, david, ida]),
             ([carla, flip], [evi, gary, ida], [aldo, beat, david, hugo]),
             ([carla, flip], [evi, hugo, ida], [aldo, beat, david, gary]),
             ([carla, flip], [gary, hugo, ida], [aldo, beat, david, evi]),
             ([carla, gary], [aldo, beat, david], [evi, flip, hugo, ida]),
             ([carla, gary], [aldo, beat, evi], [david, flip, hugo, ida]),
             ([carla, gary], [aldo, beat, flip], [david, evi, hugo, ida]),
             ([carla, gary], [aldo, beat, hugo], [david, evi, flip, ida]),
             ([carla, gary], [aldo, beat, ida], [david, evi, flip, hugo]),
             ([carla, gary], [aldo, david, evi], [beat, flip, hugo, ida]),
             ([carla, gary], [aldo, david, flip], [beat, evi, hugo, ida]),
             ([carla, gary], [aldo, david, hugo], [beat, evi, flip, ida]),
             ([carla, gary], [aldo, david, ida], [beat, evi, flip, hugo]),
             ([carla, gary], [aldo, evi, flip], [beat, david, hugo, ida]),
             ([carla, gary], [aldo, evi, hugo], [beat, david, flip, ida]),
             ([carla, gary], [aldo, evi, ida], [beat, david, flip, hugo]),
             ([carla, gary], [aldo, flip, hugo], [beat, david, evi, ida]),
             ([carla, gary], [aldo, flip, ida], [beat, david, evi, hugo]),
             ([carla, gary], [aldo, hugo, ida], [beat, david, evi, flip]),
             ([carla, gary], [beat, david, evi], [aldo, flip, hugo, ida]),
             ([carla, gary], [beat, david, flip], [aldo, evi, hugo, ida]),
             ([carla, gary], [beat, david, hugo], [aldo, evi, flip, ida]),
             ([carla, gary], [beat, david, ida], [aldo, evi, flip, hugo]),
             ([carla, gary], [beat, evi, flip], [aldo, david, hugo, ida]),
             ([carla, gary], [beat, evi, hugo], [aldo, david, flip, ida]),
             ([carla, gary], [beat, evi, ida], [aldo, david, flip, hugo]),
             ([carla, gary], [beat, flip, hugo], [aldo, david, evi, ida]),
             ([carla, gary], [beat, flip, ida], [aldo, david, evi, hugo]),
             ([carla, gary], [beat, hugo, ida], [aldo, david, evi, flip]),
             ([carla, gary], [david, evi, flip], [aldo, beat, hugo, ida]),
             ([carla, gary], [david, evi, hugo], [aldo, beat, flip, ida]),
             ([carla, gary], [david, evi, ida], [aldo, beat, flip, hugo]),
             ([carla, gary], [david, flip, hugo], [aldo, beat, evi, ida]),
             ([carla, gary], [david, flip, ida], [aldo, beat, evi, hugo]),
             ([carla, gary], [david, hugo, ida], [aldo, beat, evi, flip]),
             ([carla, gary], [evi, flip, hugo], [aldo, beat, david, ida]),
             ([carla, gary], [evi, flip, ida], [aldo, beat, david, hugo]),
             ([carla, gary], [evi, hugo, ida], [aldo, beat, david, flip]),
             ([carla, gary], [flip, hugo, ida], [aldo, beat, david, evi]),
             ([carla, hugo], [aldo, beat, david], [evi, flip, gary, ida]),
             ([carla, hugo], [aldo, beat, evi], [david, flip, gary, ida]),
             ([carla, hugo], [aldo, beat, flip], [david, evi, gary, ida]),
             ([carla, hugo], [aldo, beat, gary], [david, evi, flip, ida]),
             ([carla, hugo], [aldo, beat, ida], [david, evi, flip, gary]),
             ([carla, hugo], [aldo, david, evi], [beat, flip, gary, ida]),
             ([carla, hugo], [aldo, david, flip], [beat, evi, gary, ida]),
             ([carla, hugo], [aldo, david, gary], [beat, evi, flip, ida]),
             ([carla, hugo], [aldo, david, ida], [beat, evi, flip, gary]),
             ([carla, hugo], [aldo, evi, flip], [beat, david, gary, ida]),
             ([carla, hugo], [aldo, evi, gary], [beat, david, flip, ida]),
             ([carla, hugo], [aldo, evi, ida], [beat, david, flip, gary]),
             ([carla, hugo], [aldo, flip, gary], [beat, david, evi, ida]),
             ([carla, hugo], [aldo, flip, ida], [beat, david, evi, gary]),
             ([carla, hugo], [aldo, gary, ida], [beat, david, evi, flip]),
             ([carla, hugo], [beat, david, evi], [aldo, flip, gary, ida]),
             ([carla, hugo], [beat, david, flip], [aldo, evi, gary, ida]),
             ([carla, hugo], [beat, david, gary], [aldo, evi, flip, ida]),
             ([carla, hugo], [beat, david, ida], [aldo, evi, flip, gary]),
             ([carla, hugo], [beat, evi, flip], [aldo, david, gary, ida]),
             ([carla, hugo], [beat, evi, gary], [aldo, david, flip, ida]),
             ([carla, hugo], [beat, evi, ida], [aldo, david, flip, gary]),
             ([carla, hugo], [beat, flip, gary], [aldo, david, evi, ida]),
             ([carla, hugo], [beat, flip, ida], [aldo, david, evi, gary]),
             ([carla, hugo], [beat, gary, ida], [aldo, david, evi, flip]),
             ([carla, hugo], [david, evi, flip], [aldo, beat, gary, ida]),
             ([carla, hugo], [david, evi, gary], [aldo, beat, flip, ida]),
             ([carla, hugo], [david, evi, ida], [aldo, beat, flip, gary]),
             ([carla, hugo], [david, flip, gary], [aldo, beat, evi, ida]),
             ([carla, hugo], [david, flip, ida], [aldo, beat, evi, gary]),
             ([carla, hugo], [david, gary, ida], [aldo, beat, evi, flip]),
             ([carla, hugo], [evi, flip, gary], [aldo, beat, david, ida]),
             ([carla, hugo], [evi, flip, ida], [aldo, beat, david, gary]),
             ([carla, hugo], [evi, gary, ida], [aldo, beat, david, flip]),
             ([carla, hugo], [flip, gary, ida], [aldo, beat, david, evi]),
             ([carla, ida], [aldo, beat, david], [evi, flip, gary, hugo]),
             ([carla, ida], [aldo, beat, evi], [david, flip, gary, hugo]),
             ([carla, ida], [aldo, beat, flip], [david, evi, gary, hugo]),
             ([carla, ida], [aldo, beat, gary], [david, evi, flip, hugo]),
             ([carla, ida], [aldo, beat, hugo], [david, evi, flip, gary]),
             ([carla, ida], [aldo, david, evi], [beat, flip, gary, hugo]),
             ([carla, ida], [aldo, david, flip], [beat, evi, gary, hugo]),
             ([carla, ida], [aldo, david, gary], [beat, evi, flip, hugo]),
             ([carla, ida], [aldo, david, hugo], [beat, evi, flip, gary]),
             ([carla, ida], [aldo, evi, flip], [beat, david, gary, hugo]),
             ([carla, ida], [aldo, evi, gary], [beat, david, flip, hugo]),
             ([carla, ida], [aldo, evi, hugo], [beat, david, flip, gary]),
             ([carla, ida], [aldo, flip, gary], [beat, david, evi, hugo]),
             ([carla, ida], [aldo, flip, hugo], [beat, david, evi, gary]),
             ([carla, ida], [aldo, gary, hugo], [beat, david, evi, flip]),
             ([carla, ida], [beat, david, evi], [aldo, flip, gary, hugo]),
             ([carla, ida], [beat, david, flip], [aldo, evi, gary, hugo]),
             ([carla, ida], [beat, david, gary], [aldo, evi, flip, hugo]),
             ([carla, ida], [beat, david, hugo], [aldo, evi, flip, gary]),
             ([carla, ida], [beat, evi, flip], [aldo, david, gary, hugo]),
             ([carla, ida], [beat, evi, gary], [aldo, david, flip, hugo]),
             ([carla, ida], [beat, evi, hugo], [aldo, david, flip, gary]),
             ([carla, ida], [beat, flip, gary], [aldo, david, evi, hugo]),
             ([carla, ida], [beat, flip, hugo], [aldo, david, evi, gary]),
             ([carla, ida], [beat, gary, hugo], [aldo, david, evi, flip]),
             ([carla, ida], [david, evi, flip], [aldo, beat, gary, hugo]),
             ([carla, ida], [david, evi, gary], [aldo, beat, flip, hugo]),
             ([carla, ida], [david, evi, hugo], [aldo, beat, flip, gary]),
             ([carla, ida], [david, flip, gary], [aldo, beat, evi, hugo]),
             ([carla, ida], [david, flip, hugo], [aldo, beat, evi, gary]),
             ([carla, ida], [david, gary, hugo], [aldo, beat, evi, flip]),
             ([carla, ida], [evi, flip, gary], [aldo, beat, david, hugo]),
             ([carla, ida], [evi, flip, hugo], [aldo, beat, david, gary]),
             ([carla, ida], [evi, gary, hugo], [aldo, beat, david, flip]),
             ([carla, ida], [flip, gary, hugo], [aldo, beat, david, evi]),
             ([david, evi], [aldo, beat, carla], [flip, gary, hugo, ida]),
             ([david, evi], [aldo, beat, flip], [carla, gary, hugo, ida]),
             ([david, evi], [aldo, beat, gary], [carla, flip, hugo, ida]),
             ([david, evi], [aldo, beat, hugo], [carla, flip, gary, ida]),
             ([david, evi], [aldo, beat, ida], [carla, flip, gary, hugo]),
             ([david, evi], [aldo, carla, flip], [beat, gary, hugo, ida]),
             ([david, evi], [aldo, carla, gary], [beat, flip, hugo, ida]),
             ([david, evi], [aldo, carla, hugo], [beat, flip, gary, ida]),
             ([david, evi], [aldo, carla, ida], [beat, flip, gary, hugo]),
             ([david, evi], [aldo, flip, gary], [beat, carla, hugo, ida]),
             ([david, evi], [aldo, flip, hugo], [beat, carla, gary, ida]),
             ([david, evi], [aldo, flip, ida], [beat, carla, gary, hugo]),
             ([david, evi], [aldo, gary, hugo], [beat, carla, flip, ida]),
             ([david, evi], [aldo, gary, ida], [beat, carla, flip, hugo]),
             ([david, evi], [aldo, hugo, ida], [beat, carla, flip, gary]),
             ([david, evi], [beat, carla, flip], [aldo, gary, hugo, ida]),
             ([david, evi], [beat, carla, gary], [aldo, flip, hugo, ida]),
             ([david, evi], [beat, carla, hugo], [aldo, flip, gary, ida]),
             ([david, evi], [beat, carla, ida], [aldo, flip, gary, hugo]),
             ([david, evi], [beat, flip, gary], [aldo, carla, hugo, ida]),
             ([david, evi], [beat, flip, hugo], [aldo, carla, gary, ida]),
             ([david, evi], [beat, flip, ida], [aldo, carla, gary, hugo]),
             ([david, evi], [beat, gary, hugo], [aldo, carla, flip, ida]),
             ([david, evi], [beat, gary, ida], [aldo, carla, flip, hugo]),
             ([david, evi], [beat, hugo, ida], [aldo, carla, flip, gary]),
             ([david, evi], [carla, flip, gary], [aldo, beat, hugo, ida]),
             ([david, evi], [carla, flip, hugo], [aldo, beat, gary, ida]),
             ([david, evi], [carla, flip, ida], [aldo, beat, gary, hugo]),
             ([david, evi], [carla, gary, hugo], [aldo, beat, flip, ida]),
             ([david, evi], [carla, gary, ida], [aldo, beat, flip, hugo]),
             ([david, evi], [carla, hugo, ida], [aldo, beat, flip, gary]),
             ([david, evi], [flip, gary, hugo], [aldo, beat, carla, ida]),
             ([david, evi], [flip, gary, ida], [aldo, beat, carla, hugo]),
             ([david, evi], [flip, hugo, ida], [aldo, beat, carla, gary]),
             ([david, evi], [gary, hugo, ida], [aldo, beat, carla, flip]),
             ([david, flip], [aldo, beat, carla], [evi, gary, hugo, ida]),
             ([david, flip], [aldo, beat, evi], [carla, gary, hugo, ida]),
             ([david, flip], [aldo, beat, gary], [carla, evi, hugo, ida]),
             ([david, flip], [aldo, beat, hugo], [carla, evi, gary, ida]),
             ([david, flip], [aldo, beat, ida], [carla, evi, gary, hugo]),
             ([david, flip], [aldo, carla, evi], [beat, gary, hugo, ida]),
             ([david, flip], [aldo, carla, gary], [beat, evi, hugo, ida]),
             ([david, flip], [aldo, carla, hugo], [beat, evi, gary, ida]),
             ([david, flip], [aldo, carla, ida], [beat, evi, gary, hugo]),
             ([david, flip], [aldo, evi, gary], [beat, carla, hugo, ida]),
             ([david, flip], [aldo, evi, hugo], [beat, carla, gary, ida]),
             ([david, flip], [aldo, evi, ida], [beat, carla, gary, hugo]),
             ([david, flip], [aldo, gary, hugo], [beat, carla, evi, ida]),
             ([david, flip], [aldo, gary, ida], [beat, carla, evi, hugo]),
             ([david, flip], [aldo, hugo, ida], [beat, carla, evi, gary]),
             ([david, flip], [beat, carla, evi], [aldo, gary, hugo, ida]),
             ([david, flip], [beat, carla, gary], [aldo, evi, hugo, ida]),
             ([david, flip], [beat, carla, hugo], [aldo, evi, gary, ida]),
             ([david, flip], [beat, carla, ida], [aldo, evi, gary, hugo]),
             ([david, flip], [beat, evi, gary], [aldo, carla, hugo, ida]),
             ([david, flip], [beat, evi, hugo], [aldo, carla, gary, ida]),
             ([david, flip], [beat, evi, ida], [aldo, carla, gary, hugo]),
             ([david, flip], [beat, gary, hugo], [aldo, carla, evi, ida]),
             ([david, flip], [beat, gary, ida], [aldo, carla, evi, hugo]),
             ([david, flip], [beat, hugo, ida], [aldo, carla, evi, gary]),
             ([david, flip], [carla, evi, gary], [aldo, beat, hugo, ida]),
             ([david, flip], [carla, evi, hugo], [aldo, beat, gary, ida]),
             ([david, flip], [carla, evi, ida], [aldo, beat, gary, hugo]),
             ([david, flip], [carla, gary, hugo], [aldo, beat, evi, ida]),
             ([david, flip], [carla, gary, ida], [aldo, beat, evi, hugo]),
             ([david, flip], [carla, hugo, ida], [aldo, beat, evi, gary]),
             ([david, flip], [evi, gary, hugo], [aldo, beat, carla, ida]),
             ([david, flip], [evi, gary, ida], [aldo, beat, carla, hugo]),
             ([david, flip], [evi, hugo, ida], [aldo, beat, carla, gary]),
             ([david, flip], [gary, hugo, ida], [aldo, beat, carla, evi]),
             ([david, gary], [aldo, beat, carla], [evi, flip, hugo, ida]),
             ([david, gary], [aldo, beat, evi], [carla, flip, hugo, ida]),
             ([david, gary], [aldo, beat, flip], [carla, evi, hugo, ida]),
             ([david, gary], [aldo, beat, hugo], [carla, evi, flip, ida]),
             ([david, gary], [aldo, beat, ida], [carla, evi, flip, hugo]),
             ([david, gary], [aldo, carla, evi], [beat, flip, hugo, ida]),
             ([david, gary], [aldo, carla, flip], [beat, evi, hugo, ida]),
             ([david, gary], [aldo, carla, hugo], [beat, evi, flip, ida]),
             ([david, gary], [aldo, carla, ida], [beat, evi, flip, hugo]),
             ([david, gary], [aldo, evi, flip], [beat, carla, hugo, ida]),
             ([david, gary], [aldo, evi, hugo], [beat, carla, flip, ida]),
             ([david, gary], [aldo, evi, ida], [beat, carla, flip, hugo]),
             ([david, gary], [aldo, flip, hugo], [beat, carla, evi, ida]),
             ([david, gary], [aldo, flip, ida], [beat, carla, evi, hugo]),
             ([david, gary], [aldo, hugo, ida], [beat, carla, evi, flip]),
             ([david, gary], [beat, carla, evi], [aldo, flip, hugo, ida]),
             ([david, gary], [beat, carla, flip], [aldo, evi, hugo, ida]),
             ([david, gary], [beat, carla, hugo], [aldo, evi, flip, ida]),
             ([david, gary], [beat, carla, ida], [aldo, evi, flip, hugo]),
             ([david, gary], [beat, evi, flip], [aldo, carla, hugo, ida]),
             ([david, gary], [beat, evi, hugo], [aldo, carla, flip, ida]),
             ([david, gary], [beat, evi, ida], [aldo, carla, flip, hugo]),
             ([david, gary], [beat, flip, hugo], [aldo, carla, evi, ida]),
             ([david, gary], [beat, flip, ida], [aldo, carla, evi, hugo]),
             ([david, gary], [beat, hugo, ida], [aldo, carla, evi, flip]),
             ([david, gary], [carla, evi, flip], [aldo, beat, hugo, ida]),
             ([david, gary], [carla, evi, hugo], [aldo, beat, flip, ida]),
             ([david, gary], [carla, evi, ida], [aldo, beat, flip, hugo]),
             ([david, gary], [carla, flip, hugo], [aldo, beat, evi, ida]),
             ([david, gary], [carla, flip, ida], [aldo, beat, evi, hugo]),
             ([david, gary], [carla, hugo, ida], [aldo, beat, evi, flip]),
             ([david, gary], [evi, flip, hugo], [aldo, beat, carla, ida]),
             ([david, gary], [evi, flip, ida], [aldo, beat, carla, hugo]),
             ([david, gary], [evi, hugo, ida], [aldo, beat, carla, flip]),
             ([david, gary], [flip, hugo, ida], [aldo, beat, carla, evi]),
             ([david, hugo], [aldo, beat, carla], [evi, flip, gary, ida]),
             ([david, hugo], [aldo, beat, evi], [carla, flip, gary, ida]),
             ([david, hugo], [aldo, beat, flip], [carla, evi, gary, ida]),
             ([david, hugo], [aldo, beat, gary], [carla, evi, flip, ida]),
             ([david, hugo], [aldo, beat, ida], [carla, evi, flip, gary]),
             ([david, hugo], [aldo, carla, evi], [beat, flip, gary, ida]),
             ([david, hugo], [aldo, carla, flip], [beat, evi, gary, ida]),
             ([david, hugo], [aldo, carla, gary], [beat, evi, flip, ida]),
             ([david, hugo], [aldo, carla, ida], [beat, evi, flip, gary]),
             ([david, hugo], [aldo, evi, flip], [beat, carla, gary, ida]),
             ([david, hugo], [aldo, evi, gary], [beat, carla, flip, ida]),
             ([david, hugo], [aldo, evi, ida], [beat, carla, flip, gary]),
             ([david, hugo], [aldo, flip, gary], [beat, carla, evi, ida]),
             ([david, hugo], [aldo, flip, ida], [beat, carla, evi, gary]),
             ([david, hugo], [aldo, gary, ida], [beat, carla, evi, flip]),
             ([david, hugo], [beat, carla, evi], [aldo, flip, gary, ida]),
             ([david, hugo], [beat, carla, flip], [aldo, evi, gary, ida]),
             ([david, hugo], [beat, carla, gary], [aldo, evi, flip, ida]),
             ([david, hugo], [beat, carla, ida], [aldo, evi, flip, gary]),
             ([david, hugo], [beat, evi, flip], [aldo, carla, gary, ida]),
             ([david, hugo], [beat, evi, gary], [aldo, carla, flip, ida]),
             ([david, hugo], [beat, evi, ida], [aldo, carla, flip, gary]),
             ([david, hugo], [beat, flip, gary], [aldo, carla, evi, ida]),
             ([david, hugo], [beat, flip, ida], [aldo, carla, evi, gary]),
             ([david, hugo], [beat, gary, ida], [aldo, carla, evi, flip]),
             ([david, hugo], [carla, evi, flip], [aldo, beat, gary, ida]),
             ([david, hugo], [carla, evi, gary], [aldo, beat, flip, ida]),
             ([david, hugo], [carla, evi, ida], [aldo, beat, flip, gary]),
             ([david, hugo], [carla, flip, gary], [aldo, beat, evi, ida]),
             ([david, hugo], [carla, flip, ida], [aldo, beat, evi, gary]),
             ([david, hugo], [carla, gary, ida], [aldo, beat, evi, flip]),
             ([david, hugo], [evi, flip, gary], [aldo, beat, carla, ida]),
             ([david, hugo], [evi, flip, ida], [aldo, beat, carla, gary]),
             ([david, hugo], [evi, gary, ida], [aldo, beat, carla, flip]),
             ([david, hugo], [flip, gary, ida], [aldo, beat, carla, evi]),
             ([david, ida], [aldo, beat, carla], [evi, flip, gary, hugo]),
             ([david, ida], [aldo, beat, evi], [carla, flip, gary, hugo]),
             ([david, ida], [aldo, beat, flip], [carla, evi, gary, hugo]),
             ([david, ida], [aldo, beat, gary], [carla, evi, flip, hugo]),
             ([david, ida], [aldo, beat, hugo], [carla, evi, flip, gary]),
             ([david, ida], [aldo, carla, evi], [beat, flip, gary, hugo]),
             ([david, ida], [aldo, carla, flip], [beat, evi, gary, hugo]),
             ([david, ida], [aldo, carla, gary], [beat, evi, flip, hugo]),
             ([david, ida], [aldo, carla, hugo], [beat, evi, flip, gary]),
             ([david, ida], [aldo, evi, flip], [beat, carla, gary, hugo]),
             ([david, ida], [aldo, evi, gary], [beat, carla, flip, hugo]),
             ([david, ida], [aldo, evi, hugo], [beat, carla, flip, gary]),
             ([david, ida], [aldo, flip, gary], [beat, carla, evi, hugo]),
             ([david, ida], [aldo, flip, hugo], [beat, carla, evi, gary]),
             ([david, ida], [aldo, gary, hugo], [beat, carla, evi, flip]),
             ([david, ida], [beat, carla, evi], [aldo, flip, gary, hugo]),
             ([david, ida], [beat, carla, flip], [aldo, evi, gary, hugo]),
             ([david, ida], [beat, carla, gary], [aldo, evi, flip, hugo]),
             ([david, ida], [beat, carla, hugo], [aldo, evi, flip, gary]),
             ([david, ida], [beat, evi, flip], [aldo, carla, gary, hugo]),
             ([david, ida], [beat, evi, gary], [aldo, carla, flip, hugo]),
             ([david, ida], [beat, evi, hugo], [aldo, carla, flip, gary]),
             ([david, ida], [beat, flip, gary], [aldo, carla, evi, hugo]),
             ([david, ida], [beat, flip, hugo], [aldo, carla, evi, gary]),
             ([david, ida], [beat, gary, hugo], [aldo, carla, evi, flip]),
             ([david, ida], [carla, evi, flip], [aldo, beat, gary, hugo]),
             ([david, ida], [carla, evi, gary], [aldo, beat, flip, hugo]),
             ([david, ida], [carla, evi, hugo], [aldo, beat, flip, gary]),
             ([david, ida], [carla, flip, gary], [aldo, beat, evi, hugo]),
             ([david, ida], [carla, flip, hugo], [aldo, beat, evi, gary]),
             ([david, ida], [carla, gary, hugo], [aldo, beat, evi, flip]),
             ([david, ida], [evi, flip, gary], [aldo, beat, carla, hugo]),
             ([david, ida], [evi, flip, hugo], [aldo, beat, carla, gary]),
             ([david, ida], [evi, gary, hugo], [aldo, beat, carla, flip]),
             ([david, ida], [flip, gary, hugo], [aldo, beat, carla, evi]),
             ([evi, flip], [aldo, beat, carla], [david, gary, hugo, ida]),
             ([evi, flip], [aldo, beat, david], [carla, gary, hugo, ida]),
             ([evi, flip], [aldo, beat, gary], [carla, david, hugo, ida]),
             ([evi, flip], [aldo, beat, hugo], [carla, david, gary, ida]),
             ([evi, flip], [aldo, beat, ida], [carla, david, gary, hugo]),
             ([evi, flip], [aldo, carla, david], [beat, gary, hugo, ida]),
             ([evi, flip], [aldo, carla, gary], [beat, david, hugo, ida]),
             ([evi, flip], [aldo, carla, hugo], [beat, david, gary, ida]),
             ([evi, flip], [aldo, carla, ida], [beat, david, gary, hugo]),
             ([evi, flip], [aldo, david, gary], [beat, carla, hugo, ida]),
             ([evi, flip], [aldo, david, hugo], [beat, carla, gary, ida]),
             ([evi, flip], [aldo, david, ida], [beat, carla, gary, hugo]),
             ([evi, flip], [aldo, gary, hugo], [beat, carla, david, ida]),
             ([evi, flip], [aldo, gary, ida], [beat, carla, david, hugo]),
             ([evi, flip], [aldo, hugo, ida], [beat, carla, david, gary]),
             ([evi, flip], [beat, carla, david], [aldo, gary, hugo, ida]),
             ([evi, flip], [beat, carla, gary], [aldo, david, hugo, ida]),
             ([evi, flip], [beat, carla, hugo], [aldo, david, gary, ida]),
             ([evi, flip], [beat, carla, ida], [aldo, david, gary, hugo]),
             ([evi, flip], [beat, david, gary], [aldo, carla, hugo, ida]),
             ([evi, flip], [beat, david, hugo], [aldo, carla, gary, ida]),
             ([evi, flip], [beat, david, ida], [aldo, carla, gary, hugo]),
             ([evi, flip], [beat, gary, hugo], [aldo, carla, david, ida]),
             ([evi, flip], [beat, gary, ida], [aldo, carla, david, hugo]),
             ([evi, flip], [beat, hugo, ida], [aldo, carla, david, gary]),
             ([evi, flip], [carla, david, gary], [aldo, beat, hugo, ida]),
             ([evi, flip], [carla, david, hugo], [aldo, beat, gary, ida]),
             ([evi, flip], [carla, david, ida], [aldo, beat, gary, hugo]),
             ([evi, flip], [carla, gary, hugo], [aldo, beat, david, ida]),
             ([evi, flip], [carla, gary, ida], [aldo, beat, david, hugo]),
             ([evi, flip], [carla, hugo, ida], [aldo, beat, david, gary]),
             ([evi, flip], [david, gary, hugo], [aldo, beat, carla, ida]),
             ([evi, flip], [david, gary, ida], [aldo, beat, carla, hugo]),
             ([evi, flip], [david, hugo, ida], [aldo, beat, carla, gary]),
             ([evi, flip], [gary, hugo, ida], [aldo, beat, carla, david]),
             ([evi, gary], [aldo, beat, carla], [david, flip, hugo, ida]),
             ([evi, gary], [aldo, beat, david], [carla, flip, hugo, ida]),
             ([evi, gary], [aldo, beat, flip], [carla, david, hugo, ida]),
             ([evi, gary], [aldo, beat, hugo], [carla, david, flip, ida]),
             ([evi, gary], [aldo, beat, ida], [carla, david, flip, hugo]),
             ([evi, gary], [aldo, carla, david], [beat, flip, hugo, ida]),
             ([evi, gary], [aldo, carla, flip], [beat, david, hugo, ida]),
             ([evi, gary], [aldo, carla, hugo], [beat, david, flip, ida]),
             ([evi, gary], [aldo, carla, ida], [beat, david, flip, hugo]),
             ([evi, gary], [aldo, david, flip], [beat, carla, hugo, ida]),
             ([evi, gary], [aldo, david, hugo], [beat, carla, flip, ida]),
             ([evi, gary], [aldo, david, ida], [beat, carla, flip, hugo]),
             ([evi, gary], [aldo, flip, hugo], [beat, carla, david, ida]),
             ([evi, gary], [aldo, flip, ida], [beat, carla, david, hugo]),
             ([evi, gary], [aldo, hugo, ida], [beat, carla, david, flip]),
             ([evi, gary], [beat, carla, david], [aldo, flip, hugo, ida]),
             ([evi, gary], [beat, carla, flip], [aldo, david, hugo, ida]),
             ([evi, gary], [beat, carla, hugo], [aldo, david, flip, ida]),
             ([evi, gary], [beat, carla, ida], [aldo, david, flip, hugo]),
             ([evi, gary], [beat, david, flip], [aldo, carla, hugo, ida]),
             ([evi, gary], [beat, david, hugo], [aldo, carla, flip, ida]),
             ([evi, gary], [beat, david, ida], [aldo, carla, flip, hugo]),
             ([evi, gary], [beat, flip, hugo], [aldo, carla, david, ida]),
             ([evi, gary], [beat, flip, ida], [aldo, carla, david, hugo]),
             ([evi, gary], [beat, hugo, ida], [aldo, carla, david, flip]),
             ([evi, gary], [carla, david, flip], [aldo, beat, hugo, ida]),
             ([evi, gary], [carla, david, hugo], [aldo, beat, flip, ida]),
             ([evi, gary], [carla, david, ida], [aldo, beat, flip, hugo]),
             ([evi, gary], [carla, flip, hugo], [aldo, beat, david, ida]),
             ([evi, gary], [carla, flip, ida], [aldo, beat, david, hugo]),
             ([evi, gary], [carla, hugo, ida], [aldo, beat, david, flip]),
             ([evi, gary], [david, flip, hugo], [aldo, beat, carla, ida]),
             ([evi, gary], [david, flip, ida], [aldo, beat, carla, hugo]),
             ([evi, gary], [david, hugo, ida], [aldo, beat, carla, flip]),
             ([evi, gary], [flip, hugo, ida], [aldo, beat, carla, david]),
             ([evi, hugo], [aldo, beat, carla], [david, flip, gary, ida]),
             ([evi, hugo], [aldo, beat, david], [carla, flip, gary, ida]),
             ([evi, hugo], [aldo, beat, flip], [carla, david, gary, ida]),
             ([evi, hugo], [aldo, beat, gary], [carla, david, flip, ida]),
             ([evi, hugo], [aldo, beat, ida], [carla, david, flip, gary]),
             ([evi, hugo], [aldo, carla, david], [beat, flip, gary, ida]),
             ([evi, hugo], [aldo, carla, flip], [beat, david, gary, ida]),
             ([evi, hugo], [aldo, carla, gary], [beat, david, flip, ida]),
             ([evi, hugo], [aldo, carla, ida], [beat, david, flip, gary]),
             ([evi, hugo], [aldo, david, flip], [beat, carla, gary, ida]),
             ([evi, hugo], [aldo, david, gary], [beat, carla, flip, ida]),
             ([evi, hugo], [aldo, david, ida], [beat, carla, flip, gary]),
             ([evi, hugo], [aldo, flip, gary], [beat, carla, david, ida]),
             ([evi, hugo], [aldo, flip, ida], [beat, carla, david, gary]),
             ([evi, hugo], [aldo, gary, ida], [beat, carla, david, flip]),
             ([evi, hugo], [beat, carla, david], [aldo, flip, gary, ida]),
             ([evi, hugo], [beat, carla, flip], [aldo, david, gary, ida]),
             ([evi, hugo], [beat, carla, gary], [aldo, david, flip, ida]),
             ([evi, hugo], [beat, carla, ida], [aldo, david, flip, gary]),
             ([evi, hugo], [beat, david, flip], [aldo, carla, gary, ida]),
             ([evi, hugo], [beat, david, gary], [aldo, carla, flip, ida]),
             ([evi, hugo], [beat, david, ida], [aldo, carla, flip, gary]),
             ([evi, hugo], [beat, flip, gary], [aldo, carla, david, ida]),
             ([evi, hugo], [beat, flip, ida], [aldo, carla, david, gary]),
             ([evi, hugo], [beat, gary, ida], [aldo, carla, david, flip]),
             ([evi, hugo], [carla, david, flip], [aldo, beat, gary, ida]),
             ([evi, hugo], [carla, david, gary], [aldo, beat, flip, ida]),
             ([evi, hugo], [carla, david, ida], [aldo, beat, flip, gary]),
             ([evi, hugo], [carla, flip, gary], [aldo, beat, david, ida]),
             ([evi, hugo], [carla, flip, ida], [aldo, beat, david, gary]),
             ([evi, hugo], [carla, gary, ida], [aldo, beat, david, flip]),
             ([evi, hugo], [david, flip, gary], [aldo, beat, carla, ida]),
             ([evi, hugo], [david, flip, ida], [aldo, beat, carla, gary]),
             ([evi, hugo], [david, gary, ida], [aldo, beat, carla, flip]),
             ([evi, hugo], [flip, gary, ida], [aldo, beat, carla, david]),
             ([evi, ida], [aldo, beat, carla], [david, flip, gary, hugo]),
             ([evi, ida], [aldo, beat, david], [carla, flip, gary, hugo]),
             ([evi, ida], [aldo, beat, flip], [carla, david, gary, hugo]),
             ([evi, ida], [aldo, beat, gary], [carla, david, flip, hugo]),
             ([evi, ida], [aldo, beat, hugo], [carla, david, flip, gary]),
             ([evi, ida], [aldo, carla, david], [beat, flip, gary, hugo]),
             ([evi, ida], [aldo, carla, flip], [beat, david, gary, hugo]),
             ([evi, ida], [aldo, carla, gary], [beat, david, flip, hugo]),
             ([evi, ida], [aldo, carla, hugo], [beat, david, flip, gary]),
             ([evi, ida], [aldo, david, flip], [beat, carla, gary, hugo]),
             ([evi, ida], [aldo, david, gary], [beat, carla, flip, hugo]),
             ([evi, ida], [aldo, david, hugo], [beat, carla, flip, gary]),
             ([evi, ida], [aldo, flip, gary], [beat, carla, david, hugo]),
             ([evi, ida], [aldo, flip, hugo], [beat, carla, david, gary]),
             ([evi, ida], [aldo, gary, hugo], [beat, carla, david, flip]),
             ([evi, ida], [beat, carla, david], [aldo, flip, gary, hugo]),
             ([evi, ida], [beat, carla, flip], [aldo, david, gary, hugo]),
             ([evi, ida], [beat, carla, gary], [aldo, david, flip, hugo]),
             ([evi, ida], [beat, carla, hugo], [aldo, david, flip, gary]),
             ([evi, ida], [beat, david, flip], [aldo, carla, gary, hugo]),
             ([evi, ida], [beat, david, gary], [aldo, carla, flip, hugo]),
             ([evi, ida], [beat, david, hugo], [aldo, carla, flip, gary]),
             ([evi, ida], [beat, flip, gary], [aldo, carla, david, hugo]),
             ([evi, ida], [beat, flip, hugo], [aldo, carla, david, gary]),
             ([evi, ida], [beat, gary, hugo], [aldo, carla, david, flip]),
             ([evi, ida], [carla, david, flip], [aldo, beat, gary, hugo]),
             ([evi, ida], [carla, david, gary], [aldo, beat, flip, hugo]),
             ([evi, ida], [carla, david, hugo], [aldo, beat, flip, gary]),
             ([evi, ida], [carla, flip, gary], [aldo, beat, david, hugo]),
             ([evi, ida], [carla, flip, hugo], [aldo, beat, david, gary]),
             ([evi, ida], [carla, gary, hugo], [aldo, beat, david, flip]),
             ([evi, ida], [david, flip, gary], [aldo, beat, carla, hugo]),
             ([evi, ida], [david, flip, hugo], [aldo, beat, carla, gary]),
             ([evi, ida], [david, gary, hugo], [aldo, beat, carla, flip]),
             ([evi, ida], [flip, gary, hugo], [aldo, beat, carla, david]),
             ([flip, gary], [aldo, beat, carla], [david, evi, hugo, ida]),
             ([flip, gary], [aldo, beat, david], [carla, evi, hugo, ida]),
             ([flip, gary], [aldo, beat, evi], [carla, david, hugo, ida]),
             ([flip, gary], [aldo, beat, hugo], [carla, david, evi, ida]),
             ([flip, gary], [aldo, beat, ida], [carla, david, evi, hugo]),
             ([flip, gary], [aldo, carla, david], [beat, evi, hugo, ida]),
             ([flip, gary], [aldo, carla, evi], [beat, david, hugo, ida]),
             ([flip, gary], [aldo, carla, hugo], [beat, david, evi, ida]),
             ([flip, gary], [aldo, carla, ida], [beat, david, evi, hugo]),
             ([flip, gary], [aldo, david, evi], [beat, carla, hugo, ida]),
             ([flip, gary], [aldo, david, hugo], [beat, carla, evi, ida]),
             ([flip, gary], [aldo, david, ida], [beat, carla, evi, hugo]),
             ([flip, gary], [aldo, evi, hugo], [beat, carla, david, ida]),
             ([flip, gary], [aldo, evi, ida], [beat, carla, david, hugo]),
             ([flip, gary], [aldo, hugo, ida], [beat, carla, david, evi]),
             ([flip, gary], [beat, carla, david], [aldo, evi, hugo, ida]),
             ([flip, gary], [beat, carla, evi], [aldo, david, hugo, ida]),
             ([flip, gary], [beat, carla, hugo], [aldo, david, evi, ida]),
             ([flip, gary], [beat, carla, ida], [aldo, david, evi, hugo]),
             ([flip, gary], [beat, david, evi], [aldo, carla, hugo, ida]),
             ([flip, gary], [beat, david, hugo], [aldo, carla, evi, ida]),
             ([flip, gary], [beat, david, ida], [aldo, carla, evi, hugo]),
             ([flip, gary], [beat, evi, hugo], [aldo, carla, david, ida]),
             ([flip, gary], [beat, evi, ida], [aldo, carla, david, hugo]),
             ([flip, gary], [beat, hugo, ida], [aldo, carla, david, evi]),
             ([flip, gary], [carla, david, evi], [aldo, beat, hugo, ida]),
             ([flip, gary], [carla, david, hugo], [aldo, beat, evi, ida]),
             ([flip, gary], [carla, david, ida], [aldo, beat, evi, hugo]),
             ([flip, gary], [carla, evi, hugo], [aldo, beat, david, ida]),
             ([flip, gary], [carla, evi, ida], [aldo, beat, david, hugo]),
             ([flip, gary], [carla, hugo, ida], [aldo, beat, david, evi]),
             ([flip, gary], [david, evi, hugo], [aldo, beat, carla, ida]),
             ([flip, gary], [david, evi, ida], [aldo, beat, carla, hugo]),
             ([flip, gary], [david, hugo, ida], [aldo, beat, carla, evi]),
             ([flip, gary], [evi, hugo, ida], [aldo, beat, carla, david]),
             ([flip, hugo], [aldo, beat, carla], [david, evi, gary, ida]),
             ([flip, hugo], [aldo, beat, david], [carla, evi, gary, ida]),
             ([flip, hugo], [aldo, beat, evi], [carla, david, gary, ida]),
             ([flip, hugo], [aldo, beat, gary], [carla, david, evi, ida]),
             ([flip, hugo], [aldo, beat, ida], [carla, david, evi, gary]),
             ([flip, hugo], [aldo, carla, david], [beat, evi, gary, ida]),
             ([flip, hugo], [aldo, carla, evi], [beat, david, gary, ida]),
             ([flip, hugo], [aldo, carla, gary], [beat, david, evi, ida]),
             ([flip, hugo], [aldo, carla, ida], [beat, david, evi, gary]),
             ([flip, hugo], [aldo, david, evi], [beat, carla, gary, ida]),
             ([flip, hugo], [aldo, david, gary], [beat, carla, evi, ida]),
             ([flip, hugo], [aldo, david, ida], [beat, carla, evi, gary]),
             ([flip, hugo], [aldo, evi, gary], [beat, carla, david, ida]),
             ([flip, hugo], [aldo, evi, ida], [beat, carla, david, gary]),
             ([flip, hugo], [aldo, gary, ida], [beat, carla, david, evi]),
             ([flip, hugo], [beat, carla, david], [aldo, evi, gary, ida]),
             ([flip, hugo], [beat, carla, evi], [aldo, david, gary, ida]),
             ([flip, hugo], [beat, carla, gary], [aldo, david, evi, ida]),
             ([flip, hugo], [beat, carla, ida], [aldo, david, evi, gary]),
             ([flip, hugo], [beat, david, evi], [aldo, carla, gary, ida]),
             ([flip, hugo], [beat, david, gary], [aldo, carla, evi, ida]),
             ([flip, hugo], [beat, david, ida], [aldo, carla, evi, gary]),
             ([flip, hugo], [beat, evi, gary], [aldo, carla, david, ida]),
             ([flip, hugo], [beat, evi, ida], [aldo, carla, david, gary]),
             ([flip, hugo], [beat, gary, ida], [aldo, carla, david, evi]),
             ([flip, hugo], [carla, david, evi], [aldo, beat, gary, ida]),
             ([flip, hugo], [carla, david, gary], [aldo, beat, evi, ida]),
             ([flip, hugo], [carla, david, ida], [aldo, beat, evi, gary]),
             ([flip, hugo], [carla, evi, gary], [aldo, beat, david, ida]),
             ([flip, hugo], [carla, evi, ida], [aldo, beat, david, gary]),
             ([flip, hugo], [carla, gary, ida], [aldo, beat, david, evi]),
             ([flip, hugo], [david, evi, gary], [aldo, beat, carla, ida]),
             ([flip, hugo], [david, evi, ida], [aldo, beat, carla, gary]),
             ([flip, hugo], [david, gary, ida], [aldo, beat, carla, evi]),
             ([flip, hugo], [evi, gary, ida], [aldo, beat, carla, david]),
             ([flip, ida], [aldo, beat, carla], [david, evi, gary, hugo]),
             ([flip, ida], [aldo, beat, david], [carla, evi, gary, hugo]),
             ([flip, ida], [aldo, beat, evi], [carla, david, gary, hugo]),
             ([flip, ida], [aldo, beat, gary], [carla, david, evi, hugo]),
             ([flip, ida], [aldo, beat, hugo], [carla, david, evi, gary]),
             ([flip, ida], [aldo, carla, david], [beat, evi, gary, hugo]),
             ([flip, ida], [aldo, carla, evi], [beat, david, gary, hugo]),
             ([flip, ida], [aldo, carla, gary], [beat, david, evi, hugo]),
             ([flip, ida], [aldo, carla, hugo], [beat, david, evi, gary]),
             ([flip, ida], [aldo, david, evi], [beat, carla, gary, hugo]),
             ([flip, ida], [aldo, david, gary], [beat, carla, evi, hugo]),
             ([flip, ida], [aldo, david, hugo], [beat, carla, evi, gary]),
             ([flip, ida], [aldo, evi, gary], [beat, carla, david, hugo]),
             ([flip, ida], [aldo, evi, hugo], [beat, carla, david, gary]),
             ([flip, ida], [aldo, gary, hugo], [beat, carla, david, evi]),
             ([flip, ida], [beat, carla, david], [aldo, evi, gary, hugo]),
             ([flip, ida], [beat, carla, evi], [aldo, david, gary, hugo]),
             ([flip, ida], [beat, carla, gary], [aldo, david, evi, hugo]),
             ([flip, ida], [beat, carla, hugo], [aldo, david, evi, gary]),
             ([flip, ida], [beat, david, evi], [aldo, carla, gary, hugo]),
             ([flip, ida], [beat, david, gary], [aldo, carla, evi, hugo]),
             ([flip, ida], [beat, david, hugo], [aldo, carla, evi, gary]),
             ([flip, ida], [beat, evi, gary], [aldo, carla, david, hugo]),
             ([flip, ida], [beat, evi, hugo], [aldo, carla, david, gary]),
             ([flip, ida], [beat, gary, hugo], [aldo, carla, david, evi]),
             ([flip, ida], [carla, david, evi], [aldo, beat, gary, hugo]),
             ([flip, ida], [carla, david, gary], [aldo, beat, evi, hugo]),
             ([flip, ida], [carla, david, hugo], [aldo, beat, evi, gary]),
             ([flip, ida], [carla, evi, gary], [aldo, beat, david, hugo]),
             ([flip, ida], [carla, evi, hugo], [aldo, beat, david, gary]),
             ([flip, ida], [carla, gary, hugo], [aldo, beat, david, evi]),
             ([flip, ida], [david, evi, gary], [aldo, beat, carla, hugo]),
             ([flip, ida], [david, evi, hugo], [aldo, beat, carla, gary]),
             ([flip, ida], [david, gary, hugo], [aldo, beat, carla, evi]),
             ([flip, ida], [evi, gary, hugo], [aldo, beat, carla, david]),
             ([gary, hugo], [aldo, beat, carla], [david, evi, flip, ida]),
             ([gary, hugo], [aldo, beat, david], [carla, evi, flip, ida]),
             ([gary, hugo], [aldo, beat, evi], [carla, david, flip, ida]),
             ([gary, hugo], [aldo, beat, flip], [carla, david, evi, ida]),
             ([gary, hugo], [aldo, beat, ida], [carla, david, evi, flip]),
             ([gary, hugo], [aldo, carla, david], [beat, evi, flip, ida]),
             ([gary, hugo], [aldo, carla, evi], [beat, david, flip, ida]),
             ([gary, hugo], [aldo, carla, flip], [beat, david, evi, ida]),
             ([gary, hugo], [aldo, carla, ida], [beat, david, evi, flip]),
             ([gary, hugo], [aldo, david, evi], [beat, carla, flip, ida]),
             ([gary, hugo], [aldo, david, flip], [beat, carla, evi, ida]),
             ([gary, hugo], [aldo, david, ida], [beat, carla, evi, flip]),
             ([gary, hugo], [aldo, evi, flip], [beat, carla, david, ida]),
             ([gary, hugo], [aldo, evi, ida], [beat, carla, david, flip]),
             ([gary, hugo], [aldo, flip, ida], [beat, carla, david, evi]),
             ([gary, hugo], [beat, carla, david], [aldo, evi, flip, ida]),
             ([gary, hugo], [beat, carla, evi], [aldo, david, flip, ida]),
             ([gary, hugo], [beat, carla, flip], [aldo, david, evi, ida]),
             ([gary, hugo], [beat, carla, ida], [aldo, david, evi, flip]),
             ([gary, hugo], [beat, david, evi], [aldo, carla, flip, ida]),
             ([gary, hugo], [beat, david, flip], [aldo, carla, evi, ida]),
             ([gary, hugo], [beat, david, ida], [aldo, carla, evi, flip]),
             ([gary, hugo], [beat, evi, flip], [aldo, carla, david, ida]),
             ([gary, hugo], [beat, evi, ida], [aldo, carla, david, flip]),
             ([gary, hugo], [beat, flip, ida], [aldo, carla, david, evi]),
             ([gary, hugo], [carla, david, evi], [aldo, beat, flip, ida]),
             ([gary, hugo], [carla, david, flip], [aldo, beat, evi, ida]),
             ([gary, hugo], [carla, david, ida], [aldo, beat, evi, flip]),
             ([gary, hugo], [carla, evi, flip], [aldo, beat, david, ida]),
             ([gary, hugo], [carla, evi, ida], [aldo, beat, david, flip]),
             ([gary, hugo], [carla, flip, ida], [aldo, beat, david, evi]),
             ([gary, hugo], [david, evi, flip], [aldo, beat, carla, ida]),
             ([gary, hugo], [david, evi, ida], [aldo, beat, carla, flip]),
             ([gary, hugo], [david, flip, ida], [aldo, beat, carla, evi]),
             ([gary, hugo], [evi, flip, ida], [aldo, beat, carla, david]),
             ([gary, ida], [aldo, beat, carla], [david, evi, flip, hugo]),
             ([gary, ida], [aldo, beat, david], [carla, evi, flip, hugo]),
             ([gary, ida], [aldo, beat, evi], [carla, david, flip, hugo]),
             ([gary, ida], [aldo, beat, flip], [carla, david, evi, hugo]),
             ([gary, ida], [aldo, beat, hugo], [carla, david, evi, flip]),
             ([gary, ida], [aldo, carla, david], [beat, evi, flip, hugo]),
             ([gary, ida], [aldo, carla, evi], [beat, david, flip, hugo]),
             ([gary, ida], [aldo, carla, flip], [beat, david, evi, hugo]),
             ([gary, ida], [aldo, carla, hugo], [beat, david, evi, flip]),
             ([gary, ida], [aldo, david, evi], [beat, carla, flip, hugo]),
             ([gary, ida], [aldo, david, flip], [beat, carla, evi, hugo]),
             ([gary, ida], [aldo, david, hugo], [beat, carla, evi, flip]),
             ([gary, ida], [aldo, evi, flip], [beat, carla, david, hugo]),
             ([gary, ida], [aldo, evi, hugo], [beat, carla, david, flip]),
             ([gary, ida], [aldo, flip, hugo], [beat, carla, david, evi]),
             ([gary, ida], [beat, carla, david], [aldo, evi, flip, hugo]),
             ([gary, ida], [beat, carla, evi], [aldo, david, flip, hugo]),
             ([gary, ida], [beat, carla, flip], [aldo, david, evi, hugo]),
             ([gary, ida], [beat, carla, hugo], [aldo, david, evi, flip]),
             ([gary, ida], [beat, david, evi], [aldo, carla, flip, hugo]),
             ([gary, ida], [beat, david, flip], [aldo, carla, evi, hugo]),
             ([gary, ida], [beat, david, hugo], [aldo, carla, evi, flip]),
             ([gary, ida], [beat, evi, flip], [aldo, carla, david, hugo]),
             ([gary, ida], [beat, evi, hugo], [aldo, carla, david, flip]),
             ([gary, ida], [beat, flip, hugo], [aldo, carla, david, evi]),
             ([gary, ida], [carla, david, evi], [aldo, beat, flip, hugo]),
             ([gary, ida], [carla, david, flip], [aldo, beat, evi, hugo]),
             ([gary, ida], [carla, david, hugo], [aldo, beat, evi, flip]),
             ([gary, ida], [carla, evi, flip], [aldo, beat, david, hugo]),
             ([gary, ida], [carla, evi, hugo], [aldo, beat, david, flip]),
             ([gary, ida], [carla, flip, hugo], [aldo, beat, david, evi]),
             ([gary, ida], [david, evi, flip], [aldo, beat, carla, hugo]),
             ([gary, ida], [david, evi, hugo], [aldo, beat, carla, flip]),
             ([gary, ida], [david, flip, hugo], [aldo, beat, carla, evi]),
             ([gary, ida], [evi, flip, hugo], [aldo, beat, carla, david]),
             ([hugo, ida], [aldo, beat, carla], [david, evi, flip, gary]),
             ([hugo, ida], [aldo, beat, david], [carla, evi, flip, gary]),
             ([hugo, ida], [aldo, beat, evi], [carla, david, flip, gary]),
             ([hugo, ida], [aldo, beat, flip], [carla, david, evi, gary]),
             ([hugo, ida], [aldo, beat, gary], [carla, david, evi, flip]),
             ([hugo, ida], [aldo, carla, david], [beat, evi, flip, gary]),
             ([hugo, ida], [aldo, carla, evi], [beat, david, flip, gary]),
             ([hugo, ida], [aldo, carla, flip], [beat, david, evi, gary]),
             ([hugo, ida], [aldo, carla, gary], [beat, david, evi, flip]),
             ([hugo, ida], [aldo, david, evi], [beat, carla, flip, gary]),
             ([hugo, ida], [aldo, david, flip], [beat, carla, evi, gary]),
             ([hugo, ida], [aldo, david, gary], [beat, carla, evi, flip]),
             ([hugo, ida], [aldo, evi, flip], [beat, carla, david, gary]),
             ([hugo, ida], [aldo, evi, gary], [beat, carla, david, flip]),
             ([hugo, ida], [aldo, flip, gary], [beat, carla, david, evi]),
             ([hugo, ida], [beat, carla, david], [aldo, evi, flip, gary]),
             ([hugo, ida], [beat, carla, evi], [aldo, david, flip, gary]),
             ([hugo, ida], [beat, carla, flip], [aldo, david, evi, gary]),
             ([hugo, ida], [beat, carla, gary], [aldo, david, evi, flip]),
             ([hugo, ida], [beat, david, evi], [aldo, carla, flip, gary]),
             ([hugo, ida], [beat, david, flip], [aldo, carla, evi, gary]),
             ([hugo, ida], [beat, david, gary], [aldo, carla, evi, flip]),
             ([hugo, ida], [beat, evi, flip], [aldo, carla, david, gary]),
             ([hugo, ida], [beat, evi, gary], [aldo, carla, david, flip]),
             ([hugo, ida], [beat, flip, gary], [aldo, carla, david, evi]),
             ([hugo, ida], [carla, david, evi], [aldo, beat, flip, gary]),
             ([hugo, ida], [carla, david, flip], [aldo, beat, evi, gary]),
             ([hugo, ida], [carla, david, gary], [aldo, beat, evi, flip]),
             ([hugo, ida], [carla, evi, flip], [aldo, beat, david, gary]),
             ([hugo, ida], [carla, evi, gary], [aldo, beat, david, flip]),
             ([hugo, ida], [carla, flip, gary], [aldo, beat, david, evi]),
             ([hugo, ida], [david, evi, flip], [aldo, beat, carla, gary]),
             ([hugo, ida], [david, evi, gary], [aldo, beat, carla, flip]),
             ([hugo, ida], [david, flip, gary], [aldo, beat, carla, evi]),
             ([hugo, ida], [evi, flip, gary], [aldo, beat, carla, david])];
    }

}
