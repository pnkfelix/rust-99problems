use std::vec;
use std::num::pow;
use std::num::Bitwise;

// Logic and Codes
// 
// P46 (**) Truth tables for logical expressions.
//     Define predicates and/2, or/2, nand/2, nor/2, xor/2, impl/2 and equ/2 (for logical equivalence) which succeed or fail according to the result of their respective operations; e.g. and(A,B) will succeed, if and only if both A and B succeed. Note that A and B can be Prolog goals (not only the constants true and fail).
// 
//     A logical expression in two variables can then be written in prefix notation, as in the following example: and(or(A,B),nand(A,B)).
// 
//     Now, write a predicate table/3 which prints the truth table of a given logical expression in two variables.
// 
//     Example:
//     * table(A,B,and(A,or(A,B))).
//     true true true
//     true fail true
//     fail true fail
//     fail fail fail

#[deriving(Show, Eq, Clone)]
enum Goal {
    True,
    Fail,
}

#[deriving(Show, Eq, Clone)]
enum Expr<Sym> {
    K(Goal),
    Var(Sym),
    And(~Expr<Sym>, ~Expr<Sym>),
    Or(~Expr<Sym>, ~Expr<Sym>),
    Not(~Expr<Sym>),
    Nand(~Expr<Sym>, ~Expr<Sym>),
    Nor(~Expr<Sym>, ~Expr<Sym>),
    Xor(~Expr<Sym>, ~Expr<Sym>),
    Impl(~Expr<Sym>, ~Expr<Sym>),
    Equ(~Expr<Sym>, ~Expr<Sym>),
}

fn interp<Sym:Eq>(env: |Sym| -> Goal, expr: Expr<Sym>) -> Goal {
    match expr {
        K(g)      => g,
        Var(s)    => env(s),
        And(~a, ~b) => match interp(|s|env(s), a) { Fail => Fail, True => interp(env, b), },
        Or(~a, ~b)  => match interp(|s|env(s), a) { True => True, Fail => interp(env, b), },
        Not(~a)     => match interp(|s|env(s), a) { True => Fail, Fail => True },
        Nand(~a,~b) => match interp(|s|env(s), a) {
            Fail => True, True => match interp(env, b) { Fail => True, True => Fail }
        },
        Nor(~a,~b)  => match interp(|s|env(s), a) {
            True => Fail, Fail => match interp(env, b) { True => Fail, Fail => True }
        },
        Xor(~a,~b)  => match (interp(|s|env(s), a), interp(|s|env(s), b)) {
            (True, True) | (Fail, Fail) => Fail,
            (Fail, True) | (True, Fail) => True,
        },
        Impl(~a,~b) => match interp(|s|env(s), a) {
            Fail => True, True => match interp(|s|env(s), b) { True => True, Fail => Fail }
        },
        Equ(~a,~b)  => match (interp(|s|env(s), a), interp(|s|env(s), b)) {
            (True, True) | (Fail, Fail) => True,
            (Fail, True) | (True, Fail) => Fail,
        },
    }
}

trait ToSymList<S> {
    fn to_sym_list(&self) -> ~[S];
}

impl<S:Clone> ToSymList<S> for (S,S) {
    fn to_sym_list(&self) -> ~[S] {
        let &(ref a, ref b) = self;
        ~[a.clone(), b.clone()]
    }
}

fn subseqs<X:Clone>(lst: &[X]) -> ~[~[X]] {
    let count = pow(2u, lst.len());
    let mut ret = vec::with_capacity(count);
    for i in range(0,count) {
        let mut curr = vec::with_capacity(i.count_ones());
        for j in range(0, lst.len()) {
            if i & (1 << j) != 0 {
                curr.push(lst[j].clone());
            }
        }
        ret.push(curr);
    }
    ret
}

fn table<Sym:Eq+Clone+TotalOrd, L:ToSymList<Sym>>(l:L, expr: Expr<Sym>) -> ~[(~[Goal], Goal)] {
    let all_syms = l.to_sym_list();
    let subseqs = subseqs(all_syms);
    subseqs.move_iter().rev().map(|syms| {
        let env = |s:Sym| -> Goal {
            if syms.contains(&s) {
                True
            } else {
                Fail
            }
        };
        let context : ~[_] = all_syms.iter().map(|s|env(s.clone())).collect();
        let input : Expr<Sym> = expr.clone();
        (context, interp(env, input))
    }).collect()
}

#[test] 
fn test_table() {
    let a = Var("a");
    let b = Var("b");
    assert_eq!(table(("a", "b"), And(~a.clone(), ~Or(~a.clone(), ~b.clone()))),
               ~[(~[True, True], True),
                 (~[Fail, True], Fail),
                 (~[True, Fail], True),
                 (~[Fail, Fail], Fail)]);
}


// P47 (*) Truth tables for logical expressions (2).
//     Continue problem P46 by defining and/2, or/2, etc as being operators. This allows to write the logical expression in the more natural way, as in the example: A and (A or not B). Define operator precedence as usual; i.e. as in Java.
// 
//     Example:
//     * table(A,B, A and (A or not B)).
//     true true true
//     true fail true
//     fail true fail
//     fail fail fail

impl<S:Clone> BitAnd<Expr<S>, Expr<S>> for Expr<S> {
    fn bitand(&self, rhs: &Expr<S>) -> Expr<S> { And(~self.clone(), ~rhs.clone()) }
}

impl<S:Clone> BitOr<Expr<S>, Expr<S>> for Expr<S> {
    fn bitor(&self, rhs: &Expr<S>) -> Expr<S> { Or(~self.clone(), ~rhs.clone()) }
}

impl<S:Clone> Not<Expr<S>> for Expr<S> {
    fn not(&self) -> Expr<S> {
        Not(~self.clone())
    }
}

impl<S:Clone> BitXor<Expr<S>, Expr<S>> for Expr<S> {
    fn bitxor(&self, rhs: &Expr<S>) -> Expr<S> {
        Xor(~self.clone(), ~rhs.clone())
    }
}

/// hack: use >> to denote EQL for logical expressions, since I cannot
/// override the return type for Eq trait.
impl<S:Clone> Shr<Expr<S>, Expr<S>> for Expr<S> {
    fn shr(&self, rhs: &Expr<S>) -> Expr<S> {
        Equ(~self.clone(), ~rhs.clone())
    }
}

#[test]
fn test_table_ops_sugar() {
    let a = Var("a");
    let b = Var("b");
    assert_eq!(table(("a", "b"), a & (a | ! b)),
               ~[(~[True, True], True),
                 (~[Fail, True], Fail),
                 (~[True, Fail], True),
                 (~[Fail, Fail], Fail)]);
}

// P48 (**) Truth tables for logical expressions (3).
//     Generalize problem P47 in such a way that the logical expression may contain any number of logical variables. Define table/2 in a way that table(List,Expr) prints the truth table for the expression Expr, which contains the logical variables enumerated in List.
// 
//     Example:
//     * table([A,B,C], A and (B or C) equ A and B or A and C).
//     true true true true
//     true true fail true
//     true fail true true
//     true fail fail true
//     fail true true true
//     fail true fail true
//     fail fail true true
//     fail fail fail true

impl<S:Clone> ToSymList<S> for ~[S] {
    fn to_sym_list(&self) -> ~[S] { self.clone() }
}

#[test]
fn test_table_generalized() {
    let a = Var("a");
    let b = Var("b");
    let c = Var("c");
    assert_eq!(table(~["a", "b", "c"], (a & (b | c)) >> (a & b) | (a & c)),
               ~[(~[True, True, True], True),
                 (~[Fail, True, True], True),
                 (~[True, Fail, True], True),
                 (~[Fail, Fail, True], True),
                 (~[True, True, Fail], True),
                 (~[Fail, True, Fail], True),
                 (~[True, Fail, Fail], True),
                 (~[Fail, Fail, Fail], True)]);
}

// 
// P49 (**) Gray code.
//     An n-bit Gray code is a sequence of n-bit strings constructed according to certain rules. For example,
//     n = 1: C(1) = ['0','1'].
//     n = 2: C(2) = ['00','01','11','10'].
//     n = 3: C(3) = ['000','001','011','010',´110´,´111´,´101´,´100´].
// 
//     Find out the construction rules and write a predicate with the following specification:
// 
//     % gray(N,C) :- C is the N-bit Gray code
// 
//     Can you apply the method of "result caching" in order to make the predicate more efficient, when it is to be used repeatedly?
// 
// P50 (***) Huffman code.
//     First of all, consult a good book on discrete mathematics or algorithms for a detailed description of Huffman codes!
// 
//     We suppose a set of symbols with their frequencies, given as a list of fr(S,F) terms. Example: [fr(a,45),fr(b,13),fr(c,12),fr(d,16),fr(e,9),fr(f,5)]. Our objective is to construct a list hc(S,C) terms, where C is the Huffman code word for the symbol S. In our example, the result could be Hs = [hc(a,'0'), hc(b,'101'), hc(c,'100'), hc(d,'111'), hc(e,'1101'), hc(f,'1100')] [hc(a,'01'),...etc.]. The task shall be performed by the predicate huffman/2 defined as follows:
// 
//     % huffman(Fs,Hs) :- Hs is the Huffman code table for the frequency table Fs

