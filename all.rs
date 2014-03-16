#[feature(macro_rules)];
#[feature(globs)];
#[allow(deprecated_owned_vector)];

extern crate rand;
extern crate num;

use std::from_str::from_str;

pub mod list_work;
pub mod arithmetic;

pub fn main() {
    println!("Hello World");
    let bigint : num::bigint::BigInt = from_str("10000000").unwrap();
    println!("goldbach({}): {}", bigint.clone(), arithmetic::goldbach(bigint));

    // let bigint : num::bigint::BigInt = from_str("1000000000000000000000").unwrap();
    // println!("goldbach({}): {}", bigint, arithmetic::goldbach(bigint));
}
