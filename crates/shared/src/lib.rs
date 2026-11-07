mod fx;
mod id;
mod interner;
mod literal;
mod location;
mod num;
pub mod schema;
mod ty;
pub mod units;

pub use fx::*;
pub use id::*;
pub use interner::*;
pub use literal::*;
pub use location::*;
pub use num::*;
pub use ty::*;

pub type Error = miette::Report;
pub type Result<T> = std::result::Result<T, Error>;

id!(pub AdtId);
id!(pub BodyId);
