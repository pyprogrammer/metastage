use std::{cell::RefCell, fmt::Debug, hash::Hash, rc::Rc};

use fxhash::FxHashMap;

#[derive(Default, Debug)]
struct Counter(usize);
impl Counter {
    fn next(&mut self) -> usize {
        self.0 += 1;
        self.0 - 1
    }
}

#[derive(Clone, Copy, Eq, PartialEq, Hash)]
pub struct Sym {
    pub id: usize,
}

impl std::fmt::Debug for Sym {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "s{:?}", self.id)
    }
}

pub trait Expr {
    fn arity(&self) -> usize;
    fn inputs(&self) -> Vec<Sym>;

    fn stage(self, scope: &Rc<RefCell<Scope<Self>>>) -> Vec<Sym>
    where
        Self: PartialEq + Eq + std::hash::Hash + Expr + Debug + Sized,
    {
        scope.borrow_mut().stage(self)
    }
}

#[derive(Debug)]
pub struct Scope<T> {
    cache: FxHashMap<T, Vec<Sym>>,
    counter: Counter,
}

impl<T> Scope<T>
where
    T: PartialEq + Eq + std::hash::Hash + Expr + Debug,
{
    pub fn stage(&mut self, expr: T) -> Vec<Sym> {
        match self.cache.get(&expr) {
            Some(existing) => existing.clone(),
            None => {
                let new_syms: Vec<_> = (0..expr.arity())
                    .map(|_| Sym {
                        id: self.counter.next(),
                    })
                    .collect();
                self.cache.insert(expr, new_syms.clone());
                new_syms
            }
        }
    }
    pub fn print(&self) {
        // sort cache by roots
        let mut refs_and_deps: Vec<_> = self.cache.iter().map(|(k, v)| (v, k)).collect();
        refs_and_deps.sort_unstable_by_key(|(deps, _)| deps.iter().map(|Sym { id }| *id).max());
        for (sym, exprs) in refs_and_deps {
            let lhs = sym
                .iter()
                .map(|s| format!("{:?}", *s))
                .collect::<Vec<_>>()
                .join(", ");
            println!("{lhs} = {:?}", exprs);
        }
    }
}

impl<T> Default for Scope<T> {
    fn default() -> Self {
        Self {
            cache: Default::default(),
            counter: Default::default(),
        }
    }
}

pub type ScopeRef<E> = Rc<RefCell<Scope<E>>>;
