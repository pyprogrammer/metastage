use std::{cell::RefCell, fmt::Debug, hash::Hash, rc::Rc};

use fxhash::{FxHashMap, FxHashSet};

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

    fn stage(self, scope: &ScopeRef<Self>) -> Vec<Sym>
    where
        Self: PartialEq + Eq + std::hash::Hash + Expr + Debug + Sized,
    {
        scope.borrow_mut().stage(self)
    }

    fn simplify(self, _scope: &Scope<Self>) -> Self
    where
        Self: PartialEq + Eq + std::hash::Hash + Expr + Debug + Sized,
    {
        self
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
        let simplified = expr.simplify(&self);
        match self.cache.get(&simplified) {
            Some(existing) => existing.clone(),
            None => {
                let new_syms: Vec<_> = (0..simplified.arity())
                    .map(|_| Sym {
                        id: self.counter.next(),
                    })
                    .collect();
                self.cache.insert(simplified, new_syms.clone());
                new_syms
            }
        }
    }

    pub fn lookup<'a>(&'a self, sym: Sym) -> Option<&'a T> {
        let filtered = self.cache.iter().find(|(_, syms)| {syms.contains(&sym)});
        filtered.map(|x|x.0)
    }

    fn program_order<'a>(
        &'a self,
    ) -> impl Iterator<Item = (&T, &Vec<Sym>)> + DoubleEndedIterator + 'a {
        let mut refs_and_deps: Vec<_> = self.cache.iter().collect();
        refs_and_deps.sort_unstable_by_key(|(_, deps)| deps.iter().map(|Sym { id }| *id).max());
        refs_and_deps.into_iter()
    }

    pub fn print(&self) {
        // sort cache by roots
        for (exprs, sym) in self.program_order() {
            let lhs = sym
                .iter()
                .map(|s| format!("{:?}", *s))
                .collect::<Vec<_>>()
                .join(", ");
            println!("{lhs} = {:?}", exprs);
        }
    }

    fn calculate_live_syms(&self, mut roots: FxHashSet<Sym>) -> FxHashSet<Sym> {
        for (expr, outputs) in self.program_order().rev() {
            // If any of the outputs are in the roots set, then the op is live.
            if outputs.iter().any(|output| roots.contains(output)) {
                roots.extend(expr.inputs().iter())
            }
        }
        roots
    }

    pub fn eliminate_dead_code(&mut self, roots: FxHashSet<Sym>) {
        let live = self.calculate_live_syms(roots);
        self.cache.retain(|_, v| v.iter().any(|x| live.contains(x)));
    }

    pub fn to_dot(&self) -> graphviz_rust::dot_structures::Graph {
        use graphviz_rust::{dot_generator::*, dot_structures::*};

        let mut stmts = vec![];
        for (i, (expr, syms)) in self.program_order().enumerate() {
            let ident = format!("op_{i}");
            let nodelabel = format!("{:?}", expr).replace("\"", "\\\"");
            stmts.push(
                node!(
                    ident,
                    vec![
                        attr!("label", esc nodelabel),
                        attr!("color", "turquoise"),
                        attr!("style", "filled")
                    ]
                )
                .into(),
            );
            for sym in syms {
                stmts.push(
                    edge!(
                        node_id!(ident) => node_id!(sym.id), vec![attr!("arrowhead", "none")]
                    )
                    .into(),
                )
            }
            for input in expr.inputs() {
                stmts.push(
                    edge!(
                        node_id!(input.id) => node_id!(ident)
                    )
                    .into(),
                );
            }
        }

        Graph::DiGraph {
            id: id!("ProgramGraph"),
            strict: false,
            stmts,
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
