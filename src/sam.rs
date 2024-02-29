use crate::sym::{Expr, Sym};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Copy)]
pub enum JoinType {
    Intersect,
    Union,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Copy)]
pub enum PrimitiveOp {
    Mul,
    Add,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SamOps {
    Fiberlookup {
        reference: Sym,
        tensor: String,
        level: usize,
    },
    Repeat {
        target: Sym,
        repeat: Sym,
    },
    Arrayval {
        reference: Sym,
        tensor: String,
    },
    Join {
        ref1: Sym,
        ref2: Sym,
        crd1: Sym,
        crd2: Sym,
        tp: JoinType,
    },
    Reduce {
        inputs: Sym,
        op: PrimitiveOp,
    },
    ALU {
        op: PrimitiveOp,
        inputs: Vec<Sym>,
    },
    CoordDrop {
        inner: Sym,
        outer: Sym,
    },
    Root,
    Genref {
        coords: Sym,
    },
}

impl Expr for SamOps {
    fn arity(&self) -> usize {
        match self {
            SamOps::Fiberlookup { .. } => 2,
            SamOps::Repeat { .. } => 1,
            SamOps::Arrayval { .. } => 1,
            SamOps::Join { .. } => 3,
            SamOps::Reduce { .. } => 1,
            SamOps::ALU { .. } => 1,
            SamOps::CoordDrop { .. } => 1,
            SamOps::Root => 1,
            SamOps::Genref { .. } => 1,
        }
    }

    fn inputs(&self) -> Vec<Sym> {
        match self {
            SamOps::Fiberlookup {
                reference,
                tensor: _,
                level: _,
            } => vec![*reference],
            SamOps::Repeat { target, repeat } => vec![*target, *repeat],
            SamOps::Arrayval {
                reference,
                tensor: _,
            } => vec![*reference],
            SamOps::Join {
                ref1,
                ref2,
                crd1,
                crd2,
                tp: _,
            } => vec![*ref1, *ref2, *crd1, *crd2],
            SamOps::Reduce { inputs , .. } => vec![*inputs],
            SamOps::ALU { op: _, inputs } => inputs.to_vec(),
            SamOps::CoordDrop { inner, outer } => vec![*inner, *outer],
            SamOps::Root => vec![],
            SamOps::Genref { coords } => vec![*coords],
        }
    }

    fn simplify(self, scope: &crate::sym::Scope<Self>) -> Self
    where
        Self: PartialEq + Eq + std::hash::Hash + Expr + std::fmt::Debug + Sized,
    {
        match self {
            SamOps::Repeat { target, repeat } => {
                if let Some(&SamOps::Root) = scope.lookup(repeat) {
                    dbg!("Eliding Repeat w.r.t. Root!");
                    scope.lookup(target).unwrap().clone()
                } else {
                    self
                }
            }
            _ => self,
        }
    }
}

#[cfg(test)]
mod test {
    use crate::sym::Scope;

    use super::SamOps;

    #[test]
    fn simple_stage() {
        let mut scope = Scope::default();
        let root = scope.stage(SamOps::Root)[0];
        let read1 = scope.stage(SamOps::Fiberlookup {
            reference: root,
            tensor: "A".to_string(),
            level: 0,
        });
        let _ = scope.stage(SamOps::Fiberlookup {
            reference: read1[0],
            tensor: "A".to_string(),
            level: 1,
        });
        scope.print();
    }
}
