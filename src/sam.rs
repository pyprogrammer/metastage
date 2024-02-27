use crate::sym::{Expr, Sym};


#[derive(Debug, Clone, PartialEq, Eq, Hash, Copy)]
pub enum JoinType {
    Intersect,
    Union
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Copy)]
pub enum PrimitiveOp {
    Mul, Add
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
        tp: JoinType
    },
    Reduce {
        inputs: Sym,
    },
    ALU {
        op: PrimitiveOp,
        inputs: Vec<Sym>
    },
    CoordDrop {
        inner: Sym,
        outer: Sym,
    },
    Root,
}

impl Expr for SamOps {
    fn arity(&self) -> usize {
        match self {
            SamOps::Fiberlookup { reference: _, tensor: _, level: _ } => 2,
            SamOps::Repeat { target: _, repeat: _ } => 1,
            SamOps::Arrayval { reference: _, tensor: _ } => 1,
            SamOps::Join { ref1: _, ref2: _, crd1: _, crd2: _, tp: _ } => 3,
            SamOps::Reduce { inputs: _ } => 1,
            SamOps::ALU { op: _, inputs: _ } => 1,
            SamOps::CoordDrop { inner: _, outer: _ } => 1,
            SamOps::Root => 1,
        }
    }

    fn inputs(&self) -> Vec<Sym> {
        match self {
            SamOps::Fiberlookup { reference, tensor: _, level: _ } => vec![*reference],
            SamOps::Repeat { target, repeat } => vec![*target, *repeat],
            SamOps::Arrayval { reference, tensor: _ } => vec![*reference],
            SamOps::Join { ref1, ref2, crd1, crd2, tp: _ } => vec![*ref1, *ref2, *crd1, *crd2],
            SamOps::Reduce { inputs } => vec![*inputs],
            SamOps::ALU { op: _, inputs } => inputs.to_vec(),
            SamOps::CoordDrop { inner, outer } => vec![*inner, *outer],
            SamOps::Root => vec![],
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
        let read1 = scope.stage(SamOps::Fiberlookup { reference: root, tensor: "A".to_string(), level: 0 });
        let _ = scope.stage(SamOps::Fiberlookup { reference: read1[0], tensor: "A".to_string(), level: 1 });
        scope.print();
    }
}
