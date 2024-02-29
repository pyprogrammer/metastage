use std::rc::Rc;

use crate::{sam::SamOps, sym::Expr, tensor::Tensor};

pub fn matmul(a: Tensor, b: Tensor) -> Tensor {
    let a_meta = a.meta.clone();
    let b_meta = b.meta.clone();
    let meta0 = move |refstream, scope: &_| {
        let root = SamOps::Root.stage(scope)[0];
        let t0 = SamOps::Repeat {
            target: root,
            repeat: refstream,
        }
        .stage(scope)[0];
        let (r0, c0) = a_meta[0](t0, scope);
        let r1 = SamOps::Repeat {
            target: root,
            repeat: r0,
        }
        .stage(scope)[0];
        let (r2, c2) = b_meta[0](r1, scope);
        let r3 = SamOps::Repeat {
            target: r0,
            repeat: r2,
        }
        .stage(scope)[0];
        let (r4, c4) = a_meta[1](r3, scope);
        let (r5, c5) = b_meta[1](r2, scope);
        let icrd = SamOps::Join {
            ref1: r4,
            ref2: r5,
            crd1: c4,
            crd2: c5,
            tp: crate::sam::JoinType::Intersect,
        }
        .stage(scope)[2];
        let jk = SamOps::CoordDrop {
            inner: icrd,
            outer: c2,
        }
        .stage(scope)[0];
        let ijk = SamOps::CoordDrop {
            inner: jk,
            outer: c0,
        }
        .stage(scope)[0];
        (SamOps::Genref { coords: ijk }.stage(scope)[0], ijk)
    };

    let a_meta = a.meta.clone();
    let b_meta = b.meta.clone();
    let meta1 = move |refstream, scope: &_| {
        let root = SamOps::Root.stage(scope)[0];
        let t0 = SamOps::Repeat {
            target: root,
            repeat: refstream,
        }
        .stage(scope)[0];
        let (r0, _c0) = a_meta[0](t0, scope);
        let r1 = SamOps::Repeat {
            target: root,
            repeat: r0,
        }
        .stage(scope)[0];
        let (r2, c2) = b_meta[0](r1, scope);
        let r3 = SamOps::Repeat {
            target: r0,
            repeat: r2,
        }
        .stage(scope)[0];
        let (r4, c4) = a_meta[1](r3, scope);
        let (r5, c5) = b_meta[1](r2, scope);
        let icrd = SamOps::Join {
            ref1: r4,
            ref2: r5,
            crd1: c4,
            crd2: c5,
            tp: crate::sam::JoinType::Intersect,
        }
        .stage(scope)[2];
        let jk = SamOps::CoordDrop {
            inner: icrd,
            outer: c2,
        }
        .stage(scope)[0];
        (SamOps::Genref { coords: jk }.stage(scope)[0], jk)
    };

    let comp = move |refstream, scope: &_| {
        let root = SamOps::Root.stage(scope)[0];
        let t0 = SamOps::Repeat {
            target: root,
            repeat: refstream,
        }
        .stage(scope)[0];
        let (r0, _c0) = a.meta[0](t0, scope);
        let r1 = SamOps::Repeat {
            target: root,
            repeat: r0,
        }
        .stage(scope)[0];
        let (r2, _c2) = b.meta[0](r1, scope);
        let r3 = SamOps::Repeat {
            target: r0,
            repeat: r2,
        }
        .stage(scope)[0];
        let (r4, c4) = a.meta[1](r3, scope);
        let (r5, c5) = b.meta[1](r2, scope);
        let [ika, ikb, _icrd] = SamOps::Join {
            ref1: r4,
            ref2: r5,
            crd1: c4,
            crd2: c5,
            tp: crate::sam::JoinType::Intersect,
        }
        .stage(scope)[..] else {
            panic!()
        };
        let v_a = (a.comp)(ika, scope);
        let v_b = (b.comp)(ikb, scope);
        let mul = SamOps::ALU {
            op: crate::sam::PrimitiveOp::Mul,
            inputs: vec![v_a, v_b],
        }
        .stage(scope)[0];
        SamOps::Reduce {
            inputs: mul,
            op: crate::sam::PrimitiveOp::Add,
        }
        .stage(scope)[0]
    };
    Tensor {
        meta: vec![Rc::new(meta0), Rc::new(meta1)],
        comp: Rc::new(comp),
    }
}

#[cfg(test)]
mod test {

    use graphviz_rust::printer::{DotPrinter, PrinterContext};

    use crate::{
        sym::{Expr, ScopeRef},
        tensor::InputTensor,
    };

    use super::{matmul, SamOps};

    #[test]
    fn test_matmul() {
        let scope = ScopeRef::<SamOps>::default();
        let root = SamOps::Root.stage(&scope)[0];
        let tensor_a = InputTensor {
            name: "A".to_string(),
            dims: 2,
        };
        let tensor_b = InputTensor {
            name: "B".to_string(),
            dims: 2,
        };
        let output = matmul(tensor_a.stage(), tensor_b.stage());
        let result = (output.comp)(root, &scope);
        println!("{result:?}");

        scope.borrow_mut().print();
        println!(
            "{}",
            scope
                .borrow()
                .to_dot()
                .print(&mut PrinterContext::default())
        );
    }

    #[test]
    fn test_matmul2() {
        let scope = ScopeRef::<SamOps>::default();
        let root = SamOps::Root.stage(&scope)[0];
        let tensor_a = InputTensor {
            name: "A".to_string(),
            dims: 2,
        };
        let tensor_b = InputTensor {
            name: "B".to_string(),
            dims: 2,
        };
        let tensor_c = InputTensor {
            name: "C".to_string(),
            dims: 2,
        };
        let output = matmul(matmul(tensor_a.stage(), tensor_b.stage()), tensor_c.stage());
        let result = (output.comp)(root, &scope);
        println!("{result:?}");

        scope.borrow_mut().print();
        println!(
            "{}",
            scope
                .borrow()
                .to_dot()
                .print(&mut PrinterContext::default())
        );
    }
}
