use std::rc::Rc;

use crate::{
    sam::SamOps,
    sym::{Expr, Sym},
    tensor::Tensor,
};

pub fn matmul(a: Tensor, b: Tensor) -> Tensor {
    let a_meta = a.meta.clone();
    let meta0 = Rc::new(move |refstream, scope: &_| {
        let (r0, c0) = a_meta[0](refstream, scope);
        (r0, c0)
    });

    let b_meta = b.meta.clone();
    let meta1 = move |refstream, scope: &_| {
        //TODO: Might need to change back to repeat on root or motion repeat
        let (r2, c2) = b_meta[0](refstream, scope);
        (r2, c2)
    };

    let a_meta = a.meta.clone();
    let b_meta = b.meta.clone();
    let comp = Rc::new(move |refstreams: Vec<Sym>, scope: &_| {
        let root = SamOps::Root.stage(scope)[0];

        let prev_ref = refstreams[1];
        let (r2, c2) = a_meta[0](root, scope);
        let rep = SamOps::Repeat {
            target: r2,
            repeat: prev_ref,
        }
        .stage(scope)[0];
        let (r4, c4) = a_meta[1](rep, scope);
        let (r5, c5) = b_meta[1](prev_ref, scope);

        let [ika, ikb, _icrd] = SamOps::Join {
            refs: vec![r4, r5],
            crds: vec![c4, c5],
            tp: crate::sam::JoinType::Intersect,
        }
        .stage(scope)[..] else {
            panic!()
        };
        let v_a = (a.comp)(vec![r4, ika], scope);
        let v_b = (b.comp)(vec![r4, ikb], scope);
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
    });
    let comp_pipeline = comp.clone();
    let a_meta = a.meta.clone();
    let b_meta = b.meta.clone();

    let preprocess = move |refstream: Sym, scope: &_| {
        let root = SamOps::Root.stage(scope)[0];
        let t0 = SamOps::Repeat {
            target: root,
            repeat: refstream,
        }
        .stage(scope)[0];
        let (r0, _c0) = a_meta[0](t0, scope);
        let rep = SamOps::Repeat {
            target: root,
            repeat: r0,
        }
        .stage(scope)[0];
        let (r1, _c1) = b_meta[0](rep, scope);
        let val = comp_pipeline(vec![r0, r1], scope);
        val
    };
    Tensor {
        meta: vec![meta0, Rc::new(meta1)],
        comp,
        precomp: Rc::new(preprocess),
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
        let result = (output.precomp)(root, &scope);
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
        let result = (output.precomp)(root, &scope);
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
    fn test_matmul3() {
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
        let tensor_d = InputTensor {
            name: "D".to_string(),
            dims: 2,
        };
        let output = matmul(matmul(matmul(tensor_a.stage(), tensor_b.stage()), tensor_c.stage()), tensor_d.stage());
        let result = (output.precomp)(root, &scope);
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
