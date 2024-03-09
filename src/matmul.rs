use std::rc::Rc;

use crate::{
    sam::SamOps,
    sym::{Expr, Sym},
    tensor::Tensor,
};

pub fn matmul(a: Tensor, b: Tensor) -> Tensor {
    let a_meta = a.meta.clone();
    // let b_meta = b.meta.clone();
    let meta0 = Rc::new(move |refstream, scope: &_| {
        // let root = SamOps::Root.stage(scope)[0];
        // let t0 = SamOps::Repeat {
        //     target: root,
        //     repeat: refstream,
        // }
        // .stage(scope)[0];
        let (r0, c0) = a_meta[0](refstream, scope);
        (r0, c0)
    });
    // let ref_binding = meta0.clone();

    let a_meta = a.meta.clone();
    let b_meta = b.meta.clone();
    let meta1 = move |refstream, scope: &_| {
        // let root = SamOps::Root.stage(scope)[0];
        // let t0 = SamOps::Repeat {
        //     target: ref_binding(refstream, scope).0,
        //     repeat: refstream,
        // }
        // .stage(scope)[0];
        //TODO: Might need to change back to repeat on root or motion repeat
        let (r2, c2) = b_meta[0](refstream, scope);
        (r2, c2)
    };

    let comp = move |refstreams: Vec<Sym>, scope: &_| {
        let root = SamOps::Root.stage(scope)[0];
        let t0 = SamOps::Repeat {
            target: root,
            repeat: refstreams[1],
        }
        .stage(scope)[0];

        //TODO: Might need to bring those back
        let (r0, _c0) = a.meta[0](vec![Some(t0)], scope);
        // let t1 = SamOps::Repeat {
        //     target: root,
        //     repeat: r0,
        // }
        // .stage(scope)[0];
        let (r2, _c2) = b.meta[0](vec![Some(r0)], scope);
        // let r3 = SamOps::Repeat {
        //     target: r0,
        //     // target: t1,
        //     repeat: r2,
        // }
        // .stage(scope)[0];

        // let (r4, c4) = a.meta[1](refstreams[1], scope);
        let (r4, c4) = a.meta[1](vec![Some(t0), Some(r2)], scope);
        // let (r5, c5) = b.meta[1](vec![None, Some(refstreams[1])], scope);
        // let (r5, c5) = b.meta[1](vec![Some(r0), None], scope);
        let (r5, c5) = b.meta[1](vec![Some(refstreams[1]), None], scope);

        // let gen = SamOps::Genref { coords: refstreams }
        let [ika, ikb, _icrd] = SamOps::Join {
            refs: vec![r4, r5],
            crds: vec![c4, c5],
            tp: crate::sam::JoinType::Intersect,
        }
        .stage(scope)[..] else {
            panic!()
        };
        let v_a = (a.comp)(vec![r0, ika], scope);
        let v_b = (b.comp)(vec![r0, ikb], scope);
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
        meta: vec![meta0, Rc::new(meta1)],
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
        let result = (output.comp)(vec![root, root], &scope);
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
        let result = (output.comp)(vec![root, root], &scope);
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
