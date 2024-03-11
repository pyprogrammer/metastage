use std::rc::Rc;

use crate::{
    sam::SamOps,
    sym::{Expr, Sym},
    tensor::Tensor,
};

pub fn matadd(a: Tensor, b: Tensor) -> Tensor {
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
        let (r1, c1) = b_meta[0](t0, scope);

        let icrd = SamOps::Join {
            refs: vec![r0, r1],
            crds: vec![c0, c1],
            tp: crate::sam::JoinType::Union,
        }
        .stage(scope)[2];

        (SamOps::Genref { coords: icrd }.stage(scope)[0], icrd)
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
        let (r0, c0) = a_meta[0](t0, scope);
        let (r1, c1) = b_meta[0](t0, scope);

        let [t1_0, t2_0, _icrd_0] = SamOps::Join {
            refs: vec![r0, r1],
            crds: vec![c0, c1],
            tp: crate::sam::JoinType::Union,
        }
        .stage(scope)[..] else {
            panic!()
        };

        let (r2, c2) = a_meta[1](t1_0, scope);
        let (r3, c3) = b_meta[1](t2_0, scope);

        let icrd_1 = SamOps::Join {
            refs: vec![r2, r3],
            crds: vec![c2, c3],
            tp: crate::sam::JoinType::Union,
        }
        .stage(scope)[2];

        (SamOps::Genref { coords: icrd_1 }.stage(scope)[0], icrd_1)
    };

    let a_meta = a.meta.clone();
    let b_meta = b.meta.clone();
    let comp = Rc::new(move |refstreams: Vec<Sym>, scope: &_| {
        let root = SamOps::Root.stage(scope)[0];
        let t0 = SamOps::Repeat {
            target: root,
            repeat: refstreams[0],
        }
        .stage(scope)[0];
        let (r0, c0) = a_meta[0](t0, scope);
        let (r1, c1) = b_meta[0](t0, scope);

        let [t1_0, t2_0, _icrd_0] = SamOps::Join {
            refs: vec![r0, r1],
            crds: vec![c0, c1],
            tp: crate::sam::JoinType::Union,
        }
        .stage(scope)[..] else {
            panic!()
        };

        let (r2, c2) = a_meta[1](t1_0, scope);
        let (r3, c3) = b_meta[1](t2_0, scope);

        let [t1_1, t2_1, _icrd_1] = SamOps::Join {
            refs: vec![r2, r3],
            crds: vec![c2, c3],
            tp: crate::sam::JoinType::Union,
        }
        .stage(scope)[..] else {
            panic!()
        };

        let v_a = (a.comp)(vec![r0, t1_1], scope);
        let v_b = (b.comp)(vec![r1, t2_1], scope);
        SamOps::ALU {
            op: crate::sam::PrimitiveOp::Add,
            inputs: vec![v_a, v_b],
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
        meta: vec![Rc::new(meta0), Rc::new(meta1)],
        comp,
        precomp: Rc::new(preprocess),
    }
}

#[cfg(test)]
mod test {
    use graphviz_rust::printer::{DotPrinter, PrinterContext};

    use crate::{
        matmul::matmul,
        sym::{Expr, ScopeRef},
        tensor::InputTensor,
    };

    use super::{matadd, SamOps};

    #[test]
    fn test_matadd() {
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

        let t1 = matadd(tensor_a.stage(), tensor_b.stage());

        let result = (t1.comp)(vec![root, root], &scope);
        let m0 = (t1.meta[0])(root, &scope);
        let m1 = (t1.meta[1])(root, &scope);
        println!("{result:?} {m0:?}, {m1:?}");

        let sc = scope.borrow_mut();
        sc.print();
        println!("{}", sc.to_dot().print(&mut PrinterContext::default()));
    }

    #[test]
    fn test_nested_ops() {
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
        let tensor_e = InputTensor {
            name: "E".to_string(),
            dims: 2,
        };

        let t1 = matmul(tensor_a.stage(), tensor_b.stage());
        let t2 = matmul(matmul(tensor_c.stage(), tensor_d.stage()), tensor_e.stage());

        let output = matadd(t1, t2);

        let result = (output.comp)(vec![root, root], &scope);
        let m0 = (output.meta[0])(root, &scope);
        let m1 = (output.meta[1])(root, &scope);
        println!("{result:?} {m0:?}, {m1:?}");

        let sc = scope.borrow_mut();
        sc.print();
        println!("{}", sc.to_dot().print(&mut PrinterContext::default()));
    }

    #[test]
    fn test_ab_plus_ac() {
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

        let t1 = matmul(tensor_a.stage(), tensor_b.stage());
        let t2 = matmul(tensor_a.stage(), tensor_c.stage());

        let output = matadd(t1, t2);

        let result = (output.comp)(vec![root, root], &scope);
        println!("{result:?}");

        let sc = scope.borrow_mut();
        sc.print();
        println!("{}", sc.to_dot().print(&mut PrinterContext::default()));
    }

    #[test]
    fn test_a_times_b_plus_c() {
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

        let b_plus_c = matadd(tensor_b.stage(), tensor_c.stage());

        let output = matmul(tensor_a.stage(), b_plus_c);

        let result = (output.comp)(vec![root, root], &scope);
        println!("{result:?}");

        let sc = scope.borrow_mut();
        sc.print();
        println!("{}", sc.to_dot().print(&mut PrinterContext::default()));
    }
}
