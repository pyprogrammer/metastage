use std::rc::Rc;

use crate::{sam::SamOps, sym::Expr, tensor::Tensor};

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
            ref1: r0,
            ref2: r1,
            crd1: c0,
            crd2: c1,
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
            ref1: r0,
            ref2: r1,
            crd1: c0,
            crd2: c1,
            tp: crate::sam::JoinType::Union,
        }
        .stage(scope)[..] else {
            panic!()
        };

        let (r2, c2) = a_meta[1](t1_0, scope);
        let (r3, c3) = b_meta[1](t2_0, scope);

        let icrd_1 = SamOps::Join {
            ref1: r2,
            ref2: r3,
            crd1: c2,
            crd2: c3,
            tp: crate::sam::JoinType::Union,
        }
        .stage(scope)[2];

        (SamOps::Genref { coords: icrd_1 }.stage(scope)[0], icrd_1)
    };

    let a_meta = a.meta.clone();
    let b_meta = b.meta.clone();
    let comp = move |refstream, scope: &_| {
        let root = SamOps::Root.stage(scope)[0];
        let t0 = SamOps::Repeat {
            target: root,
            repeat: refstream,
        }
        .stage(scope)[0];
        let (r0, c0) = a_meta[0](t0, scope);
        let (r1, c1) = b_meta[0](t0, scope);

        let [t1_0, t2_0, _icrd_0] = SamOps::Join {
            ref1: r0,
            ref2: r1,
            crd1: c0,
            crd2: c1,
            tp: crate::sam::JoinType::Union,
        }
        .stage(scope)[..] else {
            panic!()
        };

        let (r2, c2) = a_meta[1](t1_0, scope);
        let (r3, c3) = b_meta[1](t2_0, scope);

        let [t1_1, t2_1, _icrd_1] = SamOps::Join {
            ref1: r2,
            ref2: r3,
            crd1: c2,
            crd2: c3,
            tp: crate::sam::JoinType::Union,
        }
        .stage(scope)[..] else {
            panic!()
        };

        let v_a = (a.comp)(t1_1, scope);
        let v_b = (b.comp)(t2_1, scope);
        SamOps::ALU {
            op: crate::sam::PrimitiveOp::Add,
            inputs: vec![v_a, v_b],
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

    use crate::{
        matmul::matmul,
        sym::{Expr, ScopeRef},
        tensor::InputTensor,
    };

    use super::{matadd, SamOps};

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

        let t1 = matmul(tensor_a.stage(), tensor_b.stage());
        let t2 = matmul(tensor_c.stage(), tensor_d.stage());

        let output = matadd(t1, t2);

        let result = (output.comp)(root, &scope);
        println!("{result:?}");

        scope.borrow_mut().print();
    }
}