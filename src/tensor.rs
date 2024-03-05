use std::rc::Rc;

use crate::{
    sam::SamOps,
    sym::{Expr, ScopeRef, Sym},
};

pub struct Tensor {
    pub meta: Vec<Rc<dyn Fn(Sym, &ScopeRef<SamOps>) -> (Sym, Sym)>>,
    pub comp: Rc<dyn Fn(Vec<Sym>, &ScopeRef<SamOps>) -> Sym>,
}

pub struct InputTensor {
    pub name: String,
    pub dims: usize,
}

impl InputTensor {
    pub fn stage(&self) -> Tensor {
        let mut meta: Vec<Rc<dyn Fn(Sym, &ScopeRef<SamOps>) -> (Sym, Sym)>> = vec![];
        for level in 0..self.dims {
            let tensor = self.name.clone();
            meta.push(Rc::new(move |refstream, scope: &ScopeRef<SamOps>| {
                let tmp = SamOps::Fiberlookup {
                    reference: refstream,
                    tensor: tensor.clone(),
                    level,
                }
                .stage(scope);
                (tmp[0], tmp[1])
            }));
        }
        let tensor = self.name.clone();
        Tensor {
            meta,
            comp: Rc::new(move |refstreams, scope: &ScopeRef<SamOps>| {
                SamOps::Arrayval {
                    reference: refstreams[refstreams.len() - 1],
                    tensor: tensor.clone(),
                }
                .stage(scope)[0]
            }),
        }
    }
}
