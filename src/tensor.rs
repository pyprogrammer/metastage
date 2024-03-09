use std::rc::Rc;

use crate::{
    sam::SamOps,
    sym::{Expr, ScopeRef, Sym},
};

pub struct Tensor {
    pub meta: Vec<Rc<dyn Fn(Vec<Option<Sym>>, &ScopeRef<SamOps>) -> (Sym, Sym)>>,
    pub comp: Rc<dyn Fn(Vec<Sym>, &ScopeRef<SamOps>) -> Sym>,
}

pub struct InputTensor {
    pub name: String,
    pub dims: usize,
}

impl InputTensor {
    pub fn stage(&self) -> Tensor {
        let mut meta: Vec<Rc<dyn Fn(Vec<Option<Sym>>, &ScopeRef<SamOps>) -> (Sym, Sym)>> = vec![];
        for level in 0..self.dims {
            let tensor = self.name.clone();
            let prev_meta = match level {
                0 => None,
                _ => Some(meta[level - 1].clone())
            };
            meta.push(Rc::new(move |mut refstream, scope: &ScopeRef<SamOps>| {
                let rep = refstream.pop().unwrap();
                let root = SamOps::Root.stage(scope)[0];
                let prev = match &prev_meta {
                    None => root,
                    Some(pm) => pm(refstream, scope).0,
                };
                let rep = match rep {
                    Some(r) => SamOps::Repeat{target : prev, repeat: r }.stage(scope)[0],
                    None => prev,
                };
                let tmp = SamOps::Fiberlookup {
                    reference: rep,
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
