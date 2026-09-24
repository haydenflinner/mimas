use std::collections::HashMap;

use shared::{AdtId, IdVec};

use crate::{ApiAdt, ApiConstant, ApiEntry, ApiFunction, ApiMethod, Intrinsic, NativeId, Registry};

pub struct Library<C> {
    natives: IdVec<NativeId, ApiEntry<C>>,
    intrinsics: HashMap<NativeId, Intrinsic>,
    adts: Vec<ApiAdt>,
    registry: Registry,
}

impl<C> Library<C> {
    pub fn new() -> Self {
        Self {
            natives: IdVec::new(),
            intrinsics: HashMap::new(),
            adts: Vec::new(),
            registry: Registry::new(),
        }
    }

    pub fn natives(&self) -> impl Iterator<Item = (NativeId, &ApiEntry<C>)> {
        self.natives.iter()
    }

    pub fn adts(&self) -> &[ApiAdt] {
        &self.adts
    }

    pub fn registry(&self) -> &Registry {
        &self.registry
    }

    pub fn registry_mut(&mut self) -> &mut Registry {
        &mut self.registry
    }

    pub fn into_registry(self) -> Registry {
        self.registry
    }

    pub fn push_adt(&mut self, adt: ApiAdt) {
        self.adts.push(adt);
    }

    /// `adt_id`s are globally allocated (builtins own the low ids), so the
    /// vec position isn't the id — scan.
    pub fn adt_mut(&mut self, id: AdtId) -> Option<&mut ApiAdt> {
        self.adts.iter_mut().find(|a| a.adt_id == id)
    }

    pub fn function(&mut self, f: ApiFunction<C>) -> NativeId {
        self.natives.push(ApiEntry::Function(f))
    }

    pub fn method(&mut self, m: ApiMethod<C>) -> NativeId {
        self.natives.push(ApiEntry::Method(m))
    }

    pub fn constant(&mut self, c: ApiConstant) -> NativeId {
        self.natives.push(ApiEntry::Constant(c))
    }

    pub fn mark_instrinsic(&mut self, nid: NativeId, i: Intrinsic) {
        self.intrinsics.insert(nid, i);
    }

    pub fn intrinsics(&self) -> &HashMap<NativeId, Intrinsic> {
        &self.intrinsics
    }
}

impl<C> Default for Library<C> {
    fn default() -> Self {
        Self::new()
    }
}
