use std::collections::HashMap;

use shared::{AdtId, IdVec};

use crate::{ApiAdt, ApiConstant, ApiEntry, ApiFunction, ApiMethod, Intrinsic, NativeId, Registry};

pub struct Library<C> {
    natives: IdVec<NativeId, ApiEntry<C>>,
    intrinsics: HashMap<NativeId, Intrinsic>,
    adts: Vec<ApiAdt>,
    registry: Registry,
    /// Mimas source the library carries (`(name, text)`): the parts of a
    /// library written in mimas itself, which every program that installs
    /// the library compiles alongside its own files.
    sources: Vec<(&'static str, &'static str)>,
}

impl<C> Library<C> {
    pub fn new() -> Self {
        Self {
            natives: IdVec::new(),
            intrinsics: HashMap::new(),
            adts: Vec::new(),
            registry: Registry::new(),
            sources: Vec::new(),
        }
    }

    /// Ship `src` with the library: programs that install it compile it too.
    pub fn add_source(&mut self, name: &'static str, src: &'static str) {
        self.sources.push((name, src));
    }

    /// A hash of what the checker takes from this library — every native's
    /// name, module and shape, every adt's name and module, every shipped
    /// source — so a solver prepared against one library is reused only
    /// under a library that would prepare the same way.
    pub fn fingerprint(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        for adt in &self.adts {
            adt.name.hash(&mut h);
            adt.module.hash(&mut h);
            adt.variants.len().hash(&mut h);
        }
        for (id, entry) in self.natives.iter() {
            id.index().hash(&mut h);
            match entry {
                ApiEntry::Function(f) => {
                    0u8.hash(&mut h);
                    f.name.hash(&mut h);
                    f.module.hash(&mut h);
                    f.parameters.len().hash(&mut h);
                }
                ApiEntry::Method(m) => {
                    1u8.hash(&mut h);
                    m.name.hash(&mut h);
                    m.parameters.len().hash(&mut h);
                }
                ApiEntry::Constant(c) => {
                    2u8.hash(&mut h);
                    c.name.hash(&mut h);
                }
            }
        }
        for (name, src) in &self.sources {
            name.hash(&mut h);
            src.hash(&mut h);
        }
        self.intrinsics.len().hash(&mut h);
        h.finish()
    }

    pub fn sources(&self) -> &[(&'static str, &'static str)] {
        &self.sources
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
