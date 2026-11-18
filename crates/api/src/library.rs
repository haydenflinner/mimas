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
    /// Bounds of the caller's `install_fn` inside the install sequence:
    /// `install_into` wraps it in inventory-sweep phases whose contents
    /// depend on which crates the binary links, so a cross-binary
    /// fingerprint must cover only this span — `(natives start..end,
    /// adts start..end)`. `None` for a hand-built library.
    install_span: Option<(usize, usize, usize, usize)>,
}

impl<C> Library<C> {
    pub fn new() -> Self {
        Self {
            natives: IdVec::new(),
            intrinsics: HashMap::new(),
            adts: Vec::new(),
            registry: Registry::new(),
            sources: Vec::new(),
            install_span: None,
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
    ///
    /// All counts and indices hash as `u64`, never `usize`: `usize` encodes
    /// its native width into the hash stream, so a fingerprint computed on
    /// wasm32 would never equal the same table's fingerprint on a 64-bit
    /// host (that broke the wasm/native `mimas_pliron_nsig` handshake).
    pub fn fingerprint(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        for adt in &self.adts {
            adt.name.hash(&mut h);
            (adt.module.len() as u64).hash(&mut h);
            for m in &adt.module {
                m.hash(&mut h);
            }
            (adt.variants.len() as u64).hash(&mut h);
        }
        for (id, entry) in self.natives.iter() {
            (id.index() as u64).hash(&mut h);
            match entry {
                ApiEntry::Function(f) => {
                    0u8.hash(&mut h);
                    f.name.hash(&mut h);
                    (f.module.len() as u64).hash(&mut h);
                    for m in &f.module {
                        m.hash(&mut h);
                    }
                    (f.parameters.len() as u64).hash(&mut h);
                }
                ApiEntry::Method(m) => {
                    1u8.hash(&mut h);
                    m.name.hash(&mut h);
                    (m.parameters.len() as u64).hash(&mut h);
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
        (self.intrinsics.len() as u64).hash(&mut h);
        h.finish()
    }

    /// Record where the caller's `install_fn` landed in the install
    /// sequence: `install_into` calls this after running it, with the
    /// native/adt counts from before it started. Entries outside the span
    /// come from `inventory` sweeps and depend on which crates the binary
    /// links — two binaries sharing one install fn can differ there
    /// legitimately, so `fingerprint_install` excludes them.
    pub fn mark_install_span(&mut self, natives_before: usize, adts_before: usize) {
        self.install_span = Some((natives_before, self.natives.len(), adts_before, self.adts.len()));
    }

    /// Like `fingerprint`, but covers only what the caller's `install_fn`
    /// registered (plus shipped sources, which only installs add). This is
    /// the fingerprint two binaries must agree on when they share an
    /// install fn — e.g. the pliron-mimas driver and the wasm shim — even
    /// though their `inventory`-swept entries legitimately differ.
    /// Panics if `install_into` never ran (a hand-built library has no
    /// span to scope to).
    pub fn fingerprint_install(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let (n0, n1, a0, a1) =
            self.install_span.expect("fingerprint_install needs install_into's span");
        let mut h = std::collections::hash_map::DefaultHasher::new();
        for adt in &self.adts[a0..a1] {
            adt.name.hash(&mut h);
            (adt.module.len() as u64).hash(&mut h);
            for m in &adt.module {
                m.hash(&mut h);
            }
            (adt.variants.len() as u64).hash(&mut h);
        }
        for (id, entry) in self.natives.iter().skip(n0).take(n1 - n0) {
            // Absolute ids: emitted code bakes them, so a binary whose
            // swept prefix is a different length misdispatches — that
            // must fail the fingerprint, not pass it.
            (id.index() as u64).hash(&mut h);
            match entry {
                ApiEntry::Function(f) => {
                    0u8.hash(&mut h);
                    f.name.hash(&mut h);
                    (f.module.len() as u64).hash(&mut h);
                    for m in &f.module {
                        m.hash(&mut h);
                    }
                    (f.parameters.len() as u64).hash(&mut h);
                }
                ApiEntry::Method(m) => {
                    1u8.hash(&mut h);
                    m.name.hash(&mut h);
                    (m.parameters.len() as u64).hash(&mut h);
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
        (self
            .intrinsics
            .keys()
            .filter(|id| (n0..n1).contains(&id.index()))
            .count() as u64)
            .hash(&mut h);
        h.finish()
    }

    /// `id:name@module#arity` for every install-span entry, plus the span
    /// bounds — the two sides of a `mimas_pliron_nsig` mismatch dump this
    /// to find the first entry that differs.
    pub fn install_debug(&self) -> String {
        let (n0, n1, a0, a1) = self
            .install_span
            .expect("install_debug needs install_into's span");
        let mut s = format!(
            "span natives[{n0}..{n1}) adts[{a0}..{a1}) total_natives={} sources={}",
            self.natives.len(),
            self.sources.len()
        );
        for (id, entry) in self.natives.iter().skip(n0).take(n1 - n0) {
            use std::fmt::Write as _;
            let desc = match entry {
                ApiEntry::Function(f) => {
                    format!("fn {}@{}#{}", f.name, f.module.join("::"), f.parameters.len())
                }
                ApiEntry::Method(m) => {
                    format!("method {}#{}", m.name, m.parameters.len())
                }
                ApiEntry::Constant(c) => format!("const {}", c.name),
            };
            write!(s, "\n  {} {desc}", id.index()).unwrap();
        }
        s
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
