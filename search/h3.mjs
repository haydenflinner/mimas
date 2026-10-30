// h3.mjs — hyperbolic 3-space (hyperboloid model) + archive-tree layout.
//
// H³ is the upper sheet of ⟨p,p⟩_M = −1 in R⁴ under the Minkowski
// metric ⟨a,b⟩_M = −a₀b₀ + a₁b₁ + a₂b₂ + a₃b₃. Points are 4-vectors
// p₀>0; tangent vectors t at p satisfy ⟨t,p⟩=0 and are spacelike
// (⟨t,t⟩>0). Isometries are 4×4 Lorentz matrices stored row-major —
// a boost translates, an SO(3) block rotates. The viewer transforms
// every vertex by one isometry uniform then divides q.xyz/q.w (Klein
// — geodesics stay straight, so fly-through is exact) or q.xyz/(q.w+1)
// (Poincaré — conformal ball, edges need their sampled subdivisions).
//
// Layout: the archive is a rooted forest (node.parent). Root 0 sits at
// the origin, later roots fan onto a sphere around it (fuzz seeds).
// Each child grows a fixed geodesic step D from its parent, along a
// direction on a cone around the parent's forward axis — child 0 runs
// straight, later siblings spiral out at the golden angle — so the
// tree blooms outward like the beam/go-explore search actually did.

export const mdot = (a, b) =>
    -a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];

const s2 = (a, b, s) => [
    a[0] + s * b[0],
    a[1] + s * b[1],
    a[2] + s * b[2],
    a[3] + s * b[3],
];

/// Project an ambient vector onto the tangent space at p.
export const projectT = (p, x) => s2(x, p, mdot(x, p));

/// Spacelike unit.
export function normT(t) {
    const n = Math.sqrt(Math.max(1e-30, mdot(t, t)));
    return [t[0] / n, t[1] / n, t[2] / n, t[3] / n];
}

/// Hyperbolic distance between two points on the sheet.
export const dist = (a, b) => Math.acosh(Math.min(1e15, Math.max(1, -mdot(a, b))));

/// Point at arclength t along the geodesic from p with unit velocity v.
export function geo(p, v, t) {
    const c = Math.cosh(t), s = Math.sinh(t);
    return [
        c * p[0] + s * v[0],
        c * p[1] + s * v[1],
        c * p[2] + s * v[2],
        c * p[3] + s * v[3],
    ];
}

/// Unit velocity at arclength t along that same geodesic.
export function geoVel(p, v, t) {
    const c = Math.cosh(t), s = Math.sinh(t);
    return [
        s * p[0] + c * v[0],
        s * p[1] + c * v[1],
        s * p[2] + c * v[2],
        s * p[3] + c * v[3],
    ];
}

/// Point at u∈[0,1] on the geodesic between sheet points a and b.
export function geoAB(a, b, u) {
    const d = dist(a, b);
    if (d < 1e-9) return a.slice();
    const s = Math.sinh(d);
    const A = Math.sinh(d * (1 - u)) / s;
    const B = Math.sinh(d * u) / s;
    return [
        A * a[0] + B * b[0],
        A * a[1] + B * b[1],
        A * a[2] + B * b[2],
        A * a[3] + B * b[3],
    ];
}

// ---- 4×4 Lorentz matrices (row-major) --------------------------------------

export const I4 = () => [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

export function mmul(A, B) {
    const o = new Array(16);
    for (let i = 0; i < 4; i++)
        for (let j = 0; j < 4; j++) {
            let s = 0;
            for (let k = 0; k < 4; k++) s += A[i * 4 + k] * B[k * 4 + j];
            o[i * 4 + j] = s;
        }
    return o;
}

export const mapply = (M, p) => [
    M[0] * p[0] + M[1] * p[1] + M[2] * p[2] + M[3] * p[3],
    M[4] * p[0] + M[5] * p[1] + M[6] * p[2] + M[7] * p[3],
    M[8] * p[0] + M[9] * p[1] + M[10] * p[2] + M[11] * p[3],
    M[12] * p[0] + M[13] * p[1] + M[14] * p[2] + M[15] * p[3],
];

/// Inverse of a Lorentz matrix: M⁻¹ = η Mᵀ η, η = diag(−1,1,1,1).
export function minv(M) {
    const s = [-1, 1, 1, 1];
    const T = new Array(16);
    for (let i = 0; i < 4; i++)
        for (let j = 0; j < 4; j++) T[i * 4 + j] = s[i] * s[j] * M[j * 4 + i];
    return T;
}

/// Boost: translate the origin along unit spatial direction (nx,ny,nz)
/// by rapidity r. Symmetric Lorentz matrix — its own transpose.
export function boost(nx, ny, nz, r) {
    const c = Math.cosh(r), s = Math.sinh(r), k = c - 1;
    return [
        c, s * nx, s * ny, s * nz,
        s * nx, 1 + k * nx * nx, k * nx * ny, k * nx * nz,
        s * ny, k * nx * ny, 1 + k * ny * ny, k * ny * nz,
        s * nz, k * nx * nz, k * ny * nz, 1 + k * nz * nz,
    ];
}

/// Isometry taking the origin e₀ to sheet point p (no rotation).
export function translateTo(p) {
    const w = Math.hypot(p[1], p[2], p[3]);
    if (w < 1e-12) return I4();
    const d = Math.acosh(Math.max(1, p[0]));
    return boost(p[1] / w, p[2] / w, p[3] / w, d);
}

/// Isometry taking sheet point p back to the origin — the recenter map.
export const translateFrom = (p) => minv(translateTo(p));

/// Spatial rotation about unit axis (ax,ay,az) by angle θ (Rodrigues),
/// sitting in the spatial block — preserves the time axis.
export function rot(ax, ay, az, th) {
    const c = Math.cos(th), s = Math.sin(th), k = 1 - c;
    const x = ax, y = ay, z = az;
    return [
        1, 0, 0, 0,
        0, c + k * x * x, k * x * y - s * z, k * x * z + s * y,
        0, k * x * y + s * z, c + k * y * y, k * y * z - s * x,
        0, k * x * z - s * y, k * y * z + s * x, c + k * z * z,
    ];
}

/// Tangent at a pointing toward b (unit). b may be a point on the sheet.
export function toward(a, b) {
    return normT(projectT(a, b));
}

// ---- tree layout ------------------------------------------------------------

const GOLD = 2.39996322972865332; // golden angle

/// Orthonormal pair {u,w} spanning the plane ⊥ fwd in p's tangent space.
/// `seed` rotates the basis choice so sibling subtrees don't all bloom
/// in the same ambient plane.
function ortho(p, fwd, seed) {
    const refs = [
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ];
    const out = [];
    for (let j = 0; j < 3 && out.length < 2; j++) {
        let t = projectT(p, refs[(seed + j) % 3]);
        t = s2(t, fwd, -mdot(t, fwd));
        for (const o of out) t = s2(t, o, -mdot(t, o));
        if (mdot(t, t) > 1e-8) out.push(normT(t));
    }
    while (out.length < 2) out.push(out[0] ? [-out[0][3], out[0][2], -out[0][1], out[0][0]] : [0, 1, 0, 0]);
    return out;
}

/// Fibonacci-sphere direction k (for scattering multiple roots / seeds).
export function fibDir(k) {
    const z = 1 - 2 * ((k * 0.6180339887498949) % 1);
    const r = Math.sqrt(Math.max(0, 1 - z * z));
    const a = k * GOLD;
    return [r * Math.cos(a), r * Math.sin(a), z];
}

/// Pull p back onto the sheet — geodesic chains drift off |⟨p,p⟩|=1
/// by cancellation once coords pass ~1e8.
export function onSheet(p) {
    const n = Math.sqrt(Math.max(1e-30, -mdot(p, p)));
    return [p[0] / n, p[1] / n, p[2] / n, p[3] / n];
}

/// Lay out the archive forest. `nodes[i].parent` is the archive index
/// this state expanded from (−1/undefined for seeds). Returns
/// { pos, fwd } per node — pos on the sheet, fwd the unit tangent the
/// subtree keeps growing along. The step decays with depth
/// (edge·decay^d): fuzz lanes grow tapes ~600 deep, and a fixed step
/// would push coordinates past e^400 — the decay keeps the whole tree
/// inside ~e^21 where float math (and the GPU's f32) still resolves it,
/// and it's how the classic H3 layouts packed levels anyway.
export function layout(nodes, opts = {}) {
    const D = opts.edge ?? 0.72; // geodesic step parent → child
    const DECAY = opts.decay ?? 0.955; // per-depth shrink
    const CONE = opts.cone ?? 1.15; // sibling spread (radians at ~5 kids)
    const R0 = opts.rootR ?? 1.9; // radius extra roots scatter to
    const out = new Array(nodes.length);
    const kids = new Map();
    const depth = new Int32Array(nodes.length);
    let roots = 0;
    for (let i = 0; i < nodes.length; i++) {
        const p = nodes[i].parent;
        if (p === undefined || p === null || p < 0 || !out[p]) {
            const k = roots++;
            if (k === 0) {
                out[i] = { pos: [1, 0, 0, 0], fwd: [0, 1, 0, 0] };
            } else {
                const d = fibDir(k - 1);
                const v = [0, d[0], d[1], d[2]];
                out[i] = { pos: geo([1, 0, 0, 0], v, R0), fwd: geoVel([1, 0, 0, 0], v, R0) };
            }
            continue;
        }
        depth[i] = depth[p] + 1;
        const k = kids.get(p) ?? 0;
        kids.set(p, k + 1);
        const { pos: P, fwd: F } = out[p];
        let v;
        if (k === 0) {
            v = F;
        } else {
            const [u, w] = ortho(P, F, i);
            const phi = k * GOLD;
            const th = Math.min(2.05, CONE * Math.sqrt(k / 5));
            const ct = Math.cos(th), st = Math.sin(th);
            const cp = Math.cos(phi), sp = Math.sin(phi);
            v = [
                ct * F[0] + st * (cp * u[0] + sp * w[0]),
                ct * F[1] + st * (cp * u[1] + sp * w[1]),
                ct * F[2] + st * (cp * u[2] + sp * w[2]),
                ct * F[3] + st * (cp * u[3] + sp * w[3]),
            ];
        }
        const d = D * Math.pow(DECAY, depth[i]);
        out[i] = {
            pos: onSheet(geo(P, v, d)),
            fwd: normT(projectT(geo(P, v, d), geoVel(P, v, d))),
        };
    }
    return out;
}
