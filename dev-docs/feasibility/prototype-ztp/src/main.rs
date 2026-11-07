//! File: main.rs. Purpose: feasibility prototype for chunked lossless tensor decode into a pre-allocated buffer.
//!
//! Throwaway measurement tool. Subcommands:
//!   compress <in> <off> <len> <out> <chunk> <layout> <spec> <threads>
//!   decode <file> <threads> <method> [pretouch]
//! layout: "bytes:W" byte planes of width W, "raw:W" one stream, "join:W" BG single stream,
//!         "q8" Q8_0 (d_lo,d_hi,qs), "q4k".. field lists (see parse_layout).
//! spec: comma list of per-stream codecs, stream 0 = lowest offset, last repeats: store|lz4|z<level> .
//! Methods: scalar (scratch+scalar interleave), simd (scratch+AVX2), stride (plane passes with stride, stored planes
//! read in place), joint (stored planes in place, compressed via scratch, one AVX2 pass), inplace (single compressed
//! plane decoded into the tail of the destination chunk, forward in-place merge, zero scratch).
use rayon::prelude::*;
use std::error::Error;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::time::Instant;

type Res<T> = Result<T, Box<dyn Error + Send + Sync>>;

/// One field of a repeated block: byte offset and byte length.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Field { off: u32, len: u32 }

#[derive(Clone, Copy, Debug, PartialEq)]
enum Codec { Store, Lz4, Zstd(i32), NoLz }

thread_local! {
    /// Per-thread compression contexts, so context setup is not billed to every chunk.
    static CCTX: std::cell::RefCell<std::collections::HashMap<i32, zstd::bulk::Compressor<'static>>> = Default::default();
}

/// Compresses with a cached context. Level i32::MIN selects entropy-only parameters: an effectively disabled match
/// finder, so all bytes stay literals and go through zstd's Huffman stage.
fn zstd_cached(level: i32, data: &[u8]) -> Res<Vec<u8>> {
    use zstd::zstd_safe::{CParameter, Strategy};
    CCTX.with(|m| {
        let mut m = m.borrow_mut();
        if !m.contains_key(&level) {
            let mut c = zstd::bulk::Compressor::new(if level == i32::MIN { 1 } else { level })?;
            if level == i32::MIN {
                c.set_parameter(CParameter::WindowLog(17))?;
                c.set_parameter(CParameter::HashLog(6))?;
                c.set_parameter(CParameter::ChainLog(6))?;
                c.set_parameter(CParameter::SearchLog(1))?;
                c.set_parameter(CParameter::MinMatch(7))?;
                c.set_parameter(CParameter::TargetLength(0))?;
                c.set_parameter(CParameter::Strategy(Strategy::ZSTD_fast))?;
            }
            m.insert(level, c);
        }
        let c = m.get_mut(&level).ok_or("no ctx")?;
        Ok(c.compress(data)?)
    })
}

/// Parses a layout name to (block bytes, fields, joined single stream flag).
fn parse_layout(s: &str) -> Res<(u32, Vec<Field>, bool)> {
    let f = |v: &[(u32, u32)]| v.iter().map(|&(off, len)| Field { off, len }).collect::<Vec<_>>();
    if let Some(w) = s.strip_prefix("bytes:") {
        let w: u32 = w.parse()?;
        return Ok((w, (0..w).map(|off| Field { off, len: 1 }).collect(), false));
    }
    if let Some(w) = s.strip_prefix("join:") {
        let w: u32 = w.parse()?;
        return Ok((w, (0..w).map(|off| Field { off, len: 1 }).collect(), true));
    }
    if let Some(w) = s.strip_prefix("raw:") {
        let w: u32 = w.parse()?;
        return Ok((w, vec![Field { off: 0, len: w }], false));
    }
    Ok(match s {
        "q8" => (34, f(&[(0, 1), (1, 1), (2, 32)]), false),
        "q4k" => (144, f(&[(0, 1), (1, 1), (2, 1), (3, 1), (4, 12), (16, 128)]), false),
        "q5k" => (176, f(&[(0, 1), (1, 1), (2, 1), (3, 1), (4, 12), (16, 32), (48, 128)]), false),
        "q6k" => (210, f(&[(0, 128), (128, 64), (192, 16), (208, 1), (209, 1)]), false),
        // rotl1 field alignment: exponent byte + (mantissa<<1|sign); byte planes after the rotation.
        "rot:2" => (2, f(&[(0, 1), (1, 1)]), false),
        "rot:4" => (4, f(&[(0, 1), (1, 1), (2, 1), (3, 1)]), false),
        other => return Err(format!("unknown layout {other}").into()),
    })
}

fn parse_codec(s: &str) -> Res<Codec> {
    Ok(match s {
        "store" => Codec::Store,
        "lz4" => Codec::Lz4,
        "h" => Codec::NoLz,
        z if z.starts_with('z') => Codec::Zstd(z[1..].parse()?),
        o => return Err(format!("bad codec {o}").into()),
    })
}

/// Extracts field `fi` of every block of `src` into `dst`.
fn gather(src: &[u8], block: usize, f: Field, dst: &mut Vec<u8>) {
    dst.clear();
    let (o, l) = (f.off as usize, f.len as usize);
    dst.reserve(src.len() / block * l);
    for b in src.chunks_exact(block) { dst.extend_from_slice(&b[o..o + l]); }
}

struct Stream { codec: u8, raw_len: u32, comp_len: u32, offset: u64 }

fn compress_one(c: Codec, data: &[u8]) -> Res<(u8, Vec<u8>)> {
    let out = match c {
        Codec::Store => return Ok((0, data.to_vec())),
        Codec::Lz4 => lz4_flex::block::compress(data),
        Codec::Zstd(l) => zstd_cached(l, data)?,
        Codec::NoLz => zstd_cached(i32::MIN, data)?,
    };
    // Store when the codec gains under 0.5%: stored planes avoid decode work entirely.
    if out.len() as u64 * 1000 >= data.len() as u64 * 995 { return Ok((0, data.to_vec())); }
    Ok((match c { Codec::Lz4 => 1, _ => 2 }, out))
}

/// compress: reads [off, off+len) of `input`, writes the prototype container.
fn cmd_compress(a: &[String]) -> Res<()> {
    let (input, off, len, out, chunk) = (&a[0], a[1].parse::<u64>()?, a[2].parse::<usize>()?, &a[3], a[4].parse::<usize>()?);
    let (block, fields, joined) = parse_layout(&a[5])?;
    let specs: Vec<Codec> = a[6].split(',').map(parse_codec).collect::<Res<_>>()?;
    let threads: usize = a[7].parse()?;
    let bs = block as usize;
    let chunk = chunk / bs * bs;
    let len = len / bs * bs;
    let mut buf = vec![0u8; len];
    let mut f = File::open(input)?;
    f.seek(SeekFrom::Start(off))?;
    f.read_exact(&mut buf)?;
    let hash = xxhash_rust::xxh3::xxh3_64(&buf);
    let rot = a[5].starts_with("rot:");
    let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build()?;
    let nf = fields.len();
    let t = Instant::now();
    if rot { pool.install(|| buf.par_chunks_mut(1 << 20).for_each(|c| rot_all(c, bs, true))); }
    let parts: Vec<Vec<(u8, u32, Vec<u8>)>> = pool.install(|| {
        buf.par_chunks(chunk).map(|c| -> Res<_> {
            let mut res = Vec::new();
            if joined {
                let mut all = Vec::with_capacity(c.len());
                let mut p = Vec::new();
                for f in &fields { gather(c, bs, *f, &mut p); all.extend_from_slice(&p); }
                let (cd, v) = compress_one(specs[0], &all)?;
                res.push((cd, all.len() as u32, v));
            } else {
                let mut p = Vec::new();
                for (i, f) in fields.iter().enumerate() {
                    let sp = specs[i.min(specs.len() - 1)];
                    gather(c, bs, *f, &mut p);
                    let (cd, v) = compress_one(sp, &p)?;
                    res.push((cd, p.len() as u32, v));
                }
            }
            Ok(res)
        }).collect::<Res<Vec<_>>>()
    })?;
    let secs = t.elapsed().as_secs_f64();
    let mut w = std::io::BufWriter::new(File::create(out)?);
    let nstreams = if joined { 1 } else { nf };
    let comp: u64 = parts.iter().flatten().map(|s| s.2.len() as u64).sum();
    let hdr = format!("{} {} {} {} {} {} {} {}\n", len, chunk, bs, parts.len(), nstreams, joined as u8, hash, a[5]);
    w.write_all(format!("{:<256}", hdr.trim_end()).as_bytes())?;
    for c in &parts { for s in c { w.write_all(&[s.0])?; w.write_all(&s.1.to_le_bytes())?; w.write_all(&(s.2.len() as u32).to_le_bytes())?; } }
    for c in &parts { for s in c { w.write_all(&s.2)?; } }
    w.flush()?;
    let table = parts.len() * nstreams * 9;
    let per: Vec<u64> = (0..nstreams).map(|i| parts.iter().map(|c| c[i].2.len() as u64).sum()).collect();
    let stored: Vec<usize> = (0..nstreams).map(|i| parts.iter().filter(|c| c[i].0 == 0).count()).collect();
    println!("compress in={} layout={} spec={} chunk={} threads={} raw={} comp={} table={} saved={:.3}% speed={:.3}GB/s per_stream={:?} stored_chunks={:?}/{}",
        input.rsplit('/').next().unwrap_or(""), a[5], a[6], chunk, threads, len, comp, table, 100.0 * (1.0 - (comp + table as u64) as f64 / len as f64), len as f64 / secs / 1e9, per, stored, parts.len());
    Ok(())
}

/// Rotates every little-endian element left (`left`=true) or right by one bit, in place. AVX2 bulk, scalar tail.
fn rot_all(d: &mut [u8], w: usize, left: bool) {
    #[target_feature(enable = "avx2")]
    unsafe fn bulk(p: *mut u8, n: usize, w: usize, left: bool) -> usize {
        use std::arch::x86_64::*;
        let mut i = 0;
        while i + 32 <= n {
            let x = _mm256_loadu_si256(p.add(i) as *const __m256i);
            let y = match (w, left) {
                (2, true) => _mm256_or_si256(_mm256_slli_epi16(x, 1), _mm256_srli_epi16(x, 15)),
                (2, false) => _mm256_or_si256(_mm256_srli_epi16(x, 1), _mm256_slli_epi16(x, 15)),
                (4, true) => _mm256_or_si256(_mm256_slli_epi32(x, 1), _mm256_srli_epi32(x, 31)),
                _ => _mm256_or_si256(_mm256_srli_epi32(x, 1), _mm256_slli_epi32(x, 31)),
            };
            _mm256_storeu_si256(p.add(i) as *mut __m256i, y);
            i += 32;
        }
        i
    }
    // SAFETY: AVX2 is required by both subcommands' callers on this machine; pointer range is d.
    let done = unsafe { bulk(d.as_mut_ptr(), d.len(), w, left) };
    for e in d[done..].chunks_exact_mut(w) {
        if w == 2 {
            let v = u16::from_le_bytes([e[0], e[1]]);
            e.copy_from_slice(&(if left { v.rotate_left(1) } else { v.rotate_right(1) }).to_le_bytes());
        } else {
            let v = u32::from_le_bytes([e[0], e[1], e[2], e[3]]);
            e.copy_from_slice(&(if left { v.rotate_left(1) } else { v.rotate_right(1) }).to_le_bytes());
        }
    }
}

/// Reads VmRSS/VmHWM in bytes.
fn mem() -> Res<(u64, u64)> {
    let s = std::fs::read_to_string("/proc/self/status")?;
    let get = |k: &str| -> Res<u64> {
        let l = s.lines().find(|l| l.starts_with(k)).ok_or("no field")?;
        Ok(l.split_whitespace().nth(1).ok_or("bad")?.parse::<u64>()? * 1024)
    };
    Ok((get("VmRSS:")?, get("VmHWM:")?))
}

fn dec_stream(codec: u8, src: &[u8], dst: &mut [u8], z: &mut zstd::bulk::Decompressor<'static>) -> Res<()> {
    match codec {
        0 => dst.copy_from_slice(src),
        1 => { let n = lz4_flex::block::decompress_into(src, dst).map_err(|e| e.to_string())?; if n != dst.len() { return Err("lz4 short".into()); } }
        _ => { let n = z.decompress_to_buffer(src, dst)?; if n != dst.len() { return Err("zstd short".into()); } }
    }
    Ok(())
}

/// Scalar interleave of byte planes `s` (all len n) into dst (len n*w).
fn weave_scalar(s: &[&[u8]], dst: &mut [u8]) {
    let w = s.len();
    match w {
        2 => { let (a, b) = (s[0], s[1]); for (i, d) in dst.chunks_exact_mut(2).enumerate() { d[0] = a[i]; d[1] = b[i]; } }
        4 => { let (a, b, c, e) = (s[0], s[1], s[2], s[3]); for (i, d) in dst.chunks_exact_mut(4).enumerate() { d[0] = a[i]; d[1] = b[i]; d[2] = c[i]; d[3] = e[i]; } }
        _ => for (i, d) in dst.chunks_exact_mut(w).enumerate() { for (k, x) in d.iter_mut().enumerate() { *x = s[k][i]; } },
    }
}

/// AVX2 interleave for 2 planes. Pointers may overlap as in the in-place merge: every 64-byte store lands strictly
/// below the next unread position of `b` by the caller's guarantee. Returns elements done.
///
/// # Safety
/// Requires AVX2; a and b readable for n bytes, d writable for 2n bytes.
#[target_feature(enable = "avx2")]
unsafe fn weave2_avx2(a: *const u8, b: *const u8, d: *mut u8, n: usize) -> usize {
    use std::arch::x86_64::*;
    let mut i = 0;
    while i + 32 <= n {
        let x = _mm256_loadu_si256(a.add(i) as *const __m256i);
        let y = _mm256_loadu_si256(b.add(i) as *const __m256i);
        let lo = _mm256_unpacklo_epi8(x, y);
        let hi = _mm256_unpackhi_epi8(x, y);
        _mm256_storeu_si256(d.add(2 * i) as *mut __m256i, _mm256_permute2x128_si256(lo, hi, 0x20));
        _mm256_storeu_si256(d.add(2 * i + 32) as *mut __m256i, _mm256_permute2x128_si256(lo, hi, 0x31));
        i += 32;
    }
    i
}

/// AVX2 interleave for 4 planes; returns elements done.
///
/// # Safety
/// Requires AVX2; each source readable for n bytes, d writable for 4n bytes.
#[target_feature(enable = "avx2")]
unsafe fn weave4_avx2(p: [*const u8; 4], d: *mut u8, n: usize) -> usize {
    use std::arch::x86_64::*;
    let mut i = 0;
    while i + 32 <= n {
        let l = |k: usize| _mm256_loadu_si256(p[k].add(i) as *const __m256i);
        let (a, b, c, e) = (l(0), l(1), l(2), l(3));
        let ab_l = _mm256_unpacklo_epi8(a, b);
        let ab_h = _mm256_unpackhi_epi8(a, b);
        let ce_l = _mm256_unpacklo_epi8(c, e);
        let ce_h = _mm256_unpackhi_epi8(c, e);
        let v0 = _mm256_unpacklo_epi16(ab_l, ce_l);
        let v1 = _mm256_unpackhi_epi16(ab_l, ce_l);
        let v2 = _mm256_unpacklo_epi16(ab_h, ce_h);
        let v3 = _mm256_unpackhi_epi16(ab_h, ce_h);
        let o = d.add(4 * i) as *mut __m256i;
        _mm256_storeu_si256(o, _mm256_permute2x128_si256(v0, v1, 0x20));
        _mm256_storeu_si256(o.add(1), _mm256_permute2x128_si256(v2, v3, 0x20));
        _mm256_storeu_si256(o.add(2), _mm256_permute2x128_si256(v0, v1, 0x31));
        _mm256_storeu_si256(o.add(3), _mm256_permute2x128_si256(v2, v3, 0x31));
        i += 32;
    }
    i
}

fn weave_simd(s: &[&[u8]], dst: &mut [u8]) {
    let n = s[0].len();
    let w = s.len();
    let done = match w {
        // SAFETY: AVX2 checked at decode start; lengths asserted by construction (planes n, dst n*w).
        2 => unsafe { weave2_avx2(s[0].as_ptr(), s[1].as_ptr(), dst.as_mut_ptr(), n) },
        4 => unsafe { weave4_avx2([s[0].as_ptr(), s[1].as_ptr(), s[2].as_ptr(), s[3].as_ptr()], dst.as_mut_ptr(), n) },
        _ => 0,
    };
    if done < n {
        let tails: Vec<&[u8]> = s.iter().map(|p| &p[done..]).collect();
        weave_scalar(&tails, &mut dst[done * w..]);
    }
}

/// In-place merge: compressed plane `ci` was decoded to the last n bytes of dst; other planes come from `src`.
fn merge_inplace(ci: usize, stored: &[&[u8]], dst: &mut [u8]) {
    let w = stored.len() + 1;
    let n = dst.len() / w;
    let hi_start = dst.len() - n;
    let p = dst.as_mut_ptr();
    // Forward merge: element i writes bytes [w*i, w*i+w) and reads hi_start+i. For w=2, the write head reaches the read
    // head only at the very last element; handle the final 64 elements scalar through a temporary.
    let safe = n.saturating_sub(64);
    // SAFETY: indices stay in dst; the write range is below hi_start+i for every i < safe (w*i+w-1 < (w-1)n + i holds
    // whenever i < n-1), so no unread plane byte is overwritten.
    unsafe {
        let hi = p.add(hi_start) as *const u8;
        let done = match (w, ci) {
            (2, 1) => { let mut i = 0; while i + 32 <= safe { i += weave2_avx2(stored[0].as_ptr().add(i), hi.add(i), p.add(2 * i), 32); } i }
            (4, 3) => { let mut i = 0; while i + 32 <= safe { i += weave4_avx2([stored[0].as_ptr().add(i), stored[1].as_ptr().add(i), stored[2].as_ptr().add(i), hi.add(i)], p.add(4 * i), 32); } i }
            _ => 0,
        };
        let mut tail = [0u8; 256];
        let tl = n - done;
        if tl <= 256 { std::ptr::copy_nonoverlapping(hi.add(done), tail.as_mut_ptr(), tl); }
        for i in done..n {
            let hv = if tl <= 256 { tail[i - done] } else { *hi.add(i) };
            for k in 0..w {
                let v = if k == ci { hv } else { stored[if k < ci { k } else { k - 1 }][i] };
                *p.add(w * i + k) = v;
            }
        }
    }
}

/// Generic stride writer for one field.
fn stride_field(src: &[u8], dst: &mut [u8], block: usize, f: Field) {
    let (o, l) = (f.off as usize, f.len as usize);
    if l == 1 {
        for (d, s) in dst.chunks_exact_mut(block).zip(src) { d[o] = *s; }
    } else {
        for (d, s) in dst.chunks_exact_mut(block).zip(src.chunks_exact(l)) { d[o..o + l].copy_from_slice(s); }
    }
}

fn cmd_decode(a: &[String]) -> Res<()> {
    let threads: usize = a[1].parse()?;
    let method = a[2].as_str();
    let pretouch = a.get(3).map(|s| s == "pretouch").unwrap_or(false);
    if !is_x86_feature_detected!("avx2") { return Err("avx2 required".into()); }
    let file = std::fs::read(&a[0])?;
    let hdr = std::str::from_utf8(&file[..256])?.trim_end().to_string();
    let h: Vec<&str> = hdr.split(' ').collect();
    let (raw_len, chunk, block, nchunks, ns, joined, hash): (usize, usize, usize, usize, usize, bool, u64) =
        (h[0].parse()?, h[1].parse()?, h[2].parse()?, h[3].parse()?, h[4].parse()?, h[5] == "1", h[6].parse()?);
    let (_, fields, _) = parse_layout(h[7])?;
    let rot = h[7].starts_with("rot:");
    let mut pos = 256usize;
    let mut off = (256 + nchunks * ns * 9) as u64;
    let mut table = Vec::with_capacity(nchunks);
    for _ in 0..nchunks {
        let mut v = Vec::with_capacity(ns);
        for _ in 0..ns {
            let cd = file[pos];
            let rl = u32::from_le_bytes(file[pos + 1..pos + 5].try_into()?);
            let cl = u32::from_le_bytes(file[pos + 5..pos + 9].try_into()?);
            v.push(Stream { codec: cd, raw_len: rl, comp_len: cl, offset: off });
            off += cl as u64;
            pos += 9;
        }
        table.push(v);
    }
    let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build()?;
    let mut dest = memmap2::MmapMut::map_anon(raw_len)?;
    if pretouch { dest.par_chunks_mut(1 << 21).for_each(|c| for p in c.chunks_mut(4096) { p[0] = 1; }); }
    let (rss0, _) = mem()?;
    std::fs::write("/proc/self/clear_refs", "5")?;
    let wide = fields.iter().all(|f| f.len == 1);
    let t = Instant::now();
    pool.install(|| {
        dest.par_chunks_mut(chunk).zip(table.par_iter()).try_for_each_init(
            || (zstd::bulk::Decompressor::new().expect("zstd ctx"), Vec::<u8>::new()),
            |(z, sc), (d, st)| -> Res<()> {
                let comp = |s: &Stream| &file[s.offset as usize..s.offset as usize + s.comp_len as usize];
                let w = fields.len();
                if ns == 1 && !joined {
                    // raw layout: one stream decodes directly into the destination.
                    return dec_stream(st[0].codec, comp(&st[0]), d, z);
                }
                let n = d.len() / block;
                if joined {
                    sc.resize(d.len(), 0);
                    dec_stream(st[0].codec, comp(&st[0]), sc, z)?;
                    let pl: Vec<&[u8]> = sc.chunks_exact(n).collect();
                    if method == "simd" { weave_simd(&pl, d) } else { weave_scalar(&pl, d) }
                    return Ok(());
                }
                match method {
                    "scalar" | "simd" => {
                        if !wide { return Err("scalar/simd need byte planes".into()); }
                        sc.resize(d.len(), 0);
                        let mut rest: &mut [u8] = sc;
                        for s in st.iter() {
                            let (a, b) = rest.split_at_mut(s.raw_len as usize);
                            dec_stream(s.codec, comp(s), a, z)?;
                            rest = b;
                        }
                        let pl: Vec<&[u8]> = sc.chunks_exact(n).collect();
                        if method == "simd" { weave_simd(&pl, d) } else { weave_scalar(&pl, d) }
                    }
                    "stride" => {
                        for (s, f) in st.iter().zip(&fields) {
                            if s.codec == 0 { stride_field(comp(s), d, block, *f); }
                            else {
                                sc.resize(s.raw_len as usize, 0);
                                dec_stream(s.codec, comp(s), sc, z)?;
                                stride_field(sc, d, block, *f);
                            }
                        }
                    }
                    "joint" => {
                        if !wide { return Err("joint needs byte planes".into()); }
                        let need: usize = st.iter().filter(|s| s.codec != 0).map(|s| s.raw_len as usize).sum();
                        sc.resize(need, 0);
                        let mut rest: &mut [u8] = sc;
                        let mut pl: Vec<&[u8]> = Vec::with_capacity(w);
                        for s in st.iter() {
                            if s.codec == 0 { pl.push(comp(s)); }
                            else {
                                let (a, b) = rest.split_at_mut(s.raw_len as usize);
                                dec_stream(s.codec, comp(s), a, z)?;
                                pl.push(&*a);
                                rest = b;
                            }
                        }
                        weave_simd(&pl, d);
                    }
                    "inplace" => {
                        // Needs exactly one compressed plane and it must be the top plane; other cases per chunk use joint.
                        let cc: Vec<usize> = st.iter().enumerate().filter(|(_, s)| s.codec != 0).map(|(i, _)| i).collect();
                        if cc == [w - 1] {
                            let hs = d.len() - n;
                            dec_stream(st[w - 1].codec, comp(&st[w - 1]), &mut d[hs..], z)?;
                            let stored: Vec<&[u8]> = st[..w - 1].iter().map(|s| comp(s)).collect();
                            merge_inplace(w - 1, &stored, d);
                        } else if cc.is_empty() {
                            let stored: Vec<&[u8]> = st.iter().map(|s| comp(s)).collect();
                            weave_simd(&stored, d);
                        } else {
                            return Err(format!("inplace: compressed planes {cc:?}").into());
                        }
                    }
                    m => return Err(format!("bad method {m}").into()),
                }
                if rot { rot_all(d, block, false); }
                Ok(())
            })
    })?;
    let secs = t.elapsed().as_secs_f64();
    let (rss1, hwm) = mem()?;
    let got = xxhash_rust::xxh3::xxh3_64(&dest);
    let extra = hwm as i64 - rss0 as i64 - if pretouch { 0 } else { raw_len as i64 };
    println!("decode file={} threads={} method={} touch={} raw={} speed={:.3}GB/s rss_above_dest={:.1}MiB rss_end_delta={:.0}MiB bitexact={}",
        a[0].rsplit('/').next().unwrap_or(""), threads, method, pretouch, raw_len, raw_len as f64 / secs / 1e9, extra as f64 / 1048576.0, (rss1 as i64 - rss0 as i64) as f64 / 1048576.0, got == hash);
    if got != hash { return Err("hash mismatch".into()); }
    Ok(())
}

fn main() -> Res<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    match a.first().map(|s| s.as_str()) {
        Some("compress") => cmd_compress(&a[1..]),
        Some("decode") => cmd_decode(&a[1..]),
        _ => Err("usage: ztp compress|decode ...".into()),
    }
}
