//! File: fs.rs
//! Purpose: [`MountFs`], the backend-neutral filesystem a backend serves: attributes, lookups,
//! listings, opens and reads, over a [`Tree`] and a core [`ReadCache`].
//!
//! Main responsibilities:
//! - build the tree of a container under a [`NamePolicy`] and a cached handle per member;
//! - answer every call a read-only filesystem needs, with [`FsError`] for failures;
//! - clamp reads at the end of a file, as a filesystem read does, and log read failures.
//!
//! Key structures: [`MountFs`], [`Attr`], [`Access`].
//!
//! Implementation notes: no OS type appears here, so a FUSE and a `WinFsp` backend share it. Reads go
//! through `CachedMember::read_at`, because backends see reads far smaller than a chunk (FUSE: 128
//! KiB) and the cache is the one place that knows window geometry. All methods take `&self` and are
//! safe from the backend's dispatch threads.

use std::time::SystemTime;

use ziptensors_core::{CacheStats, CachedMember, ReadCache};

use crate::error::{FsError, MountError};
use crate::policy::NamePolicy;
use crate::tree::{DirEntry, Node, NodeId, NodeKind, Tree};

/// The attributes of a node that do not depend on the OS. A backend adds owner and permission bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Attr {
    /// The node.
    pub id: NodeId,
    /// Directory or file.
    pub kind: NodeKind,
    /// Bytes of a file; 0 for a directory.
    pub size: u64,
    /// Hard links: 1 for a file, 2 plus the subdirectories for a directory (POSIX convention).
    pub links: u32,
    /// Modification time of every node: that of the container file.
    pub modified: SystemTime,
}

/// What an open asks for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Access {
    /// Reading only.
    Read,
    /// Writing, alone or with reading, or truncation.
    Write,
}

/// A read-only filesystem over one container. `Send + Sync`; every method takes `&self`.
#[derive(Debug)]
pub struct MountFs {
    tree: Tree,
    cache: ReadCache,
    /// One handle per member, in file order: `Node::File::member` indexes it.
    members: Vec<CachedMember>,
    modified: SystemTime,
}

impl MountFs {
    /// The filesystem of the container behind `cache`, with names checked by `policy`. `modified` is
    /// the time every node reports, normally the container file's.
    ///
    /// # Errors
    /// [`MountError::Names`] naming every member `policy` cannot show (see [`Tree::build`]), and
    /// [`MountError::Member`] if the core refuses a member handle.
    pub fn new<P>(cache: ReadCache, policy: P, modified: SystemTime) -> Result<Self, MountError>
    where
        P: NamePolicy + 'static,
    {
        let infos = cache.container().members();
        let tree = Tree::build(infos.iter().map(|info| (info.name.as_str(), info.len)), policy)?;
        let members = (0..infos.len())
            .map(|index| cache.member_at(index).map_err(|source| MountError::Member { index, source }))
            .collect::<Result<Vec<_>, _>>()?;
        tracing::info!(
            members = members.len(),
            nodes = tree.node_count(),
            policy = tree.policy(),
            "mount tree built"
        );
        Ok(Self { tree, cache, members, modified })
    }

    /// The tree, for a backend that walks it itself.
    #[must_use]
    pub fn tree(&self) -> &Tree {
        &self.tree
    }

    /// The attributes of node `id`.
    ///
    /// # Errors
    /// [`FsError::NotFound`] if there is no such node.
    pub fn attr(&self, id: NodeId) -> Result<Attr, FsError> {
        let node = self.tree.node(id).ok_or(FsError::NotFound)?;
        let (size, links) = match node {
            Node::Dir { entries, .. } => {
                let subdirs = entries.iter().filter(|entry| entry.kind == NodeKind::Dir).count();
                (0, u32::try_from(subdirs).unwrap_or(u32::MAX).saturating_add(2))
            }
            Node::File { len, .. } => (*len, 1),
        };
        Ok(Attr { id, kind: node.kind(), size, links, modified: self.modified })
    }

    /// The attributes of the entry `name` of directory `parent`.
    ///
    /// # Errors
    /// [`FsError::NotDir`] if `parent` is a file, [`FsError::NotFound`] if it does not exist or has no
    /// such entry.
    pub fn lookup(&self, parent: NodeId, name: &str) -> Result<Attr, FsError> {
        match self.tree.node(parent).ok_or(FsError::NotFound)? {
            Node::Dir { .. } => {}
            Node::File { .. } => return Err(FsError::NotDir),
        }
        self.attr(self.tree.lookup(parent, name).ok_or(FsError::NotFound)?)
    }

    /// The parent of node `id`; the root's is the root.
    ///
    /// # Errors
    /// [`FsError::NotFound`] if there is no such node.
    pub fn parent(&self, id: NodeId) -> Result<NodeId, FsError> {
        self.tree.node(id).map(Node::parent).ok_or(FsError::NotFound)
    }

    /// The entries of directory `id`, sorted by name, without `.` and `..`.
    ///
    /// # Errors
    /// [`FsError::NotFound`] if there is no such node, [`FsError::NotDir`] for a file.
    pub fn read_dir(&self, id: NodeId) -> Result<&[DirEntry], FsError> {
        match self.tree.node(id).ok_or(FsError::NotFound)? {
            Node::Dir { entries, .. } => Ok(entries),
            Node::File { .. } => Err(FsError::NotDir),
        }
    }

    /// Whether file `id` may be opened for `access`. Opens hold no state: every read names its node.
    ///
    /// # Errors
    /// [`FsError::NotFound`], [`FsError::IsDir`] for a directory, [`FsError::ReadOnly`] for any write.
    pub fn open(&self, id: NodeId, access: Access) -> Result<(), FsError> {
        match self.tree.node(id).ok_or(FsError::NotFound)? {
            Node::Dir { .. } => Err(FsError::IsDir),
            Node::File { .. } => match access {
                Access::Read => Ok(()),
                Access::Write => Err(FsError::ReadOnly),
            },
        }
    }

    /// Up to `size` bytes of file `id` from byte `offset`: fewer only at the end of the file, none at
    /// or past it, as a filesystem read returns.
    ///
    /// # Errors
    /// [`FsError::NotFound`], [`FsError::IsDir`], and [`FsError::Io`] when the core fails to read;
    /// that failure is logged with the member, the range and its `ErrorCode`.
    pub fn read(&self, id: NodeId, offset: u64, size: usize) -> Result<Vec<u8>, FsError> {
        let (member, len) = match self.tree.node(id).ok_or(FsError::NotFound)? {
            Node::Dir { .. } => return Err(FsError::IsDir),
            Node::File { member, len, .. } => (*member, *len),
        };
        let handle = self.members.get(member).ok_or(FsError::NotFound)?;
        let available = len.saturating_sub(offset);
        // `take` is at most `size`, so it fits in `usize`.
        let take = usize::try_from(available).map_or(size, |available| available.min(size));
        if take == 0 {
            // At or past the end: the core refuses an offset past the end even for no bytes.
            return Ok(Vec::new());
        }
        let mut bytes = vec![0; take];
        if let Err(error) = handle.read_at(offset, &mut bytes) {
            tracing::error!(
                member = handle.member().name().unwrap_or("<unnamed>"),
                offset,
                len = take,
                code = ?error.code(),
                %error,
                "mount read failed"
            );
            return Err(FsError::Io);
        }
        Ok(bytes)
    }

    /// Number of files and the sum of their lengths, for a backend's volume information.
    #[must_use]
    pub fn totals(&self) -> (u64, u64) {
        (self.tree.file_count(), self.tree.total_len())
    }

    /// Counters of the read cache.
    #[must_use]
    pub fn stats(&self) -> CacheStats {
        self.cache.stats()
    }
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, SystemTime};

    use ziptensors_core::{CacheOptions, Container, OpenOptions, ReadCache};

    use super::{Access, MountFs};
    use crate::error::{FsError, MountError};
    use crate::policy::{NameProblem, PosixPolicy, WindowsPolicy};
    use crate::testkit::{Fixture, Scratch};
    use crate::tree::{NodeId, NodeKind};

    const STAMP: Duration = Duration::from_secs(1_700_000_000);

    /// The filesystem of `fixture`, with the smallest cache the container allows, so reads evict.
    fn mount(fixture: &Fixture) -> MountFs {
        let container = Container::open(&fixture.container, &OpenOptions { threads: 2, ..OpenOptions::default() })
            .expect("open container");
        let options = CacheOptions { max_bytes: 2 << 20, readahead_windows: 2 };
        let cache = ReadCache::new(container, options).expect("cache");
        MountFs::new(cache, PosixPolicy, SystemTime::UNIX_EPOCH + STAMP).expect("mountable")
    }

    /// The node of `path` under the root, `/` separated.
    fn find(fs: &MountFs, path: &str) -> NodeId {
        path.split('/').fold(NodeId::ROOT, |dir, part| fs.lookup(dir, part).expect("entry").id)
    }

    #[test]
    fn every_file_reads_back_whole_and_in_small_pieces() {
        let scratch = Scratch::new("fs-read");
        let fixture = Fixture::standard(&scratch);
        let fs = mount(&fixture);
        for (name, content) in &fixture.files {
            let id = find(&fs, name);
            let attr = fs.attr(id).expect("attr");
            assert_eq!(attr.kind, NodeKind::File);
            assert_eq!(attr.size, u64::try_from(content.len()).expect("small"), "{name}");
            assert_eq!(attr.links, 1);
            assert_eq!(fs.open(id, Access::Read), Ok(()));
            assert_eq!(&fs.read(id, 0, content.len()).expect("whole"), content, "{name}");
            // 128 KiB pieces, as FUSE asks, plus an odd size that straddles windows.
            for piece in [128 << 10, 77_777] {
                let mut joined = Vec::with_capacity(content.len());
                let mut offset = 0_u64;
                loop {
                    let bytes = fs.read(id, offset, piece).expect("piece");
                    if bytes.is_empty() {
                        break;
                    }
                    offset += u64::try_from(bytes.len()).expect("small");
                    joined.extend_from_slice(&bytes);
                }
                assert_eq!(&joined, content, "{name} in pieces of {piece}");
            }
        }
        let stats = fs.stats();
        assert!(stats.hits > 0 && stats.misses > 0, "{stats:?}");
        assert!(stats.evictions > 0, "a 2 MiB cache under a 3 MiB file evicts: {stats:?}");
        assert_eq!(fs.totals(), (fixture.files.len().try_into().expect("small"), fixture.total_len()));
    }

    #[test]
    fn reads_clamp_at_the_end_of_a_file() {
        let scratch = Scratch::new("fs-clamp");
        let fixture = Fixture::standard(&scratch);
        let fs = mount(&fixture);
        let (name, content) = &fixture.files[0];
        let id = find(&fs, name);
        let len = u64::try_from(content.len()).expect("small");
        assert_eq!(fs.read(id, len - 3, 100).expect("tail"), content[content.len() - 3..]);
        assert_eq!(fs.read(id, len, 100).expect("at end"), Vec::<u8>::new());
        assert_eq!(fs.read(id, u64::MAX, 100).expect("far past end"), Vec::<u8>::new());
        assert_eq!(fs.read(id, 0, 0).expect("nothing"), Vec::<u8>::new());
        let empty = find(&fs, "empty.bin");
        assert_eq!(fs.attr(empty).map(|attr| attr.size), Ok(0));
        assert_eq!(fs.read(empty, 0, 4096).expect("empty file"), Vec::<u8>::new());
    }

    #[test]
    fn directories_list_sorted_and_refuse_file_calls() {
        let scratch = Scratch::new("fs-dirs");
        let fixture = Fixture::standard(&scratch);
        let fs = mount(&fixture);
        let root: Vec<_> = fs.read_dir(NodeId::ROOT).expect("root").iter().map(|e| e.name.as_str()).collect();
        assert_eq!(root, ["config.json", "empty.bin", "models"]);
        let models = find(&fs, "models");
        let attr = fs.attr(models).expect("attr");
        assert_eq!((attr.kind, attr.size, attr.links), (NodeKind::Dir, 0, 3));
        assert_eq!(attr.modified, SystemTime::UNIX_EPOCH + STAMP);
        assert_eq!(fs.parent(find(&fs, "models/unet")), Ok(models));
        assert_eq!(fs.parent(NodeId::ROOT), Ok(NodeId::ROOT));
        assert_eq!(fs.open(models, Access::Read), Err(FsError::IsDir));
        assert_eq!(fs.read(models, 0, 10), Err(FsError::IsDir));
        let file = find(&fs, "config.json");
        assert_eq!(fs.read_dir(file).map(<[_]>::len), Err(FsError::NotDir));
        assert_eq!(fs.lookup(file, "x"), Err(FsError::NotDir));
        assert_eq!(fs.open(file, Access::Write), Err(FsError::ReadOnly));
        assert_eq!(fs.lookup(NodeId::ROOT, "missing"), Err(FsError::NotFound));
        let unknown = NodeId::from_raw(10_000);
        assert_eq!(fs.attr(unknown), Err(FsError::NotFound));
        assert_eq!(fs.read(unknown, 0, 1), Err(FsError::NotFound));
        assert_eq!(fs.open(unknown, Access::Read), Err(FsError::NotFound));
        assert_eq!(fs.read_dir(unknown).map(<[_]>::len), Err(FsError::NotFound));
    }

    #[test]
    fn a_container_with_conflicting_names_does_not_mount() {
        let scratch = Scratch::new("fs-conflict");
        let fixture = Fixture::build(
            &scratch,
            &[("Model.bin", 1000), ("model.bin", 2000), ("dir", 10), ("dir/x.bin", 10)],
        );
        let open = || {
            let container = Container::open(&fixture.container, &OpenOptions::default()).expect("open");
            ReadCache::new(container, CacheOptions::default()).expect("cache")
        };
        let error = MountFs::new(open(), PosixPolicy, SystemTime::UNIX_EPOCH).expect_err("conflict");
        let MountError::Names(tree) = error else { panic!("not a name error: {error:?}") };
        assert_eq!(tree.bad.len(), 1);
        assert_eq!(tree.bad[0].member, "dir");
        // The same container under Windows rules also has a case collision.
        let error = MountFs::new(open(), WindowsPolicy::default(), SystemTime::UNIX_EPOCH).expect_err("conflict");
        let MountError::Names(tree) = error else { panic!("not a name error: {error:?}") };
        let problems: Vec<_> = tree.bad.iter().map(|bad| (bad.member.as_str(), &bad.problem)).collect();
        assert_eq!(
            problems,
            [
                ("dir", &NameProblem::FileIsDirectory { under: "dir/x.bin".to_owned() }),
                ("model.bin", &NameProblem::Collision { with: "Model.bin".to_owned() }),
            ]
        );
    }
}
