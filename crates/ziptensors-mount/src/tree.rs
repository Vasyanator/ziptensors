//! File: tree.rs
//! Purpose: the directory tree a container's member names describe, with stable node ids.
//!
//! Main responsibilities:
//! - [`Tree::build`]: checks every name against a [`NamePolicy`], refuses the whole tree with one
//!   [`TreeError`] naming every bad member, and numbers the nodes;
//! - lookups by parent and name, sorted listings, the node of an id.
//!
//! Key structures: [`Tree`], [`NodeId`], [`Node`], [`NodeKind`], [`DirEntry`], [`TreeError`],
//! [`BadName`].
//!
//! Implementation notes: ids are assigned breadth first from the root (1), each directory's entries in
//! name order, so the same container always gets the same ids and an id is never reused while the
//! tree lives. Two entries of one directory are the same entry when their policy keys are equal: the
//! name itself under POSIX, the case-folded name under Windows. Directories exist only as prefixes of
//! member names; an empty directory cannot be expressed in a container.

use std::collections::{BTreeMap, HashMap};
use std::fmt;

use crate::policy::{NamePolicy, NameProblem};

/// Id of a node, the inode number of a backend. The root is 1, as FUSE requires.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(u64);

impl NodeId {
    /// The root directory.
    pub const ROOT: Self = Self(1);

    /// The id with number `raw`, as a backend received it. Whether a node has it is for [`Tree::node`].
    #[must_use]
    pub fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    /// The number of this id.
    #[must_use]
    pub fn get(self) -> u64 {
        self.0
    }
}

/// Whether a node is a directory or a file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeKind {
    /// A directory, a prefix of member names.
    Dir,
    /// A regular file, one member.
    File,
}

/// One entry of a directory listing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirEntry {
    /// The name of the entry in its directory, one path part.
    pub name: String,
    /// Its node.
    pub id: NodeId,
    /// What it is.
    pub kind: NodeKind,
}

/// A node of the tree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Node {
    /// A directory.
    Dir {
        /// Its parent; the root is its own parent.
        parent: NodeId,
        /// Its entries, sorted by name (byte order).
        entries: Vec<DirEntry>,
    },
    /// A file: one member of the container.
    File {
        /// Its directory.
        parent: NodeId,
        /// Index of the member in `Container::members` (file order).
        member: usize,
        /// Length of the member in bytes.
        len: u64,
    },
}

impl Node {
    /// What the node is.
    #[must_use]
    pub fn kind(&self) -> NodeKind {
        match self {
            Self::Dir { .. } => NodeKind::Dir,
            Self::File { .. } => NodeKind::File,
        }
    }

    /// The directory holding the node; the root's is the root.
    #[must_use]
    pub fn parent(&self) -> NodeId {
        match self {
            Self::Dir { parent, .. } | Self::File { parent, .. } => *parent,
        }
    }
}

/// One member that cannot be shown, and why.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BadName {
    /// The member name.
    pub member: String,
    /// Why it cannot be shown.
    pub problem: NameProblem,
}

/// A tree that cannot be built: every member that stops it, sorted by name.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub struct TreeError {
    /// Label of the policy the names were checked against.
    pub policy: &'static str,
    /// The members that cannot be shown. Never empty.
    pub bad: Vec<BadName>,
}

impl fmt::Display for TreeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} member name(s) cannot be shown under {} rules: ", self.bad.len(), self.policy)?;
        for (at, bad) in self.bad.iter().enumerate() {
            if at > 0 {
                f.write_str("; ")?;
            }
            write!(f, "{:?} ({})", bad.member, bad.problem)?;
        }
        Ok(())
    }
}

/// The directory tree of a container, immutable once built. `Send + Sync`.
#[derive(Debug)]
pub struct Tree {
    /// Node `id` is at `nodes[id - 1]`.
    nodes: Vec<Node>,
    /// Per directory, the policy key of each entry name to the entry.
    keys: HashMap<NodeId, HashMap<String, NodeId>>,
    /// The policy the names were checked against, which also makes the lookup keys.
    policy: Box<dyn NamePolicy>,
    /// Number of file nodes.
    files: u64,
    /// Sum of the file lengths, saturating.
    total_len: u64,
}

/// A node while the tree is being built.
#[derive(Debug)]
enum Draft {
    Dir {
        /// Name of the directory in its parent.
        name: String,
        /// Policy key to draft index.
        children: BTreeMap<String, usize>,
        /// The first member found under it, for messages.
        first: String,
    },
    File {
        /// Name of the file in its directory.
        name: String,
        /// The whole member name, for messages.
        member_name: String,
        /// Index in file order.
        member: usize,
        /// Length in bytes.
        len: u64,
        /// Whether this file was already reported as being also a directory.
        reported: bool,
    },
}

impl Tree {
    /// Builds the tree of `members`, given as `(name, len)` in file order (`Container::members`).
    ///
    /// # Errors
    /// [`TreeError`] naming every member whose name `policy` refuses, every member that is the same
    /// path as an earlier one under `policy` (a case variant under Windows), and every file that is
    /// also the directory of other members. Nothing is built then.
    pub fn build<'a, P, I>(members: I, policy: P) -> Result<Self, TreeError>
    where
        P: NamePolicy + 'static,
        I: IntoIterator<Item = (&'a str, u64)>,
    {
        let mut drafts = vec![Draft::Dir { name: String::new(), children: BTreeMap::new(), first: String::new() }];
        let mut bad = Vec::new();
        for (member, (name, len)) in members.into_iter().enumerate() {
            if let Err(problem) = check_name(&policy, name) {
                bad.push(BadName { member: name.to_owned(), problem });
                continue;
            }
            insert(&mut drafts, &policy, member, name, len, &mut bad);
        }
        if !bad.is_empty() {
            bad.sort_by(|a, b| a.member.cmp(&b.member));
            return Err(TreeError { policy: policy.label(), bad });
        }
        Ok(Self::number(&drafts, Box::new(policy)))
    }

    /// Numbers a checked draft breadth first and builds the final nodes.
    fn number(drafts: &[Draft], policy: Box<dyn NamePolicy>) -> Self {
        // `order[k]` is the draft of node k + 1; `ids[draft]` its node id.
        let mut order = vec![0_usize];
        let mut ids = vec![NodeId::ROOT; drafts.len()];
        let mut parents = vec![NodeId::ROOT; drafts.len()];
        let mut next = 0;
        while let Some(&draft) = order.get(next) {
            next += 1;
            if let Some(Draft::Dir { children, .. }) = drafts.get(draft) {
                let mut sorted: Vec<usize> = children.values().copied().collect();
                sorted.sort_by(|a, b| draft_name(&drafts[*a]).cmp(draft_name(&drafts[*b])));
                for child in sorted {
                    order.push(child);
                    // Node k + 1 is `order[k]`. A `usize` count always fits in `u64` on supported targets.
                    ids[child] = NodeId(u64::try_from(order.len()).unwrap_or(u64::MAX));
                    parents[child] = ids[draft];
                }
            }
        }
        let mut keys = HashMap::new();
        let mut files = 0_u64;
        let mut total_len = 0_u64;
        let nodes = order
            .iter()
            .map(|&draft| match &drafts[draft] {
                Draft::Dir { children, .. } => {
                    let mut entries: Vec<DirEntry> = children
                        .iter()
                        .map(|(key, &child)| {
                            keys.entry(ids[draft]).or_insert_with(HashMap::new).insert(key.clone(), ids[child]);
                            DirEntry {
                                name: draft_name(&drafts[child]).to_owned(),
                                id: ids[child],
                                kind: draft_kind(&drafts[child]),
                            }
                        })
                        .collect();
                    entries.sort_by(|a, b| a.name.cmp(&b.name));
                    Node::Dir { parent: parents[draft], entries }
                }
                Draft::File { member, len, .. } => {
                    files += 1;
                    total_len = total_len.saturating_add(*len);
                    Node::File { parent: parents[draft], member: *member, len: *len }
                }
            })
            .collect();
        Self { nodes, keys, policy, files, total_len }
    }

    /// The node with id `id`, if there is one.
    #[must_use]
    pub fn node(&self, id: NodeId) -> Option<&Node> {
        let index = usize::try_from(id.0.checked_sub(1)?).ok()?;
        self.nodes.get(index)
    }

    /// The entry called `name` in directory `parent`, compared under the tree's policy.
    /// `None` if `parent` is not a directory or has no such entry.
    #[must_use]
    pub fn lookup(&self, parent: NodeId, name: &str) -> Option<NodeId> {
        self.keys.get(&parent)?.get(&*self.policy.key(name)).copied()
    }

    /// The sorted entries of directory `id`, `None` if it is not a directory.
    #[must_use]
    pub fn entries(&self, id: NodeId) -> Option<&[DirEntry]> {
        match self.node(id)? {
            Node::Dir { entries, .. } => Some(entries),
            Node::File { .. } => None,
        }
    }

    /// Number of nodes, directories and files, the root included. Ids run from 1 to this.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of files.
    #[must_use]
    pub fn file_count(&self) -> u64 {
        self.files
    }

    /// Sum of the file lengths in bytes, saturating at `u64::MAX`.
    #[must_use]
    pub fn total_len(&self) -> u64 {
        self.total_len
    }

    /// Label of the policy the tree was checked against.
    #[must_use]
    pub fn policy(&self) -> &'static str {
        self.policy.label()
    }
}

/// Checks a whole member name and each of its parts.
fn check_name(policy: &dyn NamePolicy, name: &str) -> Result<(), NameProblem> {
    policy.check_path(name)?;
    name.split('/').try_for_each(|part| policy.check_part(part))
}

/// Adds member `member`, called `name`, to the draft. A conflict goes to `bad` and adds nothing.
fn insert(
    drafts: &mut Vec<Draft>,
    policy: &dyn NamePolicy,
    member: usize,
    name: &str,
    len: u64,
    bad: &mut Vec<BadName>,
) {
    let mut dir = 0;
    let mut parts = name.split('/').peekable();
    while let Some(part) = parts.next() {
        let key = policy.key(part).into_owned();
        let last = parts.peek().is_none();
        let existing = match &drafts[dir] {
            Draft::Dir { children, .. } => children.get(&key).copied(),
            // `dir` only ever moves to a directory draft.
            Draft::File { .. } => return,
        };
        let Some(found) = existing else {
            let index = drafts.len();
            drafts.push(if last {
                Draft::File { name: part.to_owned(), member_name: name.to_owned(), member, len, reported: false }
            } else {
                Draft::Dir { name: part.to_owned(), children: BTreeMap::new(), first: name.to_owned() }
            });
            if let Draft::Dir { children, .. } = &mut drafts[dir] {
                children.insert(key, index);
            }
            if last {
                return;
            }
            dir = index;
            continue;
        };
        match &mut drafts[found] {
            Draft::Dir { name: existing_name, first, .. } => {
                if last {
                    let problem = NameProblem::FileIsDirectory { under: first.clone() };
                    bad.push(BadName { member: name.to_owned(), problem });
                    return;
                }
                if existing_name != part {
                    let problem = NameProblem::Collision { with: first.clone() };
                    bad.push(BadName { member: name.to_owned(), problem });
                    return;
                }
                dir = found;
            }
            Draft::File { member_name, reported, .. } => {
                if last {
                    let problem = NameProblem::Collision { with: member_name.clone() };
                    bad.push(BadName { member: name.to_owned(), problem });
                } else if !*reported {
                    // The file is the bad member: everything under its name could be shown without it.
                    *reported = true;
                    let problem = NameProblem::FileIsDirectory { under: name.to_owned() };
                    bad.push(BadName { member: member_name.clone(), problem });
                }
                return;
            }
        }
    }
}

/// Name of a draft node in its directory.
fn draft_name(draft: &Draft) -> &str {
    match draft {
        Draft::Dir { name, .. } | Draft::File { name, .. } => name,
    }
}

/// Kind of a draft node.
fn draft_kind(draft: &Draft) -> NodeKind {
    match draft {
        Draft::Dir { .. } => NodeKind::Dir,
        Draft::File { .. } => NodeKind::File,
    }
}

#[cfg(test)]
mod tests {
    use super::{BadName, Node, NodeId, NodeKind, Tree, TreeError};
    use crate::policy::{NameProblem, PosixPolicy, WindowsPolicy};

    /// Members called `names`, of length 10 × their index.
    fn members<'a>(names: &'a [&'a str]) -> impl Iterator<Item = (&'a str, u64)> + 'a {
        names.iter().zip(0_u64..).map(|(name, at)| (*name, at * 10))
    }

    fn bad(member: &str, problem: NameProblem) -> BadName {
        BadName { member: member.to_owned(), problem }
    }

    fn names(tree: &Tree, id: NodeId) -> Vec<&str> {
        tree.entries(id).expect("a directory").iter().map(|entry| entry.name.as_str()).collect()
    }

    #[test]
    fn a_nested_tree_has_sorted_listings_and_breadth_first_ids() {
        let tree = Tree::build(members(&["z.bin", "b/y.bin", "a.bin", "b/c/x.bin", "b/a.bin"]), PosixPolicy)
            .expect("valid names");
        assert_eq!(names(&tree, NodeId::ROOT), ["a.bin", "b", "z.bin"]);
        let b = tree.lookup(NodeId::ROOT, "b").expect("b");
        assert_eq!(names(&tree, b), ["a.bin", "c", "y.bin"]);
        // Root, then its entries in order, then b's.
        assert_eq!(b.get(), 3);
        assert_eq!(tree.lookup(NodeId::ROOT, "a.bin").map(NodeId::get), Some(2));
        assert_eq!(tree.lookup(NodeId::ROOT, "z.bin").map(NodeId::get), Some(4));
        assert_eq!(tree.lookup(b, "a.bin").map(NodeId::get), Some(5));
        assert_eq!(tree.node_count(), 8);
        let x = tree.lookup(tree.lookup(b, "c").expect("c"), "x.bin").expect("x");
        assert_eq!(tree.node(x), Some(&Node::File { parent: NodeId(6), member: 3, len: 30 }));
        assert_eq!(tree.node(b).map(Node::parent), Some(NodeId::ROOT));
        assert_eq!(tree.node(NodeId::ROOT).map(Node::parent), Some(NodeId::ROOT));
        assert_eq!(tree.file_count(), 5);
        assert_eq!(tree.total_len(), 100);
        assert_eq!(tree.policy(), "posix");
    }

    #[test]
    fn ids_do_not_depend_on_member_order() {
        let one = Tree::build(members(&["d/b", "a", "d/a", "c/x"]), PosixPolicy).expect("valid");
        let two = Tree::build(members(&["c/x", "d/a", "a", "d/b"]), PosixPolicy).expect("valid");
        for path in [["a", ""], ["c", "x"], ["d", "a"], ["d", "b"]] {
            let find = |tree: &Tree| {
                let first = tree.lookup(NodeId::ROOT, path[0]).expect("first part");
                if path[1].is_empty() { first } else { tree.lookup(first, path[1]).expect("second part") }
            };
            assert_eq!(find(&one), find(&two), "{path:?}");
        }
    }

    #[test]
    fn unknown_ids_and_names_find_nothing() {
        let tree = Tree::build(members(&["a/b"]), PosixPolicy).expect("valid");
        assert_eq!(tree.node(NodeId::from_raw(0)), None);
        assert_eq!(tree.node(NodeId::from_raw(4)), None);
        assert_eq!(tree.node(NodeId::from_raw(u64::MAX)), None);
        assert_eq!(tree.lookup(NodeId::ROOT, "b"), None);
        assert_eq!(tree.lookup(NodeId::ROOT, "A"), None);
        // A file has no entries.
        let b = tree.lookup(tree.lookup(NodeId::ROOT, "a").expect("a"), "b").expect("b");
        assert_eq!(tree.entries(b), None);
        assert_eq!(tree.lookup(b, "x"), None);
        assert_eq!(tree.node(b).map(Node::kind), Some(NodeKind::File));
    }

    #[test]
    fn an_empty_container_is_an_empty_root() {
        let tree = Tree::build(std::iter::empty(), PosixPolicy).expect("valid");
        assert_eq!(tree.node_count(), 1);
        assert_eq!(tree.entries(NodeId::ROOT).map(<[_]>::len), Some(0));
    }

    #[test]
    fn every_bad_name_is_reported_and_nothing_is_built() {
        let error = Tree::build(
            members(&["ok.bin", "a//b", "c/", "./d", "e/../f", "/abs", "g\0", "h:i", &"x".repeat(256)]),
            PosixPolicy,
        )
        .expect_err("bad names");
        let mut expected = vec![
            bad("a//b", NameProblem::EmptyPart),
            bad("c/", NameProblem::EmptyPart),
            bad("./d", NameProblem::DotPart),
            bad("e/../f", NameProblem::Unsafe),
            bad("/abs", NameProblem::Unsafe),
            bad("g\0", NameProblem::Unsafe),
            bad("h:i", NameProblem::Unsafe),
            bad(&"x".repeat(256), NameProblem::PartTooLong { len: 256, limit: 255 }),
        ];
        expected.sort_by(|a, b| a.member.cmp(&b.member));
        assert_eq!(error, TreeError { policy: "posix", bad: expected });
        assert!(error.to_string().starts_with("8 member name(s) cannot be shown under posix rules: "));
    }

    #[test]
    fn a_file_that_is_also_a_directory_is_reported_once_in_either_order() {
        let error = Tree::build(members(&["a", "a/x", "a/y", "b/z", "b"]), PosixPolicy).expect_err("conflict");
        assert_eq!(
            error.bad,
            [
                bad("a", NameProblem::FileIsDirectory { under: "a/x".to_owned() }),
                bad("b", NameProblem::FileIsDirectory { under: "b/z".to_owned() }),
            ]
        );
        let deeper = Tree::build(members(&["m/n", "m/n/o/p"]), PosixPolicy).expect_err("conflict");
        assert_eq!(deeper.bad, [bad("m/n", NameProblem::FileIsDirectory { under: "m/n/o/p".to_owned() })]);
    }

    #[test]
    fn posix_is_case_sensitive() {
        let tree = Tree::build(members(&["A.bin", "a.bin", "D/x", "d/x"]), PosixPolicy).expect("distinct");
        assert_eq!(names(&tree, NodeId::ROOT), ["A.bin", "D", "a.bin", "d"]);
    }

    #[test]
    fn windows_reports_case_collisions_of_files_and_directories() {
        let error = Tree::build(members(&["A.bin", "a.bin", "D/x", "d/y", "e/F", "E/f/g"]), WindowsPolicy::default())
            .expect_err("collisions");
        assert_eq!(
            error.bad,
            [
                bad("E/f/g", NameProblem::Collision { with: "e/F".to_owned() }),
                bad("a.bin", NameProblem::Collision { with: "A.bin".to_owned() }),
                bad("d/y", NameProblem::Collision { with: "D/x".to_owned() }),
            ]
        );
        assert_eq!(error.policy, "windows");
    }

    #[test]
    fn windows_reports_a_dotless_i_against_a_capital_i_as_a_collision() {
        // NTFS would show both; the policy folds more and refuses, by decision (see `WindowsPolicy::key`).
        let error = Tree::build(members(&["I.bin", "\u{131}.bin"]), WindowsPolicy::default()).expect_err("folded");
        assert_eq!(error.bad, [bad("\u{131}.bin", NameProblem::Collision { with: "I.bin".to_owned() })]);
        assert!(Tree::build(members(&["I.bin", "\u{131}.bin"]), PosixPolicy).is_ok());
    }

    #[test]
    fn windows_reports_reserved_and_forbidden_names() {
        let error = Tree::build(
            members(&["models/con.safetensors", "a?b", "dir./x", "ok/fine.bin", "nul"]),
            WindowsPolicy::default(),
        )
        .expect_err("bad names");
        assert_eq!(
            error.bad,
            [
                bad("a?b", NameProblem::ForbiddenChar('?')),
                bad("dir./x", NameProblem::TrailingDotOrSpace),
                bad("models/con.safetensors", NameProblem::Reserved),
                bad("nul", NameProblem::Reserved),
            ]
        );
    }

    #[test]
    fn windows_lookup_ignores_case_and_listing_keeps_it() {
        let tree = Tree::build(members(&["Models/Big.BIN"]), WindowsPolicy::default()).expect("valid");
        let dir = tree.lookup(NodeId::ROOT, "models").expect("folded lookup");
        assert_eq!(tree.lookup(NodeId::ROOT, "MODELS"), Some(dir));
        assert_eq!(names(&tree, NodeId::ROOT), ["Models"]);
        assert!(tree.lookup(dir, "big.bin").is_some());
    }
}
