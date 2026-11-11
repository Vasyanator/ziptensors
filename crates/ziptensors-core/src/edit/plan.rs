//! File: edit/plan.rs
//! Purpose: resolution of an edit's operations against the members of a container, before anything
//! is read or written.
//!
//! Main responsibilities: the rules of an operation set (each name in at most one operation, what
//! must and must not exist, which bases a delta may name) and the member order of the result, which
//! every strategy follows.
//!
//! Key structures: [`EditPlan`], [`Step`], [`Base`].
//!
//! Key functions: [`resolve`].
//!
//! Implementation notes: pure, over member names only; checks that need the file system (input files,
//! the container as an input) are in `mod.rs`. Order of the result: kept members keep their order, a
//! `Replace` keeps its member's slot, `Add`s follow in operation order. A delta base must come before
//! the member that names it in that order, because the builder adds members in order and a base must
//! be in the pack when its dependant is streamed.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use crate::error::Error;
use crate::format::check_restore_name;
use crate::write::NameRule;

use super::EditOp;

/// The base a new file of an edit is stored against, by its member name in the result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum Base {
    /// A member of the container that the edit keeps as it is.
    Kept(String),
    /// A file this edit adds or puts in a replaced slot, earlier in the result.
    New(String),
}

/// One member of the result, in result order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum Step {
    /// A member of the container, kept as it is (or re-encoded by a repack).
    Keep {
        /// Its name.
        name: String,
    },
    /// A file, added at the end or in the slot of the member it replaces.
    File {
        /// Member name in the result.
        name: String,
        /// The input file.
        path: PathBuf,
        /// The base its tensors are paired with, spec section 7a.
        delta_base: Option<Base>,
        /// An `Add`, whose name the edit introduces; else a `Replace`, which keeps its member's.
        added: bool,
    },
}

impl Step {
    /// The member name of this step in the result.
    pub(super) fn name(&self) -> &str {
        match self {
            Self::Keep { name } | Self::File { name, .. } => name,
        }
    }

    /// The rule the name of this step must pass: only a name the edit introduces must be one a
    /// restore accepts; a kept or replaced member keeps the name its valid container gave it.
    pub(super) fn name_rule(&self) -> NameRule {
        match self {
            Self::File { added: true, .. } => NameRule::New,
            Self::Keep { .. } | Self::File { added: false, .. } => NameRule::Kept,
        }
    }
}

/// A resolved operation set: the members of the result in order, and what was done to the source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct EditPlan {
    /// The result, in member order.
    pub(super) steps: Vec<Step>,
    /// Kept members that a new file names as its delta base ([`Base::Kept`]).
    pub(super) kept_bases: HashSet<String>,
    /// Members kept.
    pub(super) kept: u32,
    /// Files added at the end.
    pub(super) added: u32,
    /// Members removed.
    pub(super) removed: u32,
    /// Members replaced in their slot.
    pub(super) replaced: u32,
}

impl EditPlan {
    /// Whether a new file names a delta base.
    pub(super) fn has_delta(&self) -> bool {
        self.steps.iter().any(|step| matches!(step, Step::File { delta_base: Some(_), .. }))
    }
}

/// The error of an operation set that cannot be applied.
fn refused(detail: String) -> Error {
    Error::EditRefused { detail }
}

/// A count as the report's `u32`.
fn count(value: usize) -> Result<u32, Error> {
    u32::try_from(value).map_err(|_| refused(format!("{value} operations do not fit a u32 count")))
}

/// Resolves `ops` against `members`, the member names of the container in file order.
///
/// # Errors
/// - [`Error::EditRefused`] if a name is in two operations, an `Add` names an existing member, a
///   delta base is the member itself, a removed member, or comes later in the result.
/// - [`Error::MemberNotFound`] if a `Remove` or `Replace` names no member, or a delta base names
///   nothing in the result.
/// - [`Error::UnsafeMemberName`] if a restore would refuse the name of an `Add`.
pub(super) fn resolve(members: &[&str], ops: &[EditOp]) -> Result<EditPlan, Error> {
    let existing: HashSet<&str> = members.iter().copied().collect();
    let mut named: HashSet<&str> = HashSet::with_capacity(ops.len());
    let mut removed = HashSet::new();
    let mut replaced: HashMap<&str, (&PathBuf, Option<&String>)> = HashMap::new();
    let mut adds = Vec::new();
    for op in ops {
        let name = op.name();
        if !named.insert(name) {
            return Err(refused(format!("member {name:?} is named by two operations")));
        }
        match op {
            EditOp::Remove { name } | EditOp::Replace { name, .. }
                if !existing.contains(name.as_str()) =>
            {
                return Err(Error::MemberNotFound { name: name.clone() });
            }
            EditOp::Remove { name } => {
                removed.insert(name.as_str());
            }
            EditOp::Replace { name, path, delta_base } => {
                replaced.insert(name.as_str(), (path, delta_base.as_ref()));
            }
            EditOp::Add { name, .. } if existing.contains(name.as_str()) => {
                return Err(refused(format!(
                    "member {name:?} exists; replace it instead of adding it"
                )));
            }
            EditOp::Add { name, path, delta_base } => {
                check_restore_name(name)?;
                adds.push((name, path, delta_base.as_ref()));
            }
        }
    }

    let mut steps = Vec::with_capacity(members.len() + adds.len());
    for member in members {
        if removed.contains(member) {
            continue;
        }
        let step = match replaced.get(member) {
            Some((path, _)) => Step::File {
                name: (*member).to_owned(),
                path: (*path).clone(),
                delta_base: None,
                added: false,
            },
            None => Step::Keep { name: (*member).to_owned() },
        };
        steps.push(step);
    }
    for (name, path, _) in &adds {
        steps.push(Step::File {
            name: (*name).clone(),
            path: (*path).clone(),
            delta_base: None,
            added: true,
        });
    }

    // Bases are resolved against the result, so that a base may be a file of this edit.
    let position: HashMap<String, Slot> = steps
        .iter()
        .enumerate()
        .map(|(at, step)| {
            let kept = matches!(step, Step::Keep { .. });
            (step.name().to_owned(), Slot { at, kept })
        })
        .collect();
    let wanted: HashMap<&str, &String> = replaced
        .iter()
        .filter_map(|(name, (_, base))| Some((*name, (*base)?)))
        .chain(adds.iter().filter_map(|(name, _, base)| Some((name.as_str(), (*base)?))))
        .collect();
    let mut kept_bases = HashSet::new();
    for (at, step) in steps.iter_mut().enumerate() {
        let Step::File { name, delta_base, .. } = step else {
            continue;
        };
        let Some(base) = wanted.get(name.as_str()) else {
            continue;
        };
        *delta_base = Some(resolve_base(name, base, at, &position, &removed)?);
        if let Some(Base::Kept(base)) = delta_base {
            kept_bases.insert(base.clone());
        }
    }
    let removed_count = count(removed.len())?;
    let replaced_count = count(replaced.len())?;
    let added = count(adds.len())?;
    let kept = count(steps.len() - adds.len() - replaced.len())?;
    Ok(EditPlan { steps, kept_bases, kept, added, removed: removed_count, replaced: replaced_count })
}

/// Where a member is in the result.
#[derive(Debug, Clone, Copy)]
struct Slot {
    /// Its index in the result.
    at: usize,
    /// It is a member of the source kept as it is, and no file of this edit.
    kept: bool,
}

/// The base `base` of the file `name`, at `at` in the result.
fn resolve_base(
    name: &str,
    base: &str,
    at: usize,
    position: &HashMap<String, Slot>,
    removed: &HashSet<&str>,
) -> Result<Base, Error> {
    if base == name {
        return Err(refused(format!("member {name:?} names itself as its delta base")));
    }
    if removed.contains(base) {
        return Err(refused(format!("the delta base {base:?} of {name:?} is removed by this edit")));
    }
    let Some(slot) = position.get(base) else {
        return Err(Error::MemberNotFound { name: base.to_owned() });
    };
    if slot.at > at {
        return Err(refused(format!(
            "the delta base {base:?} of {name:?} comes after it in the result; a base must be \
             added first"
        )));
    }
    Ok(if slot.kept { Base::Kept(base.to_owned()) } else { Base::New(base.to_owned()) })
}
