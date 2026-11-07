//! File: fuse.rs
//! Purpose: the Linux backend: serves a [`MountFs`] through FUSE (`fuser`, pure Rust, `/dev/fuse` and
//! `fusermount3`), and unmounts on a signal.
//!
//! Main responsibilities:
//! - `FuseFs`, the `fuser::Filesystem` over `Arc<MountFs>`: lookups, attributes, listings, opens with
//!   `FOPEN_KEEP_CACHE`, reads; every change and every write open is refused;
//! - [`mount`] and [`MountHandle`]: a mount served by background threads, unmounted on drop;
//! - [`UnmountSignals`] and [`run_until_signal`]: a foreground mount that a first SIGINT, SIGTERM or
//!   SIGHUP unmounts and a second one abandons ([`MountEnd`]).
//!
//! Key structures: [`FuseOptions`], [`MountHandle`], [`MountEnd`], [`UnmountSignals`], `FuseFs`.
//!
//! Implementation notes:
//! - Contents never change while mounted, so attributes and entries live an hour in the kernel and
//!   every open keeps the page cache: a second read of a file costs no FUSE request.
//! - The session runs `fuser::Session::run` on a thread of ours instead of `Session::spawn`, because
//!   `spawn` moves the mount out of the shared slot a `SessionUnmounter` holds, which leaves that
//!   unmounter unable to unmount.
//! - Signals are waited for with `sigwait` on one thread. That only works if no other thread can take
//!   them, so [`UnmountSignals::block`] must run before any thread exists, the core's decode pool
//!   included, which `Container::open` starts.
//! - An unprivileged unmount is `fusermount3 -u -z`, run by `fuser`: lazy, so it does not fail on a
//!   busy mount. The mountpoint is detached at once; files already open, mappings included, are still
//!   served, and the session ends when the last one is closed. `fuser` reports that unmount as done
//!   whatever `fusermount3` did, so whether the mount went is read from `/proc/self/mountinfo`.
//! - There is no `AutoUnmount` (it needs `allow_other`): a process killed by SIGKILL leaves the mount
//!   behind, and `fusermount3 -u` removes it.

use std::ffi::{OsStr, OsString};
use std::fs;
use std::io;
use std::os::unix::ffi::OsStringExt;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use fuser::{
    Config, Errno, FileAttr, FileHandle, FileType, Filesystem, FopenFlags, Generation,
    INodeNo, IoctlFlags, LockOwner, MountOption, OpenAccMode, OpenFlags, PollEvents, PollFlags,
    PollNotifier, ReplyAttr, ReplyData, ReplyDirectory, ReplyEmpty, ReplyEntry, ReplyIoctl, ReplyOpen,
    ReplyPoll, ReplyStatfs, ReplyXattr, Request, Session, SessionUnmounter,
};
use nix::sys::pthread::{Pthread, pthread_kill, pthread_self};
use nix::fcntl::OFlag;
use nix::sys::signal::{SigSet, Signal};
use nix::unistd::{getgid, getuid};

use crate::error::FsError;
use crate::fs::{Access, Attr, MountFs};
use crate::tree::{NodeId, NodeKind};

/// How long the kernel may keep attributes and lookups. Nothing changes while mounted.
const TTL: Duration = Duration::from_secs(3600);

/// Block size `stat` reports, a buffer size hint for stdio: one chunk of the writer.
const STAT_BLOCK_SIZE: u32 = 1 << 20;

/// Block size of `statfs`.
const STATFS_BLOCK_SIZE: u32 = 4096;

/// Longest name of one path part, as `PosixPolicy` enforces.
const NAME_MAX: u32 = 255;

/// The signals that unmount a foreground mount.
const UNMOUNT_SIGNALS: [Signal; 3] = [Signal::SIGINT, Signal::SIGTERM, Signal::SIGHUP];

/// How a FUSE mount runs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FuseOptions {
    /// Threads that serve kernel requests; 0 counts as 1. Reads of different windows decode in
    /// parallel only with several.
    pub threads: usize,
    /// Source name shown by `mount` and `/proc/mounts`, normally the container path.
    pub fs_name: String,
}

/// The `fuser` filesystem: the neutral [`MountFs`] plus the owner every node shows.
#[derive(Debug)]
struct FuseFs {
    fs: Arc<MountFs>,
    uid: u32,
    gid: u32,
}

impl FuseFs {
    /// The kernel attributes of `attr`: read-only permissions, owned by the mounting user.
    fn file_attr(&self, attr: &Attr) -> FileAttr {
        let (kind, perm) = match attr.kind {
            NodeKind::Dir => (FileType::Directory, 0o555),
            NodeKind::File => (FileType::RegularFile, 0o444),
        };
        FileAttr {
            ino: INodeNo(attr.id.get()),
            size: attr.size,
            blocks: attr.size.div_ceil(512),
            atime: attr.modified,
            mtime: attr.modified,
            ctime: attr.modified,
            crtime: attr.modified,
            kind,
            perm,
            nlink: attr.links,
            uid: self.uid,
            gid: self.gid,
            rdev: 0,
            blksize: STAT_BLOCK_SIZE,
            flags: 0,
        }
    }
}

/// The errno of `error`.
fn errno(error: FsError) -> Errno {
    match error {
        FsError::NotFound => Errno::ENOENT,
        FsError::NotDir => Errno::ENOTDIR,
        FsError::IsDir => Errno::EISDIR,
        FsError::ReadOnly => Errno::EROFS,
        FsError::Io => Errno::EIO,
    }
}

/// The node of a kernel inode number.
fn node(ino: INodeNo) -> NodeId {
    NodeId::from_raw(u64::from(ino))
}

impl Filesystem for FuseFs {
    fn lookup(&self, _req: &Request, parent: INodeNo, name: &OsStr, reply: ReplyEntry) {
        // Member names are UTF-8, so a name that is not cannot exist here.
        let Some(name) = name.to_str() else {
            reply.error(Errno::ENOENT);
            return;
        };
        match self.fs.lookup(node(parent), name) {
            Ok(attr) => reply.entry(&TTL, &self.file_attr(&attr), Generation(0)),
            Err(error) => reply.error(errno(error)),
        }
    }

    fn getattr(&self, _req: &Request, ino: INodeNo, _fh: Option<FileHandle>, reply: ReplyAttr) {
        match self.fs.attr(node(ino)) {
            Ok(attr) => reply.attr(&TTL, &self.file_attr(&attr)),
            Err(error) => reply.error(errno(error)),
        }
    }

    fn open(&self, _req: &Request, ino: INodeNo, flags: OpenFlags, reply: ReplyOpen) {
        // `O_TRUNC` writes even under `O_RDONLY`.
        let writes = flags.0 & OFlag::O_TRUNC.bits() != 0;
        let access = match flags.acc_mode() {
            OpenAccMode::O_RDONLY if !writes => Access::Read,
            OpenAccMode::O_RDONLY | OpenAccMode::O_WRONLY | OpenAccMode::O_RDWR => Access::Write,
        };
        match self.fs.open(node(ino), access) {
            Ok(()) => reply.opened(FileHandle(0), FopenFlags::FOPEN_KEEP_CACHE),
            Err(error) => reply.error(errno(error)),
        }
    }

    fn read(
        &self,
        _req: &Request,
        ino: INodeNo,
        _fh: FileHandle,
        offset: u64,
        size: u32,
        _flags: OpenFlags,
        _lock_owner: Option<LockOwner>,
        reply: ReplyData,
    ) {
        let Ok(size) = usize::try_from(size) else {
            reply.error(Errno::EINVAL);
            return;
        };
        match self.fs.read(node(ino), offset, size) {
            Ok(bytes) => reply.data(&bytes),
            Err(error) => reply.error(errno(error)),
        }
    }

    // `close` sends a flush; there is nothing to flush, and the default answer logs a warning each time.
    fn flush(&self, _req: &Request, _ino: INodeNo, _fh: FileHandle, _lock_owner: LockOwner, reply: ReplyEmpty) {
        reply.ok();
    }

    fn fsync(&self, _req: &Request, _ino: INodeNo, _fh: FileHandle, _datasync: bool, reply: ReplyEmpty) {
        reply.ok();
    }

    fn opendir(&self, _req: &Request, ino: INodeNo, _flags: OpenFlags, reply: ReplyOpen) {
        match self.fs.read_dir(node(ino)) {
            Ok(_) => reply.opened(FileHandle(0), FopenFlags::empty()),
            Err(error) => reply.error(errno(error)),
        }
    }

    fn readdir(&self, _req: &Request, ino: INodeNo, _fh: FileHandle, offset: u64, mut reply: ReplyDirectory) {
        let id = node(ino);
        let (entries, parent) = match (self.fs.read_dir(id), self.fs.parent(id)) {
            (Ok(entries), Ok(parent)) => (entries, parent),
            (Err(error), _) | (_, Err(error)) => {
                reply.error(errno(error));
                return;
            }
        };
        // Entry k answers to offset k + 1: the kernel resumes a listing at the offset of the last entry
        // it took. An offset past the end is the end of the listing.
        let dots = [(id, "."), (parent, "..")].map(|(id, name)| (id, FileType::Directory, name));
        let listed = entries.iter().map(|entry| {
            let kind = match entry.kind {
                NodeKind::Dir => FileType::Directory,
                NodeKind::File => FileType::RegularFile,
            };
            (entry.id, kind, entry.name.as_str())
        });
        let skip = usize::try_from(offset).unwrap_or(usize::MAX);
        for (next, (id, kind, name)) in (1_u64..).zip(dots.into_iter().chain(listed)).skip(skip) {
            if reply.add(INodeNo(id.get()), next, kind, name) {
                break;
            }
        }
        reply.ok();
    }

    fn statfs(&self, _req: &Request, _ino: INodeNo, reply: ReplyStatfs) {
        let (files, bytes) = self.fs.totals();
        let blocks = bytes.div_ceil(u64::from(STATFS_BLOCK_SIZE));
        reply.statfs(blocks, 0, 0, files, 0, STATFS_BLOCK_SIZE, NAME_MAX, STATFS_BLOCK_SIZE);
    }

    // `ls -l` and file managers ask for ACLs and SELinux labels; the default answer logs a warning each
    // time. There are no extended attributes.
    fn getxattr(&self, _req: &Request, _ino: INodeNo, _name: &OsStr, _size: u32, reply: ReplyXattr) {
        reply.error(Errno::NO_XATTR);
    }

    fn listxattr(&self, _req: &Request, _ino: INodeNo, size: u32, reply: ReplyXattr) {
        if size == 0 {
            reply.size(0);
        } else {
            reply.data(&[]);
        }
    }

    // Python asks every file it opens whether it is a terminal (`TCGETS2`); the default answer logs a
    // warning each time. No ioctl applies to a member, and ENOTTY is what a regular file answers.
    fn ioctl(
        &self,
        _req: &Request,
        _ino: INodeNo,
        _fh: FileHandle,
        _flags: IoctlFlags,
        _cmd: u32,
        _in_data: &[u8],
        _out_size: u32,
        reply: ReplyIoctl,
    ) {
        reply.error(Errno::ENOTTY);
    }

    // ENOSYS makes the kernel answer every later poll itself with the mask of a regular file, always
    // ready, which is the truth here. Answered without the default's warning.
    fn poll(
        &self,
        _req: &Request,
        _ino: INodeNo,
        _fh: FileHandle,
        _ph: PollNotifier,
        _events: PollEvents,
        _flags: PollFlags,
        reply: ReplyPoll,
    ) {
        reply.error(Errno::ENOSYS);
    }
}

/// Locks the unmounter. A panic while it was held left no state to repair, so poison is ignored.
fn lock(unmounter: &Mutex<SessionUnmounter>) -> MutexGuard<'_, SessionUnmounter> {
    unmounter.lock().unwrap_or_else(PoisonError::into_inner)
}

/// A live FUSE mount. Dropping it unmounts and waits for the session to end, which is when the last
/// file open under the mount is closed (the unmount is lazy). When the mount is still listed after the
/// unmount, the failure is logged, nothing waits, and the session thread is left to the process.
#[derive(Debug)]
pub struct MountHandle {
    mountpoint: PathBuf,
    /// The mountpoint as `/proc/self/mountinfo` lists it: absolute, links resolved.
    listed: PathBuf,
    unmounter: Arc<Mutex<SessionUnmounter>>,
    /// The thread running the session, until it is joined or detached.
    session: Option<JoinHandle<io::Result<()>>>,
    /// Receives [`Event::Ended`] from the session thread, and [`Event::Abandoned`] from a signal thread.
    events: mpsc::Receiver<Event>,
    /// A sender of `events`, for a signal thread.
    notify: mpsc::Sender<Event>,
}

/// What [`run_until_signal`] waits for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Event {
    /// The session thread returned: unmounted and every open file closed.
    Ended,
    /// A second signal: stop waiting for the session.
    Abandoned(Signal),
}

/// How a foreground mount ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MountEnd {
    /// Unmounted, by a signal or from outside, and every file that was open under it closed.
    Unmounted,
    /// A second signal came while the session still served: files were still open, or the mount would
    /// not go. The session thread is detached; the caller should end the process now, with
    /// [`MountEnd::exit_code`], which cuts off the remaining readers.
    Abandoned {
        /// The second signal.
        signal: Signal,
    },
}

impl MountEnd {
    /// The exit status for this end: 0, or 128 + the signal number for an abandoned mount (130 for a
    /// second Ctrl-C), the shell convention for a death by signal.
    #[must_use]
    pub fn exit_code(self) -> u8 {
        match self {
            Self::Unmounted => 0,
            // Signal numbers of SIGINT, SIGTERM and SIGHUP are below 128, so the sum fits.
            Self::Abandoned { signal } => u8::try_from(128 + signal as i32).unwrap_or(u8::MAX),
        }
    }
}

/// Mounts `fs` at `mountpoint`, read-only and private to the calling user, and serves it from
/// background threads. The mount exists when this returns.
///
/// # Errors
/// A mountpoint that cannot be resolved, and the `io::Error` of `fuser`: no `/dev/fuse`, no
/// `fusermount3`, a mountpoint that is not a directory or is not allowed, a failed handshake with the
/// kernel, or a thread that cannot start.
pub fn mount(fs: Arc<MountFs>, mountpoint: &Path, options: &FuseOptions) -> io::Result<MountHandle> {
    // Resolved before mounting: afterwards, resolving it would ask the mount itself.
    let listed = mountpoint.canonicalize()?;
    let mut config = Config::default();
    config.mount_options = vec![
        MountOption::RO,
        MountOption::NoAtime,
        MountOption::NoDev,
        MountOption::NoSuid,
        // The kernel checks the 0444 and 0555 bits itself, so `X_OK` on a file is refused.
        MountOption::DefaultPermissions,
        MountOption::FSName(options.fs_name.clone()),
        MountOption::Subtype("ziptensors".to_owned()),
    ];
    config.n_threads = Some(options.threads.max(1));
    let filesystem = FuseFs { fs, uid: getuid().as_raw(), gid: getgid().as_raw() };
    tracing::info!(mountpoint = %mountpoint.display(), threads = options.threads.max(1), "mounting");
    let mut session = Session::new(filesystem, mountpoint, &config)?;
    let unmounter = Arc::new(Mutex::new(session.unmount_callable()));
    let (notify, events) = mpsc::channel();
    let ended = notify.clone();
    // If the thread cannot start, the session is dropped with the closure, and its drop unmounts.
    let thread = thread::Builder::new().name("ziptensors-fuse".to_owned()).spawn(move || {
        let result = session.run();
        // The receiver is in the handle; once the handle is gone nobody waits for this news.
        if ended.send(Event::Ended).is_err() {
            tracing::debug!("the FUSE session ended with nobody waiting");
        }
        result
    })?;
    Ok(MountHandle { mountpoint: mountpoint.to_owned(), listed, unmounter, session: Some(thread), events, notify })
}

impl MountHandle {
    /// Where the container is mounted.
    #[must_use]
    pub fn mountpoint(&self) -> &Path {
        &self.mountpoint
    }

    /// Unmounts and waits for the session to end, so for every file still open under the mount to be
    /// closed.
    ///
    /// # Errors
    /// The mount is still listed after the unmount (a failed `fusermount3`, or `umount2` as root): the
    /// session thread is then detached and the mount stays. Otherwise the session's own final error.
    pub fn unmount(mut self) -> io::Result<()> {
        if let Err(error) = unmount_now(&self.unmounter, &self.listed) {
            self.detach();
            return Err(error);
        }
        self.join()
    }

    /// Waits until the mount ends from outside, as `fusermount3 -u` does.
    ///
    /// # Errors
    /// The session's final error.
    pub fn wait(mut self) -> io::Result<()> {
        self.join()
    }

    /// Joins the session thread, once.
    fn join(&mut self) -> io::Result<()> {
        let Some(thread) = self.session.take() else {
            return Ok(());
        };
        let ended = thread.join().map_err(|_panic| io::Error::other("the FUSE session thread panicked"))?;
        tracing::info!(mountpoint = %self.mountpoint.display(), "unmounted");
        match ended {
            // A lazy unmount with files still open ends the session with ECONNABORTED instead of
            // ENODEV once they are closed: the normal end of such a mount, not a failure.
            Err(error) if error.kind() == io::ErrorKind::ConnectionAborted => {
                tracing::debug!(%error, "the FUSE connection was aborted after the unmount");
                Ok(())
            }
            other => other,
        }
    }

    /// Lets the session thread run on unjoined: nobody would end it, so joining would never return.
    fn detach(&mut self) {
        if self.session.take().is_some() {
            tracing::warn!(mountpoint = %self.mountpoint.display(), "the FUSE session is left to the process");
        }
    }
}

impl Drop for MountHandle {
    fn drop(&mut self) {
        if self.session.is_none() {
            return;
        }
        if let Err(error) = unmount_now(&self.unmounter, &self.listed) {
            tracing::error!(
                mountpoint = %self.mountpoint.display(),
                %error,
                "cannot unmount; run `fusermount3 -u -z` on the mountpoint"
            );
            self.detach();
            return;
        }
        if let Err(error) = self.join() {
            tracing::error!(mountpoint = %self.mountpoint.display(), %error, "FUSE session failed");
        }
    }
}

/// Asks `fuser` to unmount, then checks that the mount is gone. The check is needed because `fuser`
/// reports success for an unprivileged unmount whatever `fusermount3` did, and gives up its mount on the
/// first attempt either way.
///
/// # Errors
/// The error of `umount2` (as root), an unreadable `/proc/self/mountinfo`, or a mount still listed.
fn unmount_now(unmounter: &Mutex<SessionUnmounter>, listed: &Path) -> io::Result<()> {
    lock(unmounter).unmount()?;
    if mount_listed(listed)? {
        return Err(io::Error::other(format!("{} is still mounted after the unmount", listed.display())));
    }
    Ok(())
}

/// Whether `/proc/self/mountinfo` lists a mount at `path`, which must be absolute and resolved.
///
/// # Errors
/// The error of reading `/proc/self/mountinfo`.
fn mount_listed(path: &Path) -> io::Result<bool> {
    let mountinfo = fs::read_to_string("/proc/self/mountinfo")?;
    Ok(listed_in(&mountinfo, path))
}

/// Whether the text of a `mountinfo` file has a mount at `path`. The mount point is field 5 of a line,
/// with space, tab, newline and backslash written as `\ooo` octal escapes.
fn listed_in(mountinfo: &str, path: &Path) -> bool {
    mountinfo
        .lines()
        .filter_map(|line| line.split(' ').nth(4))
        .any(|field| Path::new(&unescape_octal(field)) == path)
}

/// `field` with its `\ooo` escapes decoded into bytes.
fn unescape_octal(field: &str) -> OsString {
    let bytes = field.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut at = 0;
    while let Some(&byte) = bytes.get(at) {
        let digits = bytes.get(at + 1..at + 4).filter(|_| byte == b'\\');
        let value = digits.and_then(|digits| {
            digits.iter().try_fold(0_u16, |value, digit| match digit {
                b'0'..=b'7' => Some(value * 8 + u16::from(digit - b'0')),
                _ => None,
            })
        });
        if let Some(value) = value.and_then(|value| u8::try_from(value).ok()) {
            out.push(value);
            at += 4;
        } else {
            out.push(byte);
            at += 1;
        }
    }
    OsString::from_vec(out)
}

/// Proof that SIGINT, SIGTERM and SIGHUP are blocked on the calling thread, and so on every thread it
/// starts from then on. They stay blocked for the life of that thread: unblocking them again would
/// deliver a signal still pending with its default action. Not `Send`: a signal mask belongs to one
/// thread.
#[derive(Debug)]
pub struct UnmountSignals {
    set: SigSet,
    _thread_bound: PhantomData<*const ()>,
}

impl UnmountSignals {
    /// Blocks the unmount signals. Call it before any other thread starts, in particular before
    /// `Container::open`: a thread with them unblocked would take a signal and die of it, leaving the
    /// mount behind.
    ///
    /// # Errors
    /// The error of `pthread_sigmask`.
    pub fn block() -> io::Result<Self> {
        let mut set = SigSet::empty();
        for signal in UNMOUNT_SIGNALS {
            set.add(signal);
        }
        set.thread_block()?;
        Ok(Self { set, _thread_bound: PhantomData })
    }
}

/// Serves `handle` until it ends: a first SIGINT, SIGTERM or SIGHUP unmounts it (lazily: files still
/// open are served until closed), an outside `fusermount3 -u` ends it too, and a second signal stops
/// waiting at once and returns [`MountEnd::Abandoned`], after which the caller should exit.
///
/// # Errors
/// A thread that cannot start, or the session's final error. The mount is unmounted on every error
/// path that returns early.
pub fn run_until_signal(mut handle: MountHandle, signals: &UnmountSignals) -> io::Result<MountEnd> {
    let done = Arc::new(AtomicBool::new(false));
    let (sender, receiver) = mpsc::channel();
    let watcher = {
        let set = signals.set;
        let done = Arc::clone(&done);
        let unmounter = Arc::clone(&handle.unmounter);
        let listed = handle.listed.clone();
        let notify = handle.notify.clone();
        thread::Builder::new().name("ziptensors-signals".to_owned()).spawn(move || {
            // The receiver outlives this send: it is dropped only after the join below.
            if sender.send(pthread_self()).is_ok() {
                watch(set, &unmounter, &listed, &done, &notify);
            }
        })?
    };
    let watcher_id: Pthread = receiver
        .recv()
        .map_err(|_| io::Error::other("the signal thread ended before it started waiting"))?;
    // The handle holds a sender, so the channel cannot close while it is waited on.
    let event = handle.events.recv().map_err(|_| io::Error::other("the mount channel closed"))?;
    let end = match event {
        Event::Ended => handle.join().map(|()| MountEnd::Unmounted),
        Event::Abandoned(signal) => {
            handle.detach();
            Ok(MountEnd::Abandoned { signal })
        }
    };
    done.store(true, Ordering::SeqCst);
    if !watcher.is_finished() {
        // Wakes the watcher from `sigwait`; it sees `done` and returns. A thread that is not yet joined
        // can still be signalled, even if it has just returned.
        if let Err(error) = pthread_kill(watcher_id, Signal::SIGHUP) {
            tracing::warn!(%error, "cannot wake the signal thread");
        }
    }
    watcher.join().map_err(|_panic| io::Error::other("the signal thread panicked"))?;
    end
}

/// Where the signal thread is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Watch {
    /// No signal yet.
    Serving,
    /// The first signal asked for the unmount.
    Unmounting,
}

/// What the signal thread does about one signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Step {
    /// The session is over and this was the wake-up: return.
    Stop,
    /// Unmount, then keep waiting.
    Unmount,
    /// Stop waiting for the session and tell the caller.
    Abandon,
}

/// The transition of the signal thread on a signal. `session_over` wins over everything.
fn on_signal(state: Watch, session_over: bool) -> (Watch, Step) {
    if session_over {
        return (state, Step::Stop);
    }
    match state {
        Watch::Serving => (Watch::Unmounting, Step::Unmount),
        Watch::Unmounting => (Watch::Unmounting, Step::Abandon),
    }
}

/// The signal thread: the first signal unmounts and checks that the mount went, the second abandons
/// the session; `done` makes a signal the wake-up after the end.
fn watch(set: SigSet, unmounter: &Mutex<SessionUnmounter>, listed: &Path, done: &AtomicBool, notify: &mpsc::Sender<Event>) {
    let mut state = Watch::Serving;
    loop {
        let signal = match set.wait() {
            Ok(signal) => signal,
            Err(error) => {
                tracing::error!(%error, "cannot wait for signals; unmount with `fusermount3 -u -z`");
                return;
            }
        };
        let (next, step) = on_signal(state, done.load(Ordering::SeqCst));
        state = next;
        match step {
            Step::Stop => return,
            Step::Unmount => {
                tracing::info!(%signal, mountpoint = %listed.display(), "unmounting on signal");
                match unmount_now(unmounter, listed) {
                    Ok(()) => tracing::info!(
                        mountpoint = %listed.display(),
                        "detached; files still open are served until closed, a second signal ends at once"
                    ),
                    Err(error) => tracing::error!(
                        %error,
                        mountpoint = %listed.display(),
                        "the mount is still there: run `fusermount3 -u -z` on it, or send a second signal to end at once"
                    ),
                }
            }
            Step::Abandon => {
                tracing::warn!(%signal, "second signal: ending without waiting for open files");
                // The receiver is in the handle, which `run_until_signal` holds until this thread is joined.
                if notify.send(Event::Abandoned(signal)).is_err() {
                    tracing::error!("nobody waits for the mount any more");
                }
                return;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use nix::sys::signal::Signal;

    use super::{MountEnd, Step, Watch, listed_in, on_signal};

    #[test]
    fn the_first_signal_unmounts_the_second_abandons_and_the_end_stops() {
        assert_eq!(on_signal(Watch::Serving, false), (Watch::Unmounting, Step::Unmount));
        assert_eq!(on_signal(Watch::Unmounting, false), (Watch::Unmounting, Step::Abandon));
        assert_eq!(on_signal(Watch::Serving, true), (Watch::Serving, Step::Stop));
        assert_eq!(on_signal(Watch::Unmounting, true), (Watch::Unmounting, Step::Stop));
    }

    #[test]
    fn exit_codes_follow_the_shell_convention() {
        assert_eq!(MountEnd::Unmounted.exit_code(), 0);
        assert_eq!(MountEnd::Abandoned { signal: Signal::SIGINT }.exit_code(), 130);
        assert_eq!(MountEnd::Abandoned { signal: Signal::SIGTERM }.exit_code(), 143);
        assert_eq!(MountEnd::Abandoned { signal: Signal::SIGHUP }.exit_code(), 129);
    }

    #[test]
    fn mountinfo_mount_points_are_unescaped() {
        let mountinfo = "22 1 0:21 / /proc rw,nosuid - proc proc rw\n\
            97 30 0:55 / /tmp/a\\040b/m\\134x rw,nodev - fuse.ziptensors /x/c.zt ro\n";
        assert!(listed_in(mountinfo, Path::new("/proc")));
        assert!(listed_in(mountinfo, Path::new("/tmp/a b/m\\x")));
        assert!(!listed_in(mountinfo, Path::new("/tmp/a\\040b/m\\134x")));
        assert!(!listed_in(mountinfo, Path::new("/tmp/a b")));
        assert!(!listed_in("", Path::new("/proc")));
        assert!(!listed_in("short line\n", Path::new("/proc")));
    }
}
