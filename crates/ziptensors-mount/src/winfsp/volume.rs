//! File: winfsp/volume.rs
//! Purpose: `Volume`, the state the `WinFsp` callbacks share: the filesystem, its security descriptor,
//! the device name of the volume, and the per-file count of open file objects that decides when a
//! detached mount may end.
//!
//! Key structures: [`Volume`].
//!
//! Implementation notes: the callbacks reach the volume through the `UserContext` pointer of the
//! `WinFsp` object, from several dispatcher threads at once, so everything mutable sits behind one short
//! mutex, never held across a call that can reach the file system. Only files are counted, not
//! directories: a shell whose current directory is on the mount, or an Explorer window watching it,
//! holds a directory open indefinitely and must not keep an unmount waiting. Counts are per node, so
//! the unmount wait knows which files to reopen (`poke.rs`) to bring a deferred Close. A change that can
//! end the wait wakes both waiters: the condition variable of `MountHandle::unmount` and, when a
//! foreground mount listens, its wake-up channel.

use std::collections::HashMap;
use std::sync::mpsc::Sender;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock, PoisonError};
use std::time::Duration;

use crate::fs::MountFs;
use crate::tree::NodeId;

use super::abi::{FspAddDirInfoFn, NtStatus};
use super::console::Wake;
use super::security::Security;

/// The prefix every `WinFsp` disk volume name has.
const DEVICE_PREFIX: &str = "\\Device\\Volume{";

/// The device name in `raw` if it is `\Device\Volume{...}` followed by a NUL inside `raw`, with
/// nothing but printable ASCII before the NUL.
fn valid_device(raw: &[u16]) -> Option<&[u16]> {
    let len = raw.iter().position(|unit| *unit == 0)?;
    let name = &raw[..len];
    let prefix: Vec<u16> = DEVICE_PREFIX.encode_utf16().collect();
    let shaped = name.starts_with(&prefix)
        && name.len() > prefix.len() + 1
        && name.last() == Some(&u16::from(b'}'))
        && name.iter().all(|unit| (0x21..0x7F).contains(unit));
    shaped.then_some(name)
}

/// Open file objects and the lifecycle flags they are read with.
#[derive(Debug, Default)]
struct OpenState {
    /// File objects opened and not yet closed, per file node. A node is removed at 0.
    files: HashMap<NodeId, usize>,
    /// Sum of `files`.
    total: usize,
    /// The mount point was removed: the volume now only serves files already open.
    detached: bool,
    /// The status the dispatcher ended with when the driver stopped it on its own.
    stopped: Option<NtStatus>,
}

/// What the callbacks of one mounted volume share. `Send + Sync`.
#[derive(Debug)]
pub(super) struct Volume {
    /// The filesystem served.
    pub(super) fs: Arc<MountFs>,
    /// The descriptor every node reports.
    pub(super) security: Security,
    /// `FspFileSystemAddDirInfo`, valid while the `Dll` it came from is loaded, which `MountHandle`
    /// guarantees by dropping the volume first.
    pub(super) add_dir_info: FspAddDirInfoFn,
    /// The NT device name of the volume (`\Device\Volume{...}`), set once after it is created.
    device: OnceLock<Vec<u16>>,
    state: Mutex<OpenState>,
    changed: Condvar,
    /// The wake-up channel of a foreground mount, while one listens.
    notify: Mutex<Option<Sender<Wake>>>,
}

impl Volume {
    /// A volume serving `fs` with `security`, no file open.
    #[must_use]
    pub(super) fn new(fs: Arc<MountFs>, security: Security, add_dir_info: FspAddDirInfoFn) -> Self {
        Self {
            fs,
            security,
            add_dir_info,
            device: OnceLock::new(),
            state: Mutex::new(OpenState::default()),
            changed: Condvar::new(),
            notify: Mutex::new(None),
        }
    }

    /// Locks the state. A panic while it was held left only counters, so poison is ignored.
    fn state(&self) -> MutexGuard<'_, OpenState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Records the NT device name of the volume from `raw`, the `VolumeName` buffer of the `WinFsp`
    /// object; only the first call counts. The name is accepted only in the shape `WinFsp` gives it,
    /// `\Device\Volume{...}` and NUL-terminated inside `raw`: the layout of that object is pinned by
    /// compile-time asserts alone, and an empty or garbage name would turn a reopen into
    /// `\\?\GLOBALROOT\<member path>`, a path outside this volume. Any other content leaves the name
    /// unset, which disables the reopening (`poke.rs`) with one warning; unmounts then wait for a
    /// mapped file's Close as long as the system keeps it.
    pub(super) fn set_device(&self, raw: &[u16]) {
        let Some(device) = valid_device(raw) else {
            tracing::warn!(
                name = %String::from_utf16_lossy(&raw[..raw.len().min(64)]),
                "the WinFsp volume name has an unexpected shape; files kept by a closed mapping will hold an unmount"
            );
            return;
        };
        if self.device.set(device.to_vec()).is_err() {
            tracing::error!("the device name of a volume was set twice");
        }
    }

    /// The NT device name of the volume, once set.
    #[must_use]
    pub(super) fn device(&self) -> Option<&[u16]> {
        self.device.get().map(Vec::as_slice)
    }

    /// Counts a file object of file `id` opened.
    pub(super) fn file_opened(&self, id: NodeId) {
        let mut state = self.state();
        let count = state.files.entry(id).or_insert(0);
        *count = count.saturating_add(1);
        state.total = state.total.saturating_add(1);
    }

    /// Counts a file object of file `id` closed; the last close on a detached volume wakes the waiters.
    pub(super) fn file_closed(&self, id: NodeId) {
        let idle = {
            let mut state = self.state();
            match state.files.get_mut(&id) {
                Some(count) if *count > 1 => *count -= 1,
                Some(_) => {
                    state.files.remove(&id);
                }
                None => {
                    tracing::error!(node = id.get(), "WinFsp closed a file it did not open");
                    return;
                }
            }
            state.total = state.total.saturating_sub(1);
            state.total == 0 && state.detached
        };
        if idle {
            self.wake(Wake::FilesClosed);
        }
    }

    /// Marks the mount point removed. Returns whether no file is open, in which case nothing more will
    /// come to wake a waiter and the volume can be stopped at once.
    pub(super) fn detach(&self) -> bool {
        let mut state = self.state();
        state.detached = true;
        state.total == 0
    }

    /// Records that the driver stopped the dispatcher on its own with `status`, and wakes the waiters.
    pub(super) fn stopped(&self, status: NtStatus) {
        self.state().stopped = Some(status);
        self.wake(Wake::Stopped);
    }

    /// The dispatcher's final status if the driver stopped it on its own.
    #[must_use]
    pub(super) fn stop_status(&self) -> Option<NtStatus> {
        self.state().stopped
    }

    /// File objects open now.
    #[must_use]
    pub(super) fn open_files(&self) -> usize {
        self.state().total
    }

    /// The files with an open file object now, in node order.
    #[must_use]
    pub(super) fn busy_files(&self) -> Vec<NodeId> {
        let mut busy: Vec<NodeId> = self.state().files.keys().copied().collect();
        busy.sort_unstable();
        busy
    }

    /// Blocks up to `timeout` for no file to be open or the driver to stop the volume. Returns whether
    /// that is so.
    pub(super) fn wait_idle(&self, timeout: Duration) -> bool {
        let state = self.state();
        let (state, _) = self
            .changed
            .wait_timeout_while(state, timeout, |state| state.total > 0 && state.stopped.is_none())
            .unwrap_or_else(PoisonError::into_inner);
        state.total == 0 || state.stopped.is_some()
    }

    /// Sets or clears the wake-up channel of a foreground mount.
    pub(super) fn listen(&self, sender: Option<Sender<Wake>>) {
        *self.notify.lock().unwrap_or_else(PoisonError::into_inner) = sender;
    }

    /// Wakes `MountHandle::unmount` and the foreground listener, if any.
    fn wake(&self, wake: Wake) {
        self.changed.notify_all();
        let listener = self.notify.lock().unwrap_or_else(PoisonError::into_inner).clone();
        if let Some(listener) = listener
            && listener.send(wake).is_err()
        {
            tracing::debug!(?wake, "nobody listens for the mount any more");
        }
    }
}

#[cfg(test)]
pub(super) mod tests {
    use std::ffi::c_void;
    use std::sync::mpsc;
    use std::sync::{Arc, Barrier};
    use std::thread;
    use std::time::{Duration, SystemTime};

    use ziptensors_core::{CacheOptions, Container, OpenOptions, ReadCache};

    use super::Volume;
    use crate::fs::MountFs;
    use crate::policy::WindowsPolicy;
    use crate::testkit::{Fixture, Scratch};
    use crate::tree::NodeId;
    use crate::winfsp::abi::DirInfo;
    use crate::winfsp::console::Wake;
    use crate::winfsp::security::Security;

    unsafe extern "C" fn no_dir_info(_: *mut DirInfo, _: *mut c_void, _: u32, _: *mut u32) -> u8 {
        0
    }

    /// A volume over a small container, with no `WinFsp` behind it.
    pub(in crate::winfsp) fn volume(scratch: &Scratch) -> Volume {
        let fixture = Fixture::build(scratch, &[("Models/Unet/w.bin", 100), ("a.bin", 10)]);
        let container = Container::open(&fixture.container, &OpenOptions::default()).expect("open");
        let cache = ReadCache::new(container, CacheOptions::default()).expect("cache");
        let fs = MountFs::new(cache, WindowsPolicy::default(), SystemTime::UNIX_EPOCH).expect("mountable");
        Volume::new(Arc::new(fs), Security::for_current_user().expect("descriptor"), no_dir_info)
    }

    #[test]
    fn counts_are_per_file_and_only_the_last_close_after_a_detach_wakes() {
        let scratch = Scratch::new("winfsp-volume-counts");
        let volume = volume(&scratch);
        let (sender, wakes) = mpsc::channel();
        volume.listen(Some(sender));
        let (a, b) = (NodeId::from_raw(5), NodeId::from_raw(3));
        volume.file_opened(a);
        volume.file_opened(a);
        volume.file_opened(b);
        assert_eq!((volume.open_files(), volume.busy_files()), (3, vec![b, a]));
        volume.file_closed(b);
        assert!(wakes.try_recv().is_err(), "not detached: no wake-up");
        assert!(!volume.detach(), "a file is still open");
        volume.file_closed(a);
        assert_eq!(volume.busy_files(), vec![a]);
        assert!(wakes.try_recv().is_err(), "one more object of a is open");
        volume.file_closed(a);
        assert_eq!(wakes.try_recv(), Ok(Wake::FilesClosed));
        assert_eq!((volume.open_files(), volume.busy_files()), (0, vec![]));
        // A close that was never opened changes nothing and wakes nobody.
        volume.file_closed(a);
        assert_eq!(volume.open_files(), 0);
        assert!(wakes.try_recv().is_err());
        assert!(volume.detach(), "idle");
    }

    #[test]
    fn waiters_wake_on_the_last_close_and_on_a_stop() {
        let scratch = Scratch::new("winfsp-volume-wait");
        let volume = Arc::new(volume(&scratch));
        let id = NodeId::from_raw(2);
        volume.file_opened(id);
        assert!(!volume.detach());
        assert!(!volume.wait_idle(Duration::from_millis(50)), "a file is open");
        let started = Arc::new(Barrier::new(2));
        let closer = {
            let (volume, started) = (Arc::clone(&volume), Arc::clone(&started));
            thread::spawn(move || {
                started.wait();
                thread::sleep(Duration::from_millis(100));
                volume.file_closed(id);
            })
        };
        started.wait();
        assert!(volume.wait_idle(Duration::from_secs(10)), "the close ends the wait");
        closer.join().expect("closer");

        let (sender, wakes) = mpsc::channel();
        volume.listen(Some(sender));
        volume.file_opened(id);
        assert_eq!(volume.stop_status(), None);
        volume.stopped(-1);
        assert_eq!(wakes.try_recv(), Ok(Wake::Stopped));
        assert_eq!(volume.stop_status(), Some(-1));
        assert!(volume.wait_idle(Duration::from_secs(1)), "a stopped volume ends the wait");
    }

    fn units(text: &str) -> Vec<u16> {
        text.encode_utf16().collect()
    }

    #[test]
    fn only_a_winfsp_volume_name_enables_the_reopening() {
        let scratch = Scratch::new("winfsp-volume-device");
        let good = "\\Device\\Volume{4210e682-d064-11f1-b9e9-5cbaeffe11f6}";
        let mut raw = units(good);
        raw.resize(256, 0);
        for bad in [
            vec![0_u16; 256],
            units("\\Device\\HarddiskVolume1\0"),
            units("\\Device\\Volume{\0"),
            units("\\Device\\Volume{x\0"),
            units("\\Device\\Volume{x}\\..\0"),
            units(good),
            vec![0x41; 256],
        ] {
            let volume = volume(&scratch);
            volume.set_device(&bad);
            assert_eq!(volume.device(), None, "{}", String::from_utf16_lossy(&bad));
        }
        let volume = volume(&scratch);
        volume.set_device(&raw);
        assert_eq!(volume.device(), Some(units(good).as_slice()));
        volume.set_device(&units("\\Device\\Volume{other}\0"));
        assert_eq!(volume.device(), Some(units(good).as_slice()), "only the first call counts");
    }
}
