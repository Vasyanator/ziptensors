//! File: winfsp/dll.rs
//! Purpose: finds and loads the `WinFsp` DLL at run time and resolves the exports this backend calls.
//!
//! Key structures: [`Dll`], [`Api`].
//!
//! Key functions: [`Dll::load`], [`install_dir`].
//!
//! Implementation notes: the DLL is loaded only by its full path, the installer's `InstallDir`
//! registry value plus `bin\winfsp-<arch>.dll`, never by bare name through the search path (`WinFsp`'s
//! own `FspLoad` tries that first, which a planted DLL in the current directory could answer). Its
//! dependencies are resolved from its own directory and `System32` only. The module stays
//! loaded while a [`Dll`] lives; every [`Api`] function pointer is valid only that long, so whoever
//! holds a copy of one must also hold the `Dll` (`MountHandle` does). Nothing of `WinFsp` is needed to
//! build: a machine without it gets [`WinFspError::NotInstalled`] at mount time.

#![allow(unsafe_code, reason = "LoadLibraryExW, GetProcAddress and the casts of its results to typed exports")]
#![warn(clippy::undocumented_unsafe_blocks)]

use std::ffi::CStr;
use std::io;
use std::os::windows::ffi::{OsStrExt, OsStringExt};
use std::path::{Path, PathBuf};
use std::ptr;

use windows_sys::Win32::Foundation::{ERROR_FILE_NOT_FOUND, ERROR_SUCCESS, FARPROC, FreeLibrary, HMODULE};
use windows_sys::Win32::System::LibraryLoader::{
    GetProcAddress, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR, LOAD_LIBRARY_SEARCH_SYSTEM32, LoadLibraryExW,
};
use windows_sys::Win32::System::Registry::{HKEY_LOCAL_MACHINE, RRF_RT_REG_SZ, RegGetValueW};

use super::abi::{
    FspAddDirInfoFn, FspCreateFn, FspDeleteFn, FspPreflightFn, FspRemoveMountPointFn, FspSetMountPointFn,
    FspStartDispatcherFn, FspStopDispatcherFn, FspVersionFn,
};
use super::error::WinFspError;

/// Registry key of the installer, under `HKEY_LOCAL_MACHINE`. `WinFsp`'s installer is 32-bit, so on
/// 64-bit Windows its key sits in the WOW64 view.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
const REGISTRY_KEY: &str = "SOFTWARE\\WOW6432Node\\WinFsp";
#[cfg(target_arch = "x86")]
const REGISTRY_KEY: &str = "SOFTWARE\\WinFsp";

/// The DLL of this architecture, relative to the install directory.
#[cfg(target_arch = "x86_64")]
const DLL_PATH: &str = "bin\\winfsp-x64.dll";
#[cfg(target_arch = "aarch64")]
const DLL_PATH: &str = "bin\\winfsp-a64.dll";
#[cfg(target_arch = "x86")]
const DLL_PATH: &str = "bin\\winfsp-x86.dll";

/// The exports this backend calls, typed. Valid while the [`Dll`] they came from is loaded.
#[derive(Debug, Clone, Copy)]
pub(super) struct Api {
    pub(super) preflight: FspPreflightFn,
    pub(super) create: FspCreateFn,
    pub(super) delete: FspDeleteFn,
    pub(super) set_mount_point: FspSetMountPointFn,
    pub(super) remove_mount_point: FspRemoveMountPointFn,
    pub(super) start_dispatcher: FspStartDispatcherFn,
    pub(super) stop_dispatcher: FspStopDispatcherFn,
    pub(super) add_dir_info: FspAddDirInfoFn,
    pub(super) version: FspVersionFn,
}

/// A loaded module, released on drop.
#[derive(Debug)]
struct Module(HMODULE);

impl Drop for Module {
    fn drop(&mut self) {
        // SAFETY: the module was loaded by `Dll::load` and is released once, here; the owner of the
        // `Dll` guarantees that no export is called after this.
        if unsafe { FreeLibrary(self.0) } == 0 {
            tracing::warn!(error = %io::Error::last_os_error(), "FreeLibrary of WinFsp failed");
        }
    }
}

/// The loaded `WinFsp` DLL. Unloaded on drop, which must come after every use of its [`Api`].
#[derive(Debug)]
pub(super) struct Dll {
    api: Api,
    /// `FspVersion`: major version in the high 16 bits, minor in the low.
    version: u32,
    /// Held only for its drop, which unloads the DLL; declared last, so it is released after
    /// everything else of the `Dll`.
    _module: Module,
}

// SAFETY: an `HMODULE` is a process-wide value with no thread affinity, and the exports are
// free-threaded C functions; `Dll` holds nothing else and is never mutated after `load`.
unsafe impl Send for Dll {}
// SAFETY: as above.
unsafe impl Sync for Dll {}

impl Dll {
    /// Loads the DLL named by the registry and resolves every export.
    ///
    /// # Errors
    /// [`WinFspError::NotInstalled`] when the registry has no `WinFsp` install,
    /// [`WinFspError::Registry`] when it cannot be read, [`WinFspError::Load`] when the DLL does not
    /// load, [`WinFspError::MissingExport`] when it lacks an export (a `WinFsp` older than 2.0).
    pub(super) fn load() -> Result<Self, WinFspError> {
        let path = install_dir()?.join(DLL_PATH);
        let wide: Vec<u16> = path.as_os_str().encode_wide().chain(Some(0)).collect();
        // SAFETY: `wide` is a NUL-terminated path that outlives the call; the flags only restrict where
        // dependencies are searched.
        let handle = unsafe {
            LoadLibraryExW(
                wide.as_ptr(),
                ptr::null_mut(),
                LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32,
            )
        };
        if handle.is_null() {
            return Err(WinFspError::Load { path, source: io::Error::last_os_error() });
        }
        // From here `module` unloads the DLL on every early return.
        let module = Module(handle);
        let api = resolve(&module, &path)?;
        let mut version = 0_u32;
        // SAFETY: `FspVersion` writes one `u32` to a valid pointer; the module is loaded.
        let status = unsafe { (api.version)(&raw mut version) };
        if status < 0 {
            tracing::warn!(status = format_args!("{status:#010x}"), "FspVersion failed");
        }
        tracing::info!(
            path = %path.display(),
            version = format_args!("{}.{}", version >> 16, version & 0xFFFF),
            "WinFsp loaded"
        );
        Ok(Self { api, version, _module: module })
    }

    /// The exports.
    #[must_use]
    pub(super) fn api(&self) -> &Api {
        &self.api
    }

    /// The version `FspVersion` reported: major in the high 16 bits, minor in the low.
    #[must_use]
    pub(super) fn version(&self) -> u32 {
        self.version
    }
}

/// Resolves every export of [`Api`] in `module`, loaded from `path`.
///
/// # Errors
/// [`WinFspError::MissingExport`] naming the first export the module lacks.
fn resolve(module: &Module, path: &Path) -> Result<Api, WinFspError> {
    let find = |name: &'static CStr| -> Result<FARPROC, WinFspError> {
        // SAFETY: `module` is loaded while borrowed; `name` is NUL-terminated.
        let found = unsafe { GetProcAddress(module.0, name.as_ptr().cast()) };
        if found.is_none() {
            return Err(WinFspError::MissingExport { name: name.to_str().unwrap_or("<export>"), path: path.to_owned() });
        }
        Ok(found)
    };
    // SAFETY: every transmute below turns the non-null address `find` returned for a named export into
    // that export's documented C signature (`abi.rs`). A function pointer and an `Option` of one have
    // the same size, and `None` was excluded.
    unsafe {
        Ok(Api {
            preflight: std::mem::transmute::<FARPROC, FspPreflightFn>(find(c"FspFileSystemPreflight")?),
            create: std::mem::transmute::<FARPROC, FspCreateFn>(find(c"FspFileSystemCreate")?),
            delete: std::mem::transmute::<FARPROC, FspDeleteFn>(find(c"FspFileSystemDelete")?),
            set_mount_point: std::mem::transmute::<FARPROC, FspSetMountPointFn>(find(c"FspFileSystemSetMountPoint")?),
            remove_mount_point: std::mem::transmute::<FARPROC, FspRemoveMountPointFn>(find(
                c"FspFileSystemRemoveMountPoint",
            )?),
            start_dispatcher: std::mem::transmute::<FARPROC, FspStartDispatcherFn>(find(
                c"FspFileSystemStartDispatcher",
            )?),
            stop_dispatcher: std::mem::transmute::<FARPROC, FspStopDispatcherFn>(find(c"FspFileSystemStopDispatcher")?),
            add_dir_info: std::mem::transmute::<FARPROC, FspAddDirInfoFn>(find(c"FspFileSystemAddDirInfo")?),
            version: std::mem::transmute::<FARPROC, FspVersionFn>(find(c"FspVersion")?),
        })
    }
}

/// The install directory of `WinFsp`, from the registry.
///
/// # Errors
/// [`WinFspError::NotInstalled`] when the key or value is absent, [`WinFspError::Registry`] for any
/// other registry failure.
pub(super) fn install_dir() -> Result<PathBuf, WinFspError> {
    let key: Vec<u16> = REGISTRY_KEY.encode_utf16().chain(Some(0)).collect();
    let value: Vec<u16> = "InstallDir".encode_utf16().chain(Some(0)).collect();
    let mut bytes = 0_u32;
    // SAFETY: a null data pointer asks only for the size in bytes, written to `bytes`.
    let sized = unsafe {
        RegGetValueW(HKEY_LOCAL_MACHINE, key.as_ptr(), value.as_ptr(), RRF_RT_REG_SZ, ptr::null_mut(), ptr::null_mut(), &raw mut bytes)
    };
    registry_result(sized)?;
    let mut buffer = vec![0_u16; usize::try_from(bytes).map_err(|error| WinFspError::Registry { source: io::Error::other(error) })?.div_ceil(2)];
    let mut capacity = u32::try_from(buffer.len() * 2).map_err(|error| WinFspError::Registry { source: io::Error::other(error) })?;
    // SAFETY: `buffer` is writable for `capacity` bytes; `RRF_RT_REG_SZ` makes the call NUL-terminate.
    let read = unsafe {
        RegGetValueW(
            HKEY_LOCAL_MACHINE,
            key.as_ptr(),
            value.as_ptr(),
            RRF_RT_REG_SZ,
            ptr::null_mut(),
            buffer.as_mut_ptr().cast(),
            &raw mut capacity,
        )
    };
    registry_result(read)?;
    let len = buffer.iter().position(|unit| *unit == 0).unwrap_or(buffer.len());
    buffer.truncate(len);
    let dir = PathBuf::from(std::ffi::OsString::from_wide(&buffer));
    if dir.as_os_str().is_empty() || !Path::new(&dir).is_absolute() {
        return Err(WinFspError::Registry {
            source: io::Error::other(format!("InstallDir is not an absolute path: {}", dir.display())),
        });
    }
    Ok(dir)
}

/// The meaning of a `RegGetValueW` result.
fn registry_result(code: u32) -> Result<(), WinFspError> {
    match code {
        ERROR_SUCCESS => Ok(()),
        ERROR_FILE_NOT_FOUND => Err(WinFspError::NotInstalled),
        other => Err(WinFspError::Registry {
            source: io::Error::from_raw_os_error(i32::try_from(other).unwrap_or(i32::MAX)),
        }),
    }
}
