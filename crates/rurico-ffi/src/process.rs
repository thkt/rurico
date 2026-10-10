//! Unix process/pipe primitives. Callers own the process group they signal.

use std::io;
use std::os::fd::{AsRawFd, BorrowedFd};

use libc::{ESRCH, F_GETFL, F_SETFL, O_NONBLOCK, SIGKILL, fcntl, kill};

/// Make an owned pipe nonblocking without changing its other flags.
#[allow(unsafe_code)]
pub fn set_nonblocking(fd: BorrowedFd<'_>) -> io::Result<()> {
    // SAFETY: BorrowedFd keeps the descriptor alive; these fcntl operations
    // take integer flags and do not dereference a pointer.
    let flags = unsafe { fcntl(fd.as_raw_fd(), F_GETFL) };
    if flags < 0 || unsafe { fcntl(fd.as_raw_fd(), F_SETFL, flags | O_NONBLOCK) } < 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

fn group_id(id: u32) -> io::Result<i32> {
    let id = i32::try_from(id).map_err(io::Error::other)?;
    if id <= 1 {
        return Err(io::Error::other("invalid owned process group"));
    }
    Ok(-id)
}

/// Send SIGKILL to an owned process group. An already absent group is success.
#[allow(unsafe_code)]
pub fn kill_process_group(id: u32) -> io::Result<()> {
    // SAFETY: kill takes integers, not pointers. group_id excludes kill's
    // process-wide special targets (0 and -1).
    if unsafe { kill(group_id(id)?, SIGKILL) } == 0 {
        return Ok(());
    }
    let error = io::Error::last_os_error();
    if error.raw_os_error() == Some(ESRCH) {
        Ok(())
    } else {
        Err(error)
    }
}

/// Whether any member of an owned group still exists (including zombies).
#[allow(unsafe_code)]
pub fn process_group_exists(id: u32) -> io::Result<bool> {
    // SAFETY: signal 0 only checks existence/permission; no pointers involved.
    if unsafe { kill(group_id(id)?, 0) } == 0 {
        return Ok(true);
    }
    let error = io::Error::last_os_error();
    if error.raw_os_error() == Some(ESRCH) {
        Ok(false)
    } else {
        Err(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn group_operations_reject_process_wide_signal_targets() {
        // 0 and -1 have process-wide semantics in kill(2). Neither can be
        // accepted as an owned worker's process group.
        for id in [0, 1, u32::MAX] {
            assert!(kill_process_group(id).is_err());
            assert!(process_group_exists(id).is_err());
        }
    }
}
