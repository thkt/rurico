//! Unix process/pipe primitives. Callers own the process group they signal.

use std::io;
use std::os::fd::{AsRawFd, BorrowedFd};

use libc::{EPERM, ESRCH, F_GETFL, F_SETFL, O_NONBLOCK, SIGKILL, fcntl, kill};

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
/// EPERM means the group is present, not that it has been reclaimed. Darwin's
/// killpg1 also returns EPERM for a group containing only zombies.
#[allow(unsafe_code)]
pub fn process_group_exists(id: u32) -> io::Result<bool> {
    // SAFETY: signal 0 only checks existence/permission; no pointers involved.
    if unsafe { kill(group_id(id)?, 0) } == 0 {
        return Ok(true);
    }
    let error = io::Error::last_os_error();
    if error.raw_os_error() == Some(ESRCH) {
        Ok(false)
    } else if error.raw_os_error() == Some(EPERM) {
        Ok(true)
    } else {
        Err(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn group_operations_reject_process_wide_signal_targets() {
        for id in [0, 1, u32::MAX] {
            assert!(kill_process_group(id).is_err());
            assert!(process_group_exists(id).is_err());
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[allow(unsafe_code)]
    fn zombie_group_remains_present_until_reaped_despite_signal_eperm() {
        use std::mem;
        use std::os::unix::process::CommandExt;
        use std::process::{Command, Stdio};
        use std::thread;
        use std::time::{Duration, Instant};

        let mut child = Command::new("sh")
            .args(["-c", "exit 0"])
            .process_group(0)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        let end = Instant::now() + Duration::from_secs(2);
        let observed = loop {
            // SAFETY: waitid writes to initialized siginfo storage. WNOWAIT
            // deliberately leaves this owned child as a zombie until wait.
            let mut info: libc::siginfo_t = unsafe { mem::zeroed() };
            let rc = unsafe {
                libc::waitid(
                    libc::P_PID,
                    child.id(),
                    &mut info,
                    libc::WEXITED | libc::WNOWAIT | libc::WNOHANG,
                )
            };
            if rc != 0 || info.si_pid != 0 || Instant::now() >= end {
                break (rc, info.si_pid);
            }
            thread::sleep(Duration::from_millis(1));
        };
        // SAFETY: this probes only the owned, validated group with signal 0.
        let probe = unsafe { kill(group_id(child.id()).unwrap(), 0) };
        let probe_error = io::Error::last_os_error();
        let present = process_group_exists(child.id());
        let signal = kill_process_group(child.id());
        // Reap before assertions so a failed regression never leaves a zombie.
        let _ = child.kill();
        child.wait().unwrap();
        assert_eq!(observed, (0, i32::try_from(child.id()).unwrap()));
        assert_eq!(probe, -1);
        assert_eq!(probe_error.raw_os_error(), Some(EPERM));
        assert!(present.unwrap(), "zombie group was reported absent");
        assert_eq!(signal.unwrap_err().raw_os_error(), Some(EPERM));
        assert!(!process_group_exists(child.id()).unwrap());
    }
}
