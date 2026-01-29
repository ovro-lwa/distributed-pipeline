"""File copying utilities with fsync support.

Provides file copying functions that include explicit fsync calls
to ensure data is flushed to disk, important for HPC environments
with network/parallel filesystems.

Modified from shutil to add fsync after write operations.
"""
import os
import stat
from shutil import _samefile, _stat, SpecialFileError, SameFileError, _fastcopy_sendfile, _islink, copymode, copyfileobj


def copy(src: str, dst: str, *, follow_symlinks: bool = True) -> str:
    """Copy file with mode bits, ensuring data is synced to disk.

    Args:
        src: Source file path.
        dst: Destination file or directory path.
        follow_symlinks: If False, copy symlinks as symlinks.

    Returns:
        Destination path.
    """
    if os.path.isdir(dst):
        dst = os.path.join(dst, os.path.basename(src))
    copyfile(src, dst, follow_symlinks=follow_symlinks)
    copymode(src, dst, follow_symlinks=follow_symlinks)
    return dst


def copyfile(src, dst, *, follow_symlinks=True):
    """Copy data from src to dst in the most efficient way possible.

    If follow_symlinks is not set and src is a symbolic link, a new
    symlink will be created instead of copying the file it points to.

    Copied from shutil and modified to add an fsync.

    """

    if _samefile(src, dst):
        raise SameFileError("{!r} and {!r} are the same file".format(src, dst))

    for i, fn in enumerate([src, dst]):
        try:
            st = _stat(fn)
        except OSError:
            # File most likely does not exist
            pass
        else:
            # XXX What about other special files? (sockets, devices...)
            if stat.S_ISFIFO(st.st_mode):
                fn = fn.path if isinstance(fn, os.DirEntry) else fn
                raise SpecialFileError("`%s` is a named pipe" % fn)

    if not follow_symlinks and _islink(src):
        os.symlink(os.readlink(src), dst)
    else:
        with open(src, 'rb') as fsrc:
            try:
                with open(dst, 'wb') as fdst:
                    # _fastcopy_sendfile(fsrc, fdst)
                    copyfileobj(fsrc, fdst)
                    # flush internal buffer then fsync
                    fdst.flush()
                    os.fsync(fdst.fileno())
                    return dst
            except IsADirectoryError as e:
                if not os.path.exists(dst):
                    raise FileNotFoundError(f'Directory does not exist: {dst}') from e
                else:
                    raise
    return dst