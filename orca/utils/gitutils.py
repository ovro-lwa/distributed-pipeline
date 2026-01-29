"""Git repository utilities.

Provides functions for querying git repository state, useful for
tracking code versions in output metadata.
"""
import git


def get_commit_id() -> str:
    """Get the current git commit hash.

    Returns:
        The full SHA-1 hash of the current HEAD commit.
    """
    repo = git.Repo(path=__file__, search_parent_directories=True)
    return repo.head.object.hexsha