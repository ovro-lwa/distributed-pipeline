import git


def get_commit_id():
    repo = git.Repo(path=__file__, search_parent_directories=True)
    return repo.head.object.hexsha