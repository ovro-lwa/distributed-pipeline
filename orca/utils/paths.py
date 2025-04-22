from importlib.resources import path as resource_path

def get_aoflagger_strategy(name: str) -> str:
    """Return the full path to an AOFlagger strategy file inside the installed package."""
    with resource_path("orca.resources.aoflagger_strategies", name) as p:
        return str(p)
