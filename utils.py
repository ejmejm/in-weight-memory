import equinox as eqx


def tree_replace(tree: eqx.Module, **kwargs) -> eqx.Module:
    """Replaces the values of a tree with the provided keyword arguments."""
    values = [kwargs[k] for k in kwargs]
    return eqx.tree_at(lambda x: [getattr(x, k) for k in kwargs], tree, values)