def any_none(a):
    """
    Tell if any element in a is None
    Parameters
    ----------
    a: Iterable

    Returns
    -------
    True or False
    """
    for e in a:
        if e is None:
            return True
    return False