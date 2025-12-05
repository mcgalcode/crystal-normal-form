
def maybe_profile(fn: callable):
    try:
        return profile(fn)
    except:
        return fn