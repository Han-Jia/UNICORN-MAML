def augment(x, factor):
    # Flip
    if factor == 0:
        return x.flip(-1)  # horizontal