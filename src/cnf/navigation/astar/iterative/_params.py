"""Parameter adaptation for A* search."""


def adapt_params(found, total, successful_iters,
                 current_dropout, min_dropout, current_max_iters,
                 max_iters=float('inf')):
    """Adapt dropout and max_iters based on batch success rate."""
    rate = found / total if total > 0 else 0

    if rate >= 0.67:
        new_dropout = current_dropout
    elif rate >= 0.33:
        new_dropout = max(current_dropout - 0.1, min_dropout)
    else:
        new_dropout = max(current_dropout * 0.5, min_dropout)

    if successful_iters:
        new_max_iters = min(int(1.5 * max(successful_iters)), max_iters)
    else:
        new_max_iters = min(current_max_iters * 2, max_iters)

    return new_dropout, new_max_iters
