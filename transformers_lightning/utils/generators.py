from typing import Generator


def filter_generator(generator_in: Generator, step: int = 1, offset: int = 0) -> Generator:
    r"""
    Return elements from a generator. First `offset` elements are discarded
    Then, return an element after every every `step` extracted
    """

    assert step is not None and step >= 0, f"step must be non-negative, found {step}"
    assert offset is not None and offset >= 0, f"offset must be non-negative, found {offset}"

    # advance to the target offset and return first element
    for _ in range(offset):
        try:
            next(generator_in)
        except:
            return
    try:
        yield next(generator_in)
    except:
        return

    while True:
        # consume world_size - 1 inputs
        for _ in range(step - 1):
            try:
                next(generator_in)
            except:
                return
        try:
            yield next(generator_in)
        except:
            return


def batch_filter(generator_in: Generator, size: int = 1) -> Generator:
    r"""
    By reading `size` elements at a time, we assure that no last iteration will have
    a different batch size across nodes, that would cause a fail.
    """
    assert size >= 0, f"Cannot read {size} elements at a time. size must be >= 0"
    while True:
        res = []
        for i in range(size):
            try:
                res.append(next(generator_in))
            except:
                return
        for i in res:
            yield i
