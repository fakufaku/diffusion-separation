import os


def is_rank_zero():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    node_rank = int(os.environ.get("NODE_RANK", 0))
    if local_rank == 0 and node_rank == 0:
        return True

    return False
