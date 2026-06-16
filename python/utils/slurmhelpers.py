import math

def interleaved_indices(n_items, n_tasks):
    """
    Returns indices ordered like

    n_tasks=2:
        0, mid, 1, mid+1, ...

    n_tasks=3:
        0, third, 2*third,
        1, third+1, 2*third+1, ...

    etc.
    """
    block_size = math.ceil(n_items / n_tasks)

    order = []
    for offset in range(block_size):
        for block in range(n_tasks):
            idx = block * block_size + offset
            if idx < n_items:
                order.append(idx)

    return order


def ret_arr_for_taskid(arr, SLURM_ARRAY_TASK_COUNT:int, SLURM_ARRAY_TASK_ID:int):
    ''' 
    IMPORTANT : SLURM_ARRAY_TASK_ID should start from 0, and go until SLURM_ARRAY_TASK_COUNT - 1
    returns an array for the corresponding SLURM_ARRAY_TASK_ID that it has to loop over
    '''
    SLURM_ARRAY_TASK_ID = int(SLURM_ARRAY_TASK_ID)
    SLURM_ARRAY_TASK_COUNT = int(SLURM_ARRAY_TASK_COUNT)

    assert SLURM_ARRAY_TASK_ID < SLURM_ARRAY_TASK_COUNT, "SLURM_ARRAY_TASK_ID should go from 0 to SLURM_ARRAY_TASK_COUNT - 1"
    assert SLURM_ARRAY_TASK_ID >= 0, "SLURM_ARRAY_TASK_ID should go from 0 to SLURM_ARRAY_TASK_COUNT - 1"

    order = interleaved_indices(len(arr), SLURM_ARRAY_TASK_COUNT)
    my_indices = order[SLURM_ARRAY_TASK_ID::SLURM_ARRAY_TASK_COUNT]
    my_arr = [arr[i] for i in my_indices]
    return my_arr