import math
import re 
import io as io

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


def find_converged_psi(list_of_lists, target_vpm, tolerance=0.05):
    """
    Iterates through a list of lists of filenames, parses Vpm from filenames
    matching the pattern ...Vpm_####_chimax_####.h5, opens files whose Vpm is
    within `tolerance` of `target_vpm`, and returns the converged result with
    the closest Vpm value.

    Args:
        list_of_lists:  A list of lists, each containing filenames.
        io:             An object with an open() method returning (psi, metadata).
        target_vpm:     The target Vpm value to search near.
        tolerance:      Maximum allowed difference from target_vpm (default 0.05).

    Returns:
        (psi, metadata, vpm) for the converged file with Vpm closest to
        target_vpm, or None if no converged match is found within tolerance.
    """
    pattern = re.compile(r'Vpm_([\d.]+)_chimax_[\d.]+\.h5$')

    best_psi = None
    best_metadata = None
    best_vpm = None
    best_diff = float('inf')

    for file_list in list_of_lists:
        for filename in file_list:
            match = pattern.search(filename)
            if not match:
                continue

            vpm = float(match.group(1))
            diff = abs(vpm - target_vpm)

            if diff > tolerance or diff >= best_diff:
                continue

            psi, metadata = io.load_mps_with_metadata(filename)

            if metadata.get('converged') == True:
                best_psi = psi
                best_metadata = metadata
                best_vpm = vpm
                best_diff = diff

    return (best_psi, best_metadata, best_vpm) if best_psi is not None else None
    