from airo_ipc.framework.framework import initialize_ipc

__airo_ipc_checked_multiprocessing_start_method = False

if not __airo_ipc_checked_multiprocessing_start_method:
    import multiprocessing

    current_start_method = multiprocessing.get_start_method(allow_none=True)
    if current_start_method != "spawn":
        initialize_ipc()
    __airo_ipc_checked_multiprocessing_start_method = True
