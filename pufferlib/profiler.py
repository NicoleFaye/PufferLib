import time

# These will be set by the trainer at the start of each epoch’s env loop.
current_profile = None
current_epoch = None

def section(name: str, nest: bool = False):
    """
    If profiling is active (epoch % frequency == 0), transition to
    a new section named “env_<name>” nested under the Env block.
    Otherwise a no‑op.
    """
    global current_profile, current_epoch
    if current_profile is None or current_epoch is None:
        return
    # only record on profiling epochs
    if current_epoch % current_profile.frequency != 0:
        return
    # if we’re inside the main “env” section, prefix our subsection
    if current_profile.stack and current_profile.stack[-1] == "env":
        name = f"env_{name}"
    current_profile(name, current_epoch, nest=nest)

def end_section():
    """
    Close out the last section (pop timing).
    No‑op if profiling isn’t active.
    """
    global current_profile, current_epoch
    if current_profile is None or current_epoch is None:
        return
    if current_epoch % current_profile.frequency != 0:
        return
    # pop by giving current timestamp
    current_profile.pop(time.time())