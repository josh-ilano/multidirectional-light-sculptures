import os
import time
import shutil

def safe_rmtree(path, retries=5, delay=0.4):
    """
    Robust Windows-safe folder delete.
    Tries multiple times.
    If full delete fails, deletes contents instead.
    """
    for i in range(retries):
        try:
            shutil.rmtree(path)
            return
        except PermissionError:
            time.sleep(delay)

    # fallback: delete contents only
    for root, dirs, files in os.walk(path):
        for f in files:
            try:
                os.remove(os.path.join(root, f))
            except PermissionError:
                pass
        for d in dirs:
            try:
                shutil.rmtree(os.path.join(root, d))
            except PermissionError:
                pass


def reset_output_dirs():
    # directories to reset
    dirs = [
        "outputs/debug",
        "outputs/meshes",
        "outputs/sim",
    ]

    # reset all directories safely
    for d in dirs:
        if os.path.exists(d):
            safe_rmtree(d)

    for d in dirs:
        os.makedirs(d, exist_ok=True)

    print("[PIPELINE] output folders reset")