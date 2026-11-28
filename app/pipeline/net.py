import ctypes
import socket
import sys


def _port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def has_cuda_driver() -> bool:
    """Return whether the NVIDIA driver exposes at least one CUDA device."""
    try:
        if sys.platform == "win32":
            load_library = ctypes.WinDLL
            library_name = "nvcuda.dll"
        elif sys.platform.startswith("linux"):
            load_library = ctypes.CDLL
            library_name = "libcuda.so.1"
        else:
            return False

        driver = load_library(library_name)
        initialize = driver.cuInit
        initialize.argtypes = [ctypes.c_uint]
        initialize.restype = ctypes.c_int

        get_device_count = driver.cuDeviceGetCount
        get_device_count.argtypes = [ctypes.POINTER(ctypes.c_int)]
        get_device_count.restype = ctypes.c_int

        if initialize(0) != 0:
            return False

        count = ctypes.c_int()
        return get_device_count(ctypes.byref(count)) == 0 and count.value > 0
    except (OSError, AttributeError):
        return False
