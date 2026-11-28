import ctypes
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.pipeline import net


@pytest.mark.parametrize(
    ("platform", "loader_name"),
    [
        ("win32", "nvcuda.dll"),
        ("linux", "libcuda.so.1"),
    ],
)
def test_has_cuda_driver_initializes_and_finds_device(
    monkeypatch: pytest.MonkeyPatch,
    platform: str,
    loader_name: str,
) -> None:
    def initialize(_: int) -> int:
        return 0

    def get_device_count(count_pointer: object) -> int:
        ctypes.cast(count_pointer, ctypes.POINTER(ctypes.c_int))[0] = 1
        return 0

    driver = SimpleNamespace(
        cuInit=initialize,
        cuDeviceGetCount=get_device_count,
    )
    loader = Mock(return_value=driver)
    monkeypatch.setattr(net.sys, "platform", platform)
    if platform == "win32":
        monkeypatch.setattr(net.ctypes, "WinDLL", loader, raising=False)
    else:
        monkeypatch.setattr(net.ctypes, "CDLL", loader)

    assert net.has_cuda_driver()
    loader.assert_called_once_with(loader_name)


@pytest.mark.parametrize(
    ("initialize_result", "count_result", "device_count"),
    [
        (1, 0, 1),
        (0, 1, 1),
        (0, 0, 0),
    ],
)
def test_has_cuda_driver_rejects_failed_or_empty_driver(
    monkeypatch: pytest.MonkeyPatch,
    initialize_result: int,
    count_result: int,
    device_count: int,
) -> None:
    def initialize(_: int) -> int:
        return initialize_result

    def get_device_count(count_pointer: object) -> int:
        ctypes.cast(count_pointer, ctypes.POINTER(ctypes.c_int))[0] = device_count
        return count_result

    driver = SimpleNamespace(
        cuInit=initialize,
        cuDeviceGetCount=get_device_count,
    )
    monkeypatch.setattr(net.sys, "platform", "linux")
    monkeypatch.setattr(net.ctypes, "CDLL", lambda _: driver)

    assert not net.has_cuda_driver()


@pytest.mark.parametrize("platform", ["win32", "linux"])
def test_has_cuda_driver_returns_false_when_driver_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    platform: str,
) -> None:
    def missing_driver(_: str) -> object:
        raise OSError("driver library unavailable")

    monkeypatch.setattr(net.sys, "platform", platform)
    if platform == "win32":
        monkeypatch.setattr(net.ctypes, "WinDLL", missing_driver, raising=False)
    else:
        monkeypatch.setattr(net.ctypes, "CDLL", missing_driver)

    assert not net.has_cuda_driver()


def test_has_cuda_driver_returns_false_when_driver_symbol_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def initialize(_: int) -> int:
        return 0

    monkeypatch.setattr(net.sys, "platform", "linux")
    monkeypatch.setattr(
        net.ctypes,
        "CDLL",
        lambda _: SimpleNamespace(cuInit=initialize),
    )

    assert not net.has_cuda_driver()
