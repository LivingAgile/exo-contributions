import ctypes
import sys
from typing import cast

GPU_DRAIN_TIMEOUT_SECONDS = 120
GPU_DRAIN_LIMIT_BYTES = 1024**3
GPU_DRAIN_LOW_SAMPLES = 3
GPU_DRAIN_POLL_SECONDS = 0.05


class MacOSAllocationReader:
    def __init__(self) -> None:
        if sys.platform != "darwin":
            raise RuntimeError("GPU drain requires the macOS allocation reader")
        self.io = ctypes.CDLL("/System/Library/Frameworks/IOKit.framework/IOKit")
        self.cf = ctypes.CDLL(
            "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"
        )
        for library, name, arguments, result in (
            (self.io, "IOServiceMatching", [ctypes.c_char_p], ctypes.c_void_p),
            (
                self.io,
                "IOServiceGetMatchingServices",
                [ctypes.c_uint, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)],
                ctypes.c_int,
            ),
            (self.io, "IOIteratorNext", [ctypes.c_uint], ctypes.c_uint),
            (self.io, "IOObjectRelease", [ctypes.c_uint], ctypes.c_int),
            (
                self.io,
                "IORegistryEntryCreateCFProperty",
                [ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint],
                ctypes.c_void_p,
            ),
            (
                self.cf,
                "CFStringCreateWithCString",
                [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint],
                ctypes.c_void_p,
            ),
            (
                self.cf,
                "CFDictionaryGetValue",
                [ctypes.c_void_p, ctypes.c_void_p],
                ctypes.c_void_p,
            ),
            (self.cf, "CFGetTypeID", [ctypes.c_void_p], ctypes.c_ulong),
            (self.cf, "CFDictionaryGetTypeID", [], ctypes.c_ulong),
            (self.cf, "CFNumberGetTypeID", [], ctypes.c_ulong),
            (
                self.cf,
                "CFNumberGetValue",
                [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p],
                ctypes.c_bool,
            ),
            (self.cf, "CFRelease", [ctypes.c_void_p], None),
        ):
            function = cast(object, getattr(library, name))
            for attribute, value in (("argtypes", arguments), ("restype", result)):
                setattr(function, attribute, value)
        self.service = 0
        self.statistics_key: int | None = None
        self.allocation_key: int | None = None
        iterator = ctypes.c_uint()
        if (
            self.io.IOServiceGetMatchingServices(
                0, self.io.IOServiceMatching(b"AGXAccelerator"), ctypes.byref(iterator)
            )
            != 0
        ):
            raise RuntimeError("accelerator enumeration failed")
        try:
            self.service = cast(int, self.io.IOIteratorNext(iterator.value))
            extra_service = cast(int, self.io.IOIteratorNext(iterator.value))
            if extra_service:
                self.io.IOObjectRelease(extra_service)
            if not self.service or extra_service:
                raise RuntimeError("accelerator root is not unique")
            self.statistics_key = self.cf.CFStringCreateWithCString(
                None, b"PerformanceStatistics", 0x08000100
            )
            self.allocation_key = self.cf.CFStringCreateWithCString(
                None, b"Alloc system memory", 0x08000100
            )
            if not self.statistics_key or not self.allocation_key:
                raise RuntimeError("allocation reader initialization failed")
        except BaseException:
            self.close()
            raise
        finally:
            self.io.IOObjectRelease(iterator.value)

    def allocation(self) -> int:
        statistics = cast(
            int | None,
            self.io.IORegistryEntryCreateCFProperty(
                self.service, self.statistics_key, None, 0
            ),
        )
        if not statistics:
            raise RuntimeError("accelerator statistics unavailable")
        try:
            if self.cf.CFGetTypeID(statistics) != self.cf.CFDictionaryGetTypeID():
                raise RuntimeError("accelerator statistics type changed")
            number = cast(
                int | None,
                self.cf.CFDictionaryGetValue(statistics, self.allocation_key),
            )
            if not number or self.cf.CFGetTypeID(number) != self.cf.CFNumberGetTypeID():
                raise RuntimeError("accelerator allocation type changed")
            value = ctypes.c_int64()
            if (
                not self.cf.CFNumberGetValue(number, 4, ctypes.byref(value))
                or value.value < 0
            ):
                raise RuntimeError("accelerator allocation unreadable")
            return value.value
        finally:
            self.cf.CFRelease(statistics)

    def close(self) -> None:
        for key in (self.statistics_key, self.allocation_key):
            if key:
                self.cf.CFRelease(key)
        if self.service:
            self.io.IOObjectRelease(self.service)
        self.service = 0
        self.statistics_key = None
        self.allocation_key = None
