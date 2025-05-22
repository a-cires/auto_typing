
import time

def run_for_duration(duration_seconds, func, rate_hz=10, *args, **kwargs):
    interval = 1.0 / rate_hz
    start_time = time.time()
    while time.time() - start_time < duration_seconds:
        loop_start = time.time()
        func(*args, **kwargs)
        elapsed = time.time() - loop_start
        sleep_time = max(0, interval - elapsed)
        time.sleep(sleep_time)
