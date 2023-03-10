"""script for measuring the precision/accuracy of time.sleep()
Note that this will heavily depend on context (what other processes are running?), as the scheduler is the culprit of most of the differences.

Python 3.11 promises  better results https://docs.python.org/3/whatsnew/3.11.html#time

inspired by https://stackoverflow.com/questions/1133857/how-accurate-is-pythons-time-sleep
"""

import time

import matplotlib.pyplot as plt


def measure_sleep_time(amount_in_s):
    start = time.time_ns()
    time.sleep(amount_in_s)
    end = time.time_ns()
    difference_in_ns = end - start
    difference_in_ms = difference_in_ns / 1000000
    return difference_in_ms


def measure_sleeping_performance(sleep_times_in_ms):
    sleep_time_measurements = {}
    for sleep_time in sleep_times_in_ms:
        # do 100 measurements
        measurements = []
        for i in range(100):
            measurements.append(measure_sleep_time(sleep_time / 1000))

        sleep_time_measurements[sleep_time] = measurements

    return sleep_time_measurements


if __name__ == "__main__":
    import pathlib

    import numpy as np

    path = pathlib.Path(__file__).parent.absolute()
    # test a few discrete sleep times, to also get an estimation of the variance
    sleep_times_in_ms = [0.1, 0.2, 0.5, 1, 2, 5, 10]
    sleep_time_measurements = measure_sleeping_performance(sleep_times_in_ms)
    for sleep_time in sleep_time_measurements.keys():
        measurements = sleep_time_measurements[sleep_time]
        plt.scatter([sleep_time] * len(measurements), measurements, marker="x", color="red")
    plt.plot([0.1, 10], [0.1, 10], color="blue")
    plt.xlabel("desired sleep time in ms")
    plt.ylabel("actual sleep in ms")
    plt.xscale("log")
    plt.yscale("log")
    plt.title("accuracy of time.sleep() on Ubuntu 20.04 regular kernel")
    plt.savefig(path / "sleep_actual_vs_desired.png")

    plt.clf()
    for sleep_time in sleep_time_measurements.keys():
        measurements = sleep_time_measurements[sleep_time]
        plt.scatter([sleep_time] * len(measurements), np.array(measurements) - sleep_time, marker="x", color="red")
    plt.xlabel("desired sleep time in ms")
    plt.ylabel("errorin ms")
    plt.xscale("log")
    plt.title("accuracy of time.sleep() on Ubuntu 20.04 regular kernel")
    plt.savefig(path / "sleep_error.png")
