import re
from datetime import datetime
from collections import Counter, defaultdict
from aoc.util.load_input import load_input, build_filename


def parse_line(line):
    #            (date), (desc (guard)              (wakes up) (falls asleep))
    pattern = r"\[(.*)\] (Guard #(\d+) begins shift|(wakes up)|(falls asleep))"
    res = re.match(pattern, line)
    date, _, guard, wakes_up, falls_asleep = res.groups()

    datetime_format = "%Y-%m-%d %H:%M"
    date = datetime.strptime(date, datetime_format)
    if guard:
        guard = int(guard)
    return (date, guard, wakes_up, falls_asleep)


def get_input_from_file(name, parse=True, sort_by_date=True):
    input = load_input(build_filename(__file__, name), typ=str)

    if parse:
        input = [parse_line(l) for l in input]
        if sort_by_date:
            input.sort(key=lambda x: x[0])
    else:
        assert not sort_by_date, "can only sort if parsing"

    return input


def solve_day4_part1(entries):
    minute_tracker = defaultdict(Counter)
    current_guard = None
    current_guard_fell_asleep = None
    most_minutes_slept = 0
    guard_who_slept_most = None
    for date, guard, wakes_up, falls_asleep in entries:
        if guard:
            current_guard = guard
            current_guard_minutes_slept = 0
        elif falls_asleep:
            current_guard_fell_asleep = date
        elif wakes_up:
            for minute in range(current_guard_fell_asleep.minute, date.minute):
                minute_tracker[current_guard][minute] += 1
            minutes_asleep = date.minute - current_guard_fell_asleep.minute
            current_guard_minutes_slept += minutes_asleep
            if current_guard_minutes_slept > most_minutes_slept:
                most_minutes_slept = current_guard_minutes_slept
                guard_who_slept_most = current_guard
        

    minutes = minute_tracker[guard_who_slept_most]
    minute, _ = minutes.most_common(1)[0]

    return minute * guard_who_slept_most

if __name__ == "__main__":
    test_input_raw = get_input_from_file("test_input", parse=False, sort_by_date=False)
    test_input = get_input_from_file("test_input", parse=True)

    events = get_input_from_file("input", parse = True)
    answer = solve_day4_part1(events)
    print("Day 4 Part 1 answer: The product of guard x minutes slept is {}".format(answer))

