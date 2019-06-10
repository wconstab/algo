from aoc.util.load_input import load_input, build_filename


def solve_part_1(input):
    answer = sum(input)
    return answer


def solve_part_2(input):
    seen = set()
    answer = None
    freq = 0
    seen.add(freq)
    i = 0
    while answer is None:
        delta = input[i % len(input)]
        freq += delta
        if freq in seen:
            answer = freq
            break
        seen.add(freq)
        i += 1

    return answer


def test_part_2():
    test_cases = [([+1, -1], 0),
                  ([+3, +3, +4, -2, -4], 10),
                  ([-6, +3, +8, +5, -6], 5),
                  ([+7, +7, -2, -7, -4], 14)]
    for input, output in test_cases:
        answer = solve_part_2(input)
        assert output == answer, "{} should give {} but gave {}".format(input, output, answer)


if __name__ == "__main__":
    filename = build_filename(__file__, "input")
    input = load_input(filename)
    answer = solve_part_1(input)
    print("Part 1: The frequency is {}".format(answer))
    
    test_part_2()

    input = load_input(filename)
    answer = solve_part_2(input)
    print("Part 2: The frequency is {}".format(answer))


