from collections import Counter
from aoc.util.load_input import load_input, build_filename


def test_part_1():
    input = ['abcdef',
             'bababc',
             'abbcde',
             'abcccd',
             'aabcdd',
             'abcdee',
             'ababab']
    expected = 12
    answer = solve_part_1(input)
    assert answer == expected


def solve_part_1(input):
    twos = 0
    threes = 0
    for i in input:
        _twos, _threes = counts(i)
        twos += 1 if _twos else 0
        threes += 1 if _threes else 0
    answer = twos * threes
    return answer


def test_part_2():
    input = ['abcde',
             'fghij',
             'klmno',
             'pqrst',
             'fguij',
             'axcye',
             'wvxyz']
    expected = 'fgij'
    answer = solve_part_2(input)
    assert expected == answer


def solve_part_2(input):
    match = find_diff_by(input)
    print(match)
    common = find_common_chars(*match)
    print(common)
    return common


def find_diff_by(strings, max_difference=1):
    matches = list()
    for i, s in enumerate(strings):
        if i == len(strings) - 1:
            break

        for s2 in strings[i+1:]:
            assert len(s) == len(s2), "Allowed to have different length?"
            diff = 0
            
            for c_idx in range(len(s)):
                if s[c_idx] != s2[c_idx]:
                    diff += max_difference
                if diff > max_difference:
                    break
            if diff == max_difference:
                matches.append((s, s2))
   
    assert len(matches) == 1, "expected only a single match, got {}".format(len(matches))
    return matches[0]


def find_common_chars(s, s2):
    common = list()
    for c_idx in range(len(s)):
        if s[c_idx] != s2[c_idx]:
            continue
        else:
            common.append(s[c_idx])
    return ''.join(common)


def counts(id):
    twos = set()
    threes = set()
    counter = Counter(id)
    for letter in counter:
        count = counter[letter]
        if count == 2:
            twos.add(letter)
        if count == 3:
            threes.add(letter)

    #print(twos, threes)
    return len(twos), len(threes)
#def solve_part_1(input):
    #answer = sum(input)
    #return answer


if __name__ == "__main__":
    test_part_1()
    filename = build_filename(__file__, "input")
    input = load_input(filename, str)
    answer = solve_part_1(input)
    print("Part 1: The answer is {}".format(answer))

    test_part_2()
    input = load_input(filename, str)
    answer = solve_part_2(input)
    print("Part 2: The answer is {}".format(answer))

