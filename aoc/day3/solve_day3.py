import re


"""
Each claim's rectangle is defined as follows:

- The number of inches between the left edge of the fabric and the left edge of the rectangle.
- The number of inches between the top edge of the fabric and the top edge of the rectangle.
- The width of the rectangle in inches.
- The height of the rectangle in inches.
"""

class Claim(object):
    
    pattern = "#(\d+) @ (\d+),(\d+): (\d+)x(\d+)"
    parser = re.compile(pattern)

    def __init__(self, init_str=None):
       
        self.idx = None
        self.left_offset = None
        self.top_offset = None
        self.width = None
        self.height = None
        if init_str:
            self.parse_from_string(init_str)

    def parse_from_string(self, init_str):
        res = Claim.parser.match(init_str)
        idx, left_offset, top_offset, width, height = res.groups()
        self.idx = int(idx)
        self.left_offset = int(left_offset)
        self.top_offset = int(top_offset)
        self.width = int(width)
        self.height = int(height)
         

    @property
    def col_offset(self):
        return self.left_offset


    @property
    def row_offset(self):
        return self.top_offset


    def __str__(self):
        return "#{idx} @ {left_offset},{top_offset}: {width}x{height}".format(**self.__dict__)


def parse_input_line(line):
    claim = Claim(line)
    return claim


def test_parse_input(input):
    for test in input:
        claim = parse_input_line(test)
        assert str(claim) == test.strip(), "Mismatch {} !=  {}".format(str(claim), test)


def covered_by_claims(point, claims, min_claims=2):
    covered = 0
    row, col = point
    for claim in claims:
        if (row >= claim.row_offset and
            col >= claim.col_offset and
            row < claim.row_offset + claim.height and
            col < claim.col_offset + claim.width):
            covered += 1
        if covered >= min_claims:
            break

    return covered >= min_claims


def solve_part_1(claims):
    rows = 0
    cols = 0
    for claim in claims:
        if claim.row_offset + claim.height > rows:
            rows = claim.row_offset + claim.height
        if claim.col_offset + claim.width > cols:
            cols = claim.col_offset + claim.width

    covered_points = set()
    for r in range(rows):
        for c in range(cols):
            if r % 10 == 0 and c == 0:
                print(r, c)
            if covered_by_claims((r, c), claims, min_claims=2):
                covered_points.add((r,c))
    return len(covered_points)


def load_input(filename="input"):
    with open(filename) as f:
        return f.readlines()

if __name__ == "__main__":
    test_input = ["#1 @ 1,3: 4x4",
                  "#2 @ 3,1: 4x4",
                  "#3 @ 5,5: 2x2"]
    test_expected_answer = 4

    test_parse_input(test_input)

    input_lines = load_input()
    test_parse_input(input_lines)
    input = list(map(parse_input_line, input_lines))

    test_answer = solve_part_1(list(map(parse_input_line, test_input)))
    assert test_answer == test_expected_answer, "Expected {} got {}".format(test_expected_answer, test_answer)

    answer = solve_part_1(input)
    right_answer = 119572
    print(answer)
