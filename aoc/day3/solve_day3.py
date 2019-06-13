import re
import time

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


    def overlaps(self, other):
        def segment_overlap(a_start, a_end, b_start, b_end):
            return (a_start <= b_start and a_end >= b_start) or (b_start <= a_start and b_end >= a_start)
        row_overlap = segment_overlap(self.row_start, self.row_end, other.row_start, other.row_end)
        col_overlap = segment_overlap(self.col_start, self.col_end, other.col_start, other.col_end)
        # print(self, other, "overlap? row {} col {}".format(row_overlap, col_overlap))
        return row_overlap and col_overlap

    @property
    def row_start(self):
        return self.top_offset

    @property
    def col_start(self):
        return self.left_offset

    @property
    def row_end(self):
        return self.top_offset + self.height - 1

    @property
    def col_end(self):
        return self.left_offset + self.width - 1

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


def solve_part_2(claims):
    t0 = time.time()
    remaining = set(claims)
    no_overlap = set()
    overlapping = set()
    processed = 0
    total = len(remaining)
    while len(remaining):
        claim = remaining.pop()
        for other in remaining:
            if claim.overlaps(other):
                overlapping.add(claim)
                overlapping.add(other)
        
        remaining -= overlapping
        processed += len(overlapping) + 1
        if processed % 100 == 0:
            print("Processed {}, elapsed {} sec".format(processed, time.time()-t0))
        
        if claim not in overlapping:
            no_overlap.add(claim)
        
        overlapping.clear()

    answer = list(no_overlap)
    t1 = time.time()
    elapsed = t1 - t0
    print("day3 part 2 found {} time {} sec".format(len(answer), elapsed))
    return answer


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

    #test_answer = solve_part_1(list(map(parse_input_line, test_input)))
    #assert test_answer == test_expected_answer, "Expected {} got {}".format(test_expected_answer, test_answer)

    #answer = solve_part_1(input)
    #right_answer = 119572
    #print(answer)

    test_answer = solve_part_2(map(Claim, test_input))
    assert len(test_answer) == 1 and test_answer[0].idx == 3, "Expected claim 3 as answer from test data"
    answer = solve_part_2(input)
    assert len(answer) == 1, "expected single answer, got {}".format(len(answer))

