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
    
    pattern = r"#(\d+) @ (\d+),(\d+): (\d+)x(\d+)"
    parser = re.compile(pattern)

    def __init__(self, init_str=None):
       
        self.idx = None
        self.col_start = None
        self.row_start = None
        self.width = None
        self.height = None
        if init_str:
            self.parse_from_string(init_str)

    def parse_from_string(self, init_str):
        res = Claim.parser.match(init_str)
        idx, col_start, row_start, width, height = res.groups()
        self.idx = int(idx)
        self.col_start = int(col_start)
        self.row_start = int(row_start)
        self.width = int(width)
        self.height = int(height)


    def overlaps(self, other):
        def segment_overlap(a_start, a_end, b_start, b_end):
            return (a_start <= b_start and a_end >= b_start) or (b_start <= a_start and b_end >= a_start)
        row_overlap = segment_overlap(self.row_start, self.row_end, other.row_start, other.row_end)
        col_overlap = segment_overlap(self.col_start, self.col_end, other.col_start, other.col_end)
        return row_overlap and col_overlap

    @property
    def row_end(self):
        return self.row_start + self.height - 1

    @property
    def col_end(self):
        return self.col_start + self.width - 1

    def __str__(self):
        return "#{idx} @ {col_start},{row_start}: {width}x{height}".format(**self.__dict__)


def test_parse_input(input):
    for test in input:
        claim = Claim(test)
        assert str(claim) == test.strip(), "Mismatch {} !=  {}".format(str(claim), test)


def covered_by_claims(point, claims, min_claims=2):
    covered = 0
    row, col = point
    for claim in claims:
        if (row >= claim.row_start and
            col >= claim.col_start and
            row < claim.row_start + claim.height and
            col < claim.col_start + claim.width):
            covered += 1
        if covered >= min_claims:
            break

    return covered >= min_claims


def calculate_fabric_size(claims):
    rows = 0
    cols = 0
    for claim in claims:
        if claim.row_start + claim.height > rows:
            rows = claim.row_start + claim.height
        if claim.col_start + claim.width > cols:
            cols = claim.col_start + claim.width

    return rows, cols

def solve_part_1(claims):
    rows, cols = calculate_fabric_size(claims)
    covered_points = set()
    for r in range(rows):
        for c in range(cols):
            # if r % 10 == 0 and c == 0:
            #     print(r, c)
            if covered_by_claims((r, c), claims, min_claims=2):
                covered_points.add((r,c))
    return len(covered_points)


def visualize(claims_by_color, plot_size):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    rows, cols = plot_size
    _, ax = plt.subplots(1)
    ax.set_xlim(left=0, right=cols)
    ax.set_ylim(top=0, bottom=rows)
   
    def add_claim(claim, color='r'):
        rect = patches.Rectangle((claim.col_start, claim.row_start), claim.width, claim.height, linewidth=1, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
  
    for color, claims in claims_by_color:
        for claim in claims:
            add_claim(claim, color)
    
    plt.show()

def solve_part_2(claims):
    remaining = list(claims)
    no_overlap = set()
    overlapping = set()
    while len(remaining):
        claim = remaining.pop()
        for other in remaining:
            if claim.overlaps(other):
                overlapping.add(claim)
                overlapping.add(other)

        if claim not in overlapping:
            no_overlap.add(claim)

    answer = list(no_overlap)
    return answer


def load_input(filename="input"):
    with open(filename) as f:
        return f.readlines()

if __name__ == "__main__":
    test_input = ["#1 @ 1,3: 4x4",
                  "#2 @ 3,1: 4x4",
                  "#3 @ 5,5: 2x2"]

    test_claims = list(map(Claim, test_input))
    test_expected_answer = 4

    test_parse_input(test_input)

    input_lines = load_input()
    test_parse_input(input_lines)
    input = list(map(Claim, input_lines))

    test_answer = solve_part_1(list(map(Claim, test_input)))
    assert test_answer == test_expected_answer, "Expected {} got {}".format(test_expected_answer, test_answer)

    print("day3 part 1 solver beginning, wait several minutes.")
    answer = solve_part_1(input)
    assert answer == 119572, "incorrect answer {}".format(answer)
    print("day3 part 1 answer: There are {} square inches of fabric with at least two claims covering them.".format(len(answer)))

    test_answer = solve_part_2(test_claims)
    assert len(test_answer) == 1 and test_answer[0].idx == 3, "Expected claim 3 as answer from test data"

    test_overlapping = ["#932 @ 845,514: 10x15",
                        "#778 @ 833,497: 18x28"]
    test_overlapping = list(map(Claim, test_overlapping))
    assert test_overlapping[0].overlaps(test_overlapping[1])

    answer = solve_part_2(input)
    assert len(answer) == 1, "expected single answer, got {}".format(len(answer))
    assert answer[0].idx == 775, "correct answer is claim 775, found {}".format(answer[0])
    print("day3 part 2 answer: The only non-overlapping claim is {}".format(answer[0]))

    # plot_size = calculate_fabric_size(input)
    # claims_by_color = list()
    # claims_by_color.append(('b', input))
    # claims_by_color.append(('r', answer))
    # visualize(claims_by_color, plot_size)



