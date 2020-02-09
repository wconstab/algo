import argparse

OP_ADD  = 1
OP_MUL  = 2
OP_HALT = 99 
    
class Day2(object):
    def __init__(self):
        pass

    def solve_part1(self, filename):
        
        def parse_csv_codes(filename):
            with open(filename) as f:
                x = f.readline()
                return map(int, x.split(','))

        codes = parse_csv_codes(filename)
        print(codes)
        
        pc = 0
        while codes[pc] != OP_HALT:
            a_idx, b_idx, c_idx = codes[pc+1], codes[pc+2], codes[pc+3]
            
            if codes[pc] == OP_ADD:
                codes[c_idx] = codes[a_idx] + codes[b_idx]
            elif codes[pc] == OP_MUL:
                codes[c_idx] = codes[a_idx] * codes[b_idx]
            else:
                assert False, "Bug"

            pc += 4
        return codes

    def solve_part2(self, filename):
        def parse_csv_codes(filename):
            with open(filename) as f:
                x = f.readline()
                return map(int, x.split(','))

        codes = parse_csv_codes(filename)
        codes[1] = "X"
        codes[2] = "Y"
        print(codes)
        
        pc = 0
        while codes[pc] != OP_HALT:
            a_idx, b_idx, c_idx = codes[pc+1], codes[pc+2], codes[pc+3]

            if codes[pc] == OP_ADD:
                if isinstance(a_idx, str) or isinstance(b_idx, str):
                    c = "(" + str(a_idx) + ") + (" + str(b_idx) + ")"
                else:
                    a, b = codes[a_idx], codes[b_idx]
                    c = a + b
                codes[c_idx] = c
            elif codes[pc] == OP_MUL:
                if isinstance(a_idx, str) or isinstance(b_idx, str):
                    c = "(" + str(a_idx) + ") * (" + str(b_idx) + ")"
                else:
                    a, b = codes[a_idx], codes[b_idx]
                    c = a * b
                codes[c_idx] = c
            else:
                assert False, "Bug"

            pc += 4
        return codes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("part", type=int)
    parser.add_argument("input")
    args = parser.parse_args()

    solver = Day2()

    if args.part == 1:
        print(solver.solve_part1(args.input))

    elif args.part == 2:
        solver.solve_part2(args.input)
