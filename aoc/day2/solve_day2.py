from ..util import load_input

if __name__ == "__main__":
    filename="input"
    input = load_input(filename)
    answer = solve_part_1(input)
    print("Part 1: The frequency is {}".format(answer))
    
    test_part_2()


