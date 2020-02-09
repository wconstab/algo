test_input = """
1, 1
1, 6
8, 3
3, 4
5, 5
8, 9
"""

def load_input_str(str):
	input = list()
	for coord_str in str.strip().split("\n"):
		sep = coord_str.find(',')
		x = int(coord_str[0:sep])
		y = int(coord_str[sep + 1:len(coord_str)])
		input.append((x, y))
	return input

def solve_day_6(input):
	max_x = -2**16
	max_y = -2**16
	min_x = 2**16-1
	min_y = 2**16-1

	for x, y in input:
		if x < min_x:
			min_x = x
		if y < min_y:
			min_y = y
		if x > max_x:
			max_x = x
		if y > max_y:
			max_y = y

	rows = max_y - min_y + 1
	cols = max_x - min_x + 1
	grid = [['.' for c in range(cols)] for r in range(rows)]

	for row in grid:
		print("".join(row))


if __name__ == "__main__":
	input = load_input_str(test_input)
	print(input)
	answer = solve_day_6(input)

