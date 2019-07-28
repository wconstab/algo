test_input = [
			   ("dabAcCaCBAcCcaDA", "dabCBAcaDA"),
			  ]

def load_input(name):
	with open(name) as f:
		input = f.read()
		return input


def react(a, b):
	return a.lower() == b.lower() and a != b


def solve_day_5_part_1(input_str):
	progress = True
	input = list(input_str)


	while progress:
		progress = False
		for i in range(len(input)-1):
			if react(input[i], input[i+1]):
				progress = True
				input.pop(i)
				input.pop(i)
				break

	return "".join(input)


if __name__ == "__main__":
	for t, a in test_input:
		result = solve_day_5(t)
		assert a == result, "expected {} actual {}".format(a, result)

	real_input = load_input("input")
	answer = solve_day_5_part_1(real_input)
	print("Original Polymer length {}, after reaction {}".format(len(real_input), len(answer)))
	# Original Polymer length 50000, after reaction 10762
