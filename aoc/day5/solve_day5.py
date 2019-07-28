from string import ascii_lowercase

test_input = [
			   ("dabAcCaCBAcCcaDA", "dabCBAcaDA"),
			  ]
test_input_part_2 = [
			   ("dabAcCaCBAcCcaDA", "a", "dbCBcD"),
			   ("dabAcCaCBAcCcaDA", "b", "daCAcaDA"),
			   ("dabAcCaCBAcCcaDA", "c", "daDA"),
			   ("dabAcCaCBAcCcaDA", "d", "abCBAc"),
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
		i = 0
		while i < len(input) - 1:
			if react(input[i], input[i+1]):
				progress = True
				input.pop(i)
				input.pop(i)
				# Heuristic: the reaction might have opened a new possibility one to the left
				if i > 0:
					i -= 1
			else:
				i += 1

	return "".join(input)


def solve_day_5_part_2(input_str):
	answers = {}
	min_length = 2**32
	min_c = None
	for c in ascii_lowercase:
		modified_input = remove_char(input_str, c)
		answer = solve_day_5_part_1(modified_input)
		answers[c] = answer
		if len(answer) < min_length:
			min_length = len(answer)
			min_c = c
	return min_c, answers[min_c]


def remove_char(string, char):
	return string.replace(char.lower(), "").replace(char.upper(), "")


if __name__ == "__main__":
	for t, a in test_input:
		result = solve_day_5_part_1(t)
		assert a == result, "expected {} actual {}".format(a, result)

	real_input = load_input("input")
	answer = solve_day_5_part_1(real_input)
	print("Part 1: Original Polymer length {}, after reaction {}".format(len(real_input), len(answer)))
	# Part 1: Original Polymer length 50000, after reaction 10762

	for input, c, a in test_input_part_2:
		result = solve_day_5_part_1(remove_char(input, c))
		assert a == result, "Original {} removed {} expected {} actual {}".format(input, c, a, result)

	min_c, answer = solve_day_5_part_2(real_input)
	print("Part 2: After removing {}, reaction yields length {}".format(min_c, len(answer)))
	# Part 2: After removing m, reaction yields length 6946