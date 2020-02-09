def solve_pt1():
    nums = []
    with open('input') as f:
        for line in f:
            nums.append(int(line))

    fuel = 0
    for n in nums:
        fuel += (n // 3 - 2)

    print(fuel)

def solve_pt2():
    nums = []
    with open('input_pt2') as f:
        for line in f:
            nums.append(int(line))

    def fuel_for_module(module_mass):
        fuel = 0
        fuel_inc = module_mass
        while fuel_inc > 0:
            fuel_inc = fuel_inc // 3 - 2
            if fuel_inc > 0:
                fuel += fuel_inc
        return fuel


    fuel = 0
    for n in nums:
        fuel += fuel_for_module(n)

    print(fuel)


if __name__ == "__main__":
    solve_pt2()
