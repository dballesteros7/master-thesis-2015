
def load_csv_data(filename):
    with open(filename, 'r') as input_file:
        return [[int(item) for item in line.strip().split(',')]
                for line in input_file]
