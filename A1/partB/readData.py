# Read all lines from breast cancer data file (contains newline char)
def read(option='all'):
    """
    Reads and returns data from the data file, specifically the data
    presenting in the option argument.

    :return: List of data elements, including classification value (first in the list), or None if data file is empty.
    :rtype: list or None
    :param option: either 'all', or a list containing a combination of: 'mean', 'stdev', 'largest'.
    :type option: string or list[string]
    """
    assert 'all' in option \
           or 'mean' in option \
           or 'stdev' in option \
           or 'largest' in option

    data = []
    with open('data/wdbc.data', 'r') as f:
        strs = f.readlines()

    if len(strs) == 0:
        return None

    # Define the indexes to collect from
    collection = []
    if 'all' in option:
        collection += [*range(1, len(strs[0]))]
    if 'mean' in option:
        collection += [1] + [*range(2, 12)]
    if 'stdev' in option:
        collection += [1] + [*range(12, 22)]
    if 'largest' in option:
        collection += [1] + [*range(22, 32)]

    # Remove duplicates from list by converting to dict, then back to list
    collection = list(dict.fromkeys(collection))

    # Sort the list in ascending order
    collection.sort()

    # Collect the data, using the collection indexes
    for line in strs:
        lineArr = line.strip().split(',')

        dataBuilder = []
        for i in range(len(lineArr)):
            if i in collection:
                # When building data, try to convert numbers to their float vals.
                # If you can't it's likely a char (classification), so just include the str.
                try:
                    dataBuilder += [float(lineArr[i])]
                except ValueError:
                    dataBuilder += [lineArr[i]]
        data.append(dataBuilder)
    classification = [x[0] for x in data]
    attributes = [x[1:] for x in data]
    return classification, attributes


def countClassification(data, classification):
    count = 0
    for entry in data:
        if entry[0] == classification:
            count += 1
    return count


if __name__ == "__main__":
    data = read()
    print(countClassification(data, 'M'), len(data))
