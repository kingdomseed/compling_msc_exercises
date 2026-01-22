import csv


def find_happiest(file):
    happiest_per_continent = {}
    """Reads a file containing names and happiness scores,
    and returns the name
    with the highest happiness score.

    file format: csv
    example: 1,Afghanistan,3.632,Asia

    Args:
        file (str): The path to the file containing names
        and happiness scores.

    Returns:
        dict: The most happy country per continent.
        output format:
        {
        'Asia ': 'Israel ',
        'Europe': 'Switzerland',
        'North America': 'Canada',
        ... }
    """
    with open(file, newline="", encoding='utf-8', ) as happiness_report:
        rows = list(csv.reader(happiness_report))  # rows: list[list[str]]
        for row in rows:  # row: list[str]
            continent = row[3]  # continent_name
            country = row[1]  # country_name
            score = float(row[2])  # score
            if continent not in happiest_per_continent:
                happiest_per_continent[continent] = (score, country)
            elif score > happiest_per_continent[continent][0]:
                happiest_per_continent[continent] = (score, country)
        print(happiest_per_continent)
        sorted_results = sorted(
            happiest_per_continent.items(),
            key=lambda x: x[1][0],
            reverse=True
        )
        print(sorted_results)
    return sorted_results
