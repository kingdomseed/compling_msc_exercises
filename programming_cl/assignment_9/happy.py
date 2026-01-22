import csv


def find_happiest(file):
    happiest_per_continent = {}
    """Reads a file containing names and happiness scores, and returns the name
    with the highest happiness score.

    file format: csv
    headers: id,country_name,score,continent_name
    example: 1,Afghanistan,3.632,Asia

    Args:
        file (str): The path to the file containing names and happiness scores.

    Returns:
        dict: The most happy country per continent.
    """
    with open(file, newline="", encoding='utf-8', ) as happiness_report:
        rows = list(csv.reader(happiness_report))  # rows: list[list[str]]
        for row in rows:
            continent = row[3]
            country = row[1]
            score = float(row[2])
            if continent not in happiest_per_continent:
                happiest_per_continent[continent] = (country, score)
            elif score > happiest_per_continent[continent][1]:
                happiest_per_continent[continent] = (country, score)

    return happiest_per_continent
