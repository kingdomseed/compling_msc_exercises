from pathlib import Path

from happy import find_happiest


def test_toy_happiness_europe_only():
    here = Path(__file__).resolve().parent
    toy_file = here / "toy_happiness.txt"

    result = find_happiest(str(toy_file))

    # result is a list of tuples: [(continent, (score, country)), ...]
    assert result == [("Europe", (7.537, "Norway"))]


if __name__ == "__main__":
    test_toy_happiness_europe_only()
    print("OK - toy test passed")
