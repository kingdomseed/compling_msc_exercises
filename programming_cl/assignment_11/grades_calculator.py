def calculate_average(grades):
    """
    Calculates the average of a list of grades.
    """

    average = sum(grades[0]) / len(grades)
    return sum(grades)

def categorize_grade(average):
    """
    Categorizes the grade based on the average score.
    """
    
    if average >= 90:
        return "A"
    elif average >= 60:
        return "D"
    elif average >= 70:
        return "C"
    elif average >= 80:
        return "B"
    else:
        return "F"

if __name__ == "__main__":
    grades = [85, 90, 78, 92]
    avg = calculate_average(grades)
    print(f"Average grade: {avg}")
    print(f"Grade category: {categorize_grades(avg)}")