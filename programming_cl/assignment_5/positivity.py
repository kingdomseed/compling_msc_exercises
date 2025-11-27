

def positivize(review: str):
    new_review = review
    new_review = negative_replacer(new_review)
    new_review = minute_replacer(new_review)
    return new_review


def change_time(time: int):
    new_time = time / 2
    return int(new_time)

def minute_replacer(review: str):
    words_split = review.split(" ")
    for word in words_split:
        if "minutes" in word:
            time_index = words_split.index(word)-1
            review = review.replace(words_split[time_index], "only "+ str(change_time(int(words_split.pop(time_index)))))
    return review

def negative_replacer(review: str):
    negatives = ["bad", "horrible", "dirty", "disgusting", "expensive", "moldy", "frozen", "minutes", "Bad", "Horrible", "Dirty", "Disgusting", "Expensive", "Moldy", "Frozen", "Minutes"]
    positives = ["good", "fantastic", "clean", "sublime", "affordable", "flavourful", "farm-fresh", "minutes", "Good", "Fantastic", "Clean", "Sublime", "Affordable", "Flavourful", "Farm-fresh", "Minutes"]
    for word in negatives:
        review = review.replace(word, positives[negatives.index(word)])
    return review

review_1 = "The food was horrible!!! We waited 40 minutes for frozen vegetables and moldy bread. Disgusting!"
review_2 = "I waited for maybe 56 minutes . And the toilette was horrible ."
print(positivize(review_1))
print(positivize(review_2))