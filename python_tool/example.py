from tool import *

naiveBayesClassifier = MultinomialNaiveBayesClassifier()
data = [["love", "happy", "joy", "joy", "love"],
        ["happy", "love", "kick", "joy", "happy"],
        ["love", "move", "joy", "good"],
        ["love", "happy", "joy", "pain", "love"],
        ["joy", "love", "pain", "kick", "pain"],
        ["pain", "pain", "love", "kick"]]
label = ["Yes", "Yes", "Yes", "Yes", "No", "No"]
value = ["love", "pain", "joy", "happy", "kick", "happy"]

naiveBayesClassifier.fit(np.array(data, dtype=object), np.array(label, dtype=object))
print(naiveBayesClassifier.predict(value))