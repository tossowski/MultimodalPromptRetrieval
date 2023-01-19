
from question_category import QuestionCategoryBucket

from random import sample
import random





class SpecificQuestionCategoryBucket(QuestionCategoryBucket):
    def __init__(self, required_words, q_category, keywords, templates, q_type="open", seed=88):
       super().__init__(q_category, keywords, templates, q_type, seed)
       self.required_words = required_words
    
    # Given keywords from ROCO caption, generate question. Return None if no keywords match
    def get_question(self, picture_keywords):
        questions = []
        answers = []
        for keyword in self.keywords:
            keyword = keyword.split()[0].lower()
            
            if keyword in picture_keywords:
                has_required_word = None
                for required_word in self.required_words:
                    if required_word in picture_keywords:
                        has_required_word = required_word
                if not has_required_word: # Has the shape like irregular/oval, but not the correct organ
                    continue
                if self.q_type == "open":
                    questions.append(sample(self.templates, 1)[0].format(required_word))
                    answers.append(keyword)
                

        if questions:
            return questions, answers
        

