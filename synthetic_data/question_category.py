
from random import sample
import random




class QuestionCategoryBucket():
    def __init__(self, q_category, keywords, templates, q_type="open", seed=88):
        random.seed(seed)
        self.q_type = q_type
        self.keywords = keywords
        self.templates = templates
        self.q_category = q_category
    
    # Given keywords from ROCO caption, generate question. Return None if no keywords match
    def get_question(self, picture_keywords):
        questions = []
        answers = []
        for keyword in self.keywords:
            keyword = keyword.split()[0].lower()
        
            if keyword in picture_keywords:
                if self.q_type == "open":
                    questions.append(sample(self.templates, 1)[0])
                    answers.append(keyword)
                else: # Yes no question
                    if random.random() > 0.4: # Answer is yes
                        questions.append(sample(self.templates, 1)[0].format(keyword))
                        answers.append("yes")
                    else:
                        all_but_correct_ans = [x for x in self.keywords if x != keyword]
                        incorrect_ans = sample(all_but_correct_ans, 1)[0]
                        questions.append(sample(self.templates, 1)[0].format(incorrect_ans))
                        answers.append("no")
                

        if questions:
            return questions, answers
        

