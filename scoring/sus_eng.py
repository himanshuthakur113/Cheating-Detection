import time
from scoring.sus_rule import points

class SusEng:
    def __init__(self):
        self.score = 0
        self.last_happen = time.time()

    def add_score(self, action):
        if action in points:
            self.score += points[action]
    
        self.score = min(self.score, 100)

    def decay_score(self):
        now = time.time()
        if now - self.last_happen > 1:
            self.score = max(self.score - 1, 0)
            self.last_happen = now

    def get_score(self):
        return int(self.score)
            