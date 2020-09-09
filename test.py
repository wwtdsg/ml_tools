"""
姓名：王滔
电话：185662661158
"""
import numpy as np
import random
def mean_dis(s1, s2):
    l1, l2 = len(s1), len(s2)
    dp = np.zeros((l1 + 1, l2 + 1))
    dp[0, range(1, l2 + 1)] = np.array(range(1, l2 + 1))
    dp[range(1, l1 + 1), 0] = np.array(range(1, l1 + 1))
    for i in range(1, l1 + 1):
        for j in range(1, l2 + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i, j] = dp[i - 1, j - 1]
            else:
                dp[i, j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + 1)
    return dp[l1, l2]


def rand1000():
    rel = 0
    for i in range(10):
        rel += (2 ** i) * random.randint(0, 2)
    if rel > 1000:
        return rand1000()
    return rel


class Te:
    def __init__(self):
        pass

    @classmethod
    def pp(cls):
        print('pp')

    @staticmethod
    def pps():
        print('pps')

    @property
    def lll(self):
        return 1



Te.pp()
Te.pps()

# rand1000()
# rooms = 'cdef'
# devs = 'abcc'
# print(mean_dis(rooms, devs))
