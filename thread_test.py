from collections import Counter
from typing import List


class Solution:
    def find_word(self, s, word):
        beg = 0
        idxs = []
        while beg < len(s) - self.substr_len + 1:
            idx = s.find(word, beg)
            if idx == -1:
                return idxs
            else:
                beg = idx + 1
                idxs.append(idx)
        return idxs

    def findSubstring(self, s: str, words: List[str]) -> List[int]:
        return [i for i in range(len(s) - len(words) * len(words[0]) + 1)
                if Counter(
                [s[a:a + len(words[0])] for a in range(i, i + len(words) * len(words[0]), len(words[0]))]) == Counter(
                words)] if words else []

    def nextPermutation(self, nums: List[int]) -> None:
        for end_idx in range(len(nums) - 1, 0, -1):
            s_idx = end_idx - 1
            while s_idx >= 0:
                if nums[s_idx] >= nums[end_idx]:
                    s_idx -= 1
                else:
                    tmp = nums[end_idx]
                    nums[end_idx] = nums[s_idx]
                    nums[s_idx] = tmp
                    return
        for s_idx in range(0, len(nums) // 2):
            tmp = nums[s_idx]
            nums[s_idx] = nums[len(nums) - 1 - s_idx]
            nums[len(nums) - 1 - s_idx] = tmp

    def search(self, nums: List[int], target: int) -> int:
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                return mid
            if nums[left] <= nums[mid]:
                if target > nums[mid] or target < nums[left]:
                    left = mid + 1
                else:
                    right = mid - 1
            else:
                if target > nums[right] or target < nums[mid]:
                    right = mid - 1
                else:
                    left = mid + 1
        return -1

    def searchRange(self, nums: List[int], target: int) -> List[int]:
        left, right = 0, len(nums) - 1
        left_find, right_find = -1, -1
        while left <= right:
            mid = (left + right) // 2
            # 先找右边界
            if nums[mid] == target and (mid == len(nums) - 1 or nums[mid + 1] > target):
                right_find = mid
                break
            if nums[mid] <= target:
                left = mid + 1
            else:
                right = mid - 1
        else:
            return [-1, -1]
        left, right = 0, right_find - 1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target and (mid == 0 or nums[mid - 1] < target):
                left_find = mid
                break
            if nums[mid] >= target:
                right = mid - 1
            else:
                left = mid + 1
        return [left_find if left_find != -1 else right_find, right_find]

    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        candidates = sorted(candidates)
        ans = []

        def dfs(left, trace, self_idx):
            if left == 0:
                ans.append(trace)
            else:
                for i, can in enumerate(candidates[self_idx:]):
                    if left < can:
                        break
                    else:
                        dfs(left - can, trace + [can], self_idx + i)
        dfs(target, [], 0)
        # ans = [tuple(sorted(x)) for x in ans]
        return ans

    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        candidates = sorted(candidates)
        ans = []

        def dfs(left, trace, self_idx):
            if left == 0:
                ans.append(trace)
            else:
                for i, can in enumerate(candidates[self_idx:]):
                    if left < can:
                        break
                    else:
                        dfs(left - can, trace + [can], self_idx + i + 1)
        dfs(target, [], 0)
        ans = set([tuple(sorted(x)) for x in ans])
        return list(ans)

    def trap(self, height: List[int]) -> int:
        if len(height) == 0:
            return 0
        left = [height[0]]
        right = [height[-1]]
        result = 0

        for i in range(1, len(height)):
            left.append(max(left[-1], height[i]))

        for i in range(len(height) - 2, -1, -1):
            right.append(max(right[-1], height[i]))
        right = right[::-1]

        for i in range(len(height)):
            result += min(left[i], right[i]) - height[i]

        return result

    def jump(self, nums: List[int]) -> int:
        lenth = len(nums)
        prob_nums = []
        for i in range(lenth - 2, 0, -1):
            if nums[i] + i >= lenth - 1:
                return self.jump(nums[:-1]) + 1

    def myPow(self, x: float, n: int) -> float:
        def pow(N):
            if N == 0:
                return 1
            y = pow(N // 2)
            if N % 2 == 1:
                return y * y * x
            else:
                return y * y

        return pow(n) if n >= 0 else 1.0 / pow(-n)

    def solveNQueens(self, n: int) -> List[List[str]]:
        valid_matrix = [[True] * n for _ in range(n)]
        # def choose_one(matrix, col, vol):
        valid_pos = []
        for i in range(n):
            for j in range(n):
                if valid_matrix[i][j] == True:
                    valid_pos.append(j)
                    valid_matrix

m = Solution()
nums = []
rel = m.combinationSum2(nums, 5)
print(rel)
