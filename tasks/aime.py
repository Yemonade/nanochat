"""
AIME 2024/2025 评估任务
AIME (American Invitational Mathematics Examination) 是美国高中数学竞赛的一部分。
问题难度较高，答案为 0-999 之间的整数。

数据来源: https://huggingface.co/datasets/ZT20241128/AIME_2024_2025
"""

import re
from datasets import load_dataset
from tasks.common import Task


def extract_aime_answer(text):
    """
    从 AIME 问题或回答中提取数字答案。
    AIME 答案通常是 0-999 之间的整数。
    
    支持的格式:
    - "答案是 123"
    - "最终答案: 456"
    - "#### 789" (类似 GSM8K 格式)
    - "\\boxed{123}" (LaTeX 格式)
    - 纯数字 "123"
    """
    if not text:
        return None
    
    # 尝试匹配 #### 标记 (类似 GSM8K)
    match = re.search(r'####\s*(\d+)', text)
    if match:
        num = match.group(1)
        if num.isdigit() and 0 <= int(num) <= 999:
            return num
    
    # 尝试匹配 LaTeX boxed 格式
    match = re.search(r'\\boxed\{(\d+)\}', text)
    if match:
        num = match.group(1)
        if num.isdigit() and 0 <= int(num) <= 999:
            return num
    
    # 尝试匹配 "答案是/为" 或 "answer is" 格式
    match = re.search(r'(?:答案|Answer|answer)(?:是|为|:|is|:)\s*(\d+)', text, re.IGNORECASE)
    if match:
        num = match.group(1)
        if num.isdigit() and 0 <= int(num) <= 999:
            return num
    
    # 尝试匹配最后出现的 0-999 范围内的数字
    matches = re.findall(r'\b(\d+)\b', text)
    for num in reversed(matches):  # 从后往前找
        if num.isdigit() and 0 <= int(num) <= 999:
            return num
    
    return None


class AIME(Task):
    """
    AIME 数学竞赛评估任务。
    
    特点:
    - 高难度数学问题
    - 答案为 0-999 之间的整数
    - 需要复杂推理和计算能力
    - 评估类型: generative (生成式)
    """

    def __init__(self, split="test", **kwargs):
        super().__init__(**kwargs)
        assert split in ["train", "test", "all"], "AIME split must be train|test|all"
        
        # 加载数据集
        # 使用 qq8933/AIME_1983_2024 数据集
        try:
            ds = load_dataset("ZT20241128/AIME_2024_2025", split="train")
            
            # 根据年份筛选 2024/2025 的数据
            # 假设数据集有 'year' 字段，我们筛选 2024 和 2025
            filtered_data = []
            for item in ds:
                year = item.get('year')
                if year and int(year) >= 2024:
                    filtered_data.append(item)
            
            # 如果没有找到 2024/2025 的数据，则使用最新的数据
            if len(filtered_data) == 0:
                # 使用全部数据，按年份排序并取最新的 30 题（典型的 AIME 一年约 15 题 x 2 轮）
                sorted_data = sorted(ds, key=lambda x: (x.get('year', 0), x.get('problem_number', 0)), reverse=True)
                filtered_data = sorted_data[:30]
            
            # 划分训练集和测试集 (80/20 split)
            split_idx = int(len(filtered_data) * 0.8)
            if split == "train":
                self.ds = filtered_data[:split_idx]
            elif split == "test":
                self.ds = filtered_data[split_idx:]
            else:  # all
                self.ds = filtered_data
                
        except Exception as e:
            # 如果加载失败，使用备用方案：创建示例数据
            print(f"警告: 无法加载 AIME 数据集 ({e})，使用示例数据")
            
            

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        """
        获取单个 AIME 问题。
        """
        row = self.ds[index]
        
        # 提取问题和答案
        question = row.get('problem', '')
        answer = str(row.get('answer', ''))
        solution = row.get('solution', '')
        
        
        # 构建用户消息
        user_message = question + "\n\nYou must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>. And your final answer will be extracted automatically by the \\boxed{} tag."
        
        # 构建模型消息 (ground truth)
        if solution:
            assistant_content = f"{solution}\n\\boxed{{answer}}"
        else:
            assistant_content = f"\\boxed{answer}"
        
        # 创建对话
        messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_content}
        ]
        
        conversation = {
            "messages": messages,
            "expected_answer": answer,
        }
        
        return conversation

    def evaluate(self, conversation, assistant_response):
        """
        评估模型回答是否正确。
        
        参数:
            conversation: 包含问题和标准答案的对话
            assistant_response: 模型生成的回答 (字符串)
        
        返回:
            1 如果正确，0 如果错误
        """
        assert isinstance(assistant_response, str), "Expected string response"
        
        # 从标准答案中提取正确答案
        assistant_message = conversation['messages'][-1]
        ground_truth_text = assistant_message['content']
        ground_truth = extract_aime_answer(ground_truth_text)
        
        # 也可以从 conversation 中直接获取
        if ground_truth is None and 'expected_answer' in conversation:
            ground_truth = str(conversation['expected_answer'])
        
        # 从模型回答中提取答案
        predicted = extract_aime_answer(assistant_response)
        
        # 比较答案
        if ground_truth and predicted:
            is_correct = (predicted == ground_truth)
            return int(is_correct)
        
        # 如果无法提取答案，则判定为错误
        return 0

    def reward(self, conversation, assistant_response):
        """
        用于强化学习的奖励函数。
        """
        is_correct = self.evaluate(conversation, assistant_response)
        return float(is_correct)


# 测试代码
if __name__ == "__main__":
    # 测试 AIME 任务
    print("测试 AIME 任务实现...")
    
    # 创建任务对象
    task = AIME(split="test")
    print(f"数据集大小: {len(task)}")
    
    # 测试获取示例
    if len(task) > 0:
        example = task[0]
        print("\n示例问题:")
        print(example['messages'][0]['content'])
        print("\n标准答案:")
        print(example['messages'][1]['content'])
        
        # 测试答案提取
        print("\n测试答案提取:")
        test_cases = [
            "经过计算，答案是 672",
            "The answer is \\boxed{123}",
            "#### 456",
            "最终答案为 789",
            "答案: 100",
        ]
        for test in test_cases:
            extracted = extract_aime_answer(test)
            print(f"  '{test}' -> {extracted}")
        
        # 测试评估函数
        print("\n测试评估函数:")
        correct_response = "经过推理，答案是 672 #### 672"
        wrong_response = "我认为答案是 123 #### 123"
        
        result1 = task.evaluate(example, correct_response)
        result2 = task.evaluate(example, wrong_response)
        print(f"  正确答案评估结果: {result1}")
        print(f"  错误答案评估结果: {result2}")
    
    print("\n✓ AIME 任务实现测试完成")
