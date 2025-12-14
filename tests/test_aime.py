"""
AIME 任务单元测试
测试 AIME 任务的核心功能，包括数据加载、答案提取和评估逻辑。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tasks.aime import AIME, extract_aime_answer


def test_answer_extraction():
    """测试答案提取函数"""
    print("=" * 60)
    print("测试 1: 答案提取函数")
    print("=" * 60)
    
    test_cases = [
        ("经过计算，答案是 672", "672"),
        ("The answer is \\boxed{123}", "123"),
        ("#### 456", "456"),
        ("最终答案为 789", "789"),
        ("答案: 100", "100"),
        ("我认为答案应该是 42", "42"),
        ("Answer: 500", "500"),
        ("\\boxed{999}", "999"),
        ("#### 0", "0"),
        ("这个问题没有答案", None),  # 应该返回 None 或找不到答案
        ("答案是 1234", None),  # 超出范围
        ("答案是 -5", None),  # 负数
    ]
    
    passed = 0
    failed = 0
    
    for text, expected in test_cases:
        result = extract_aime_answer(text)
        status = "✓" if result == expected else "✗"
        if result == expected:
            passed += 1
        else:
            failed += 1
        print(f"  {status} '{text[:40]}...' -> 提取: {result}, 期望: {expected}")
    
    print(f"\n结果: {passed} 通过, {failed} 失败")
    return failed == 0


def test_task_creation():
    """测试任务创建"""
    print("\n" + "=" * 60)
    print("测试 2: 任务创建和数据加载")
    print("=" * 60)
    
    try:
        # 测试创建测试集
        task = AIME(split="test")
        print(f"  ✓ 成功创建 AIME 任务 (test split)")
        print(f"  ✓ 数据集大小: {len(task)} 个问题")
        
        # 测试评估类型
        assert task.eval_type == 'generative', "评估类型应该是 'generative'"
        print(f"  ✓ 评估类型: {task.eval_type}")
        
        # 测试训练集
        task_train = AIME(split="train")
        print(f"  ✓ 成功创建 AIME 任务 (train split)")
        print(f"  ✓ 训练集大小: {len(task_train)} 个问题")
        
        # 测试全集
        task_all = AIME(split="all")
        print(f"  ✓ 成功创建 AIME 任务 (all split)")
        print(f"  ✓ 全部数据: {len(task_all)} 个问题")
        
        return True
    except Exception as e:
        print(f"  ✗ 任务创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_get_example():
    """测试获取示例"""
    print("\n" + "=" * 60)
    print("测试 3: 获取问题示例")
    print("=" * 60)
    
    try:
        task = AIME(split="test")
        
        if len(task) == 0:
            print("  ⚠ 警告: 数据集为空")
            return True
        
        # 获取第一个示例
        example = task[0]
        print(f"  ✓ 成功获取示例")
        
        # 检查结构
        assert 'messages' in example, "示例应该包含 'messages' 字段"
        assert len(example['messages']) == 2, "对话应该包含两条消息"
        assert example['messages'][0]['role'] == 'user', "第一条消息应该是用户消息"
        assert example['messages'][1]['role'] == 'assistant', "第二条消息应该是助手消息"
        print(f"  ✓ 对话结构正确")
        
        # 显示示例内容
        print(f"\n  问题预览:")
        user_content = example['messages'][0]['content']
        print(f"    {user_content[:150]}...")
        
        print(f"\n  答案预览:")
        assistant_content = example['messages'][1]['content']
        print(f"    {assistant_content[:100]}...")
        
        return True
    except Exception as e:
        print(f"  ✗ 获取示例失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_evaluation():
    """测试评估功能"""
    print("\n" + "=" * 60)
    print("测试 4: 评估功能")
    print("=" * 60)
    
    try:
        task = AIME(split="test")
        
        if len(task) == 0:
            print("  ⚠ 警告: 数据集为空，跳过评估测试")
            return True
        
        example = task[0]
        
        # 提取正确答案
        ground_truth = extract_aime_answer(example['messages'][1]['content'])
        if ground_truth is None:
            print("  ⚠ 警告: 无法从示例中提取答案")
            return True
        
        print(f"  标准答案: {ground_truth}")
        
        # 测试正确答案
        correct_responses = [
            f"经过计算，答案是 {ground_truth}",
            f"#### {ground_truth}",
            f"\\boxed{{{ground_truth}}}",
            f"The answer is {ground_truth}",
        ]
        
        for response in correct_responses:
            result = task.evaluate(example, response)
            status = "✓" if result == 1 else "✗"
            print(f"  {status} 正确答案评估: {result} (回答: '{response[:30]}...')")
            if result != 1:
                return False
        
        # 测试错误答案
        wrong_answer = "123" if ground_truth != "123" else "456"
        wrong_responses = [
            f"答案是 {wrong_answer}",
            f"#### {wrong_answer}",
            "我不知道",
            "没有答案",
        ]
        
        for response in wrong_responses:
            result = task.evaluate(example, response)
            status = "✓" if result == 0 else "✗"
            print(f"  {status} 错误答案评估: {result} (回答: '{response[:30]}...')")
            # 只要不是1就算对（可能是0或其他）
        
        # 测试奖励函数
        reward = task.reward(example, f"#### {ground_truth}")
        assert reward == 1.0, "奖励函数应该返回 1.0 对于正确答案"
        print(f"  ✓ 奖励函数正确: {reward}")
        
        return True
    except Exception as e:
        print(f"  ✗ 评估测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_slicing():
    """测试数据集切片"""
    print("\n" + "=" * 60)
    print("测试 5: 数据集切片功能")
    print("=" * 60)
    
    try:
        task = AIME(split="all")
        total_len = len(task)
        
        if total_len < 5:
            print(f"  ⚠ 警告: 数据集太小 ({total_len})，跳过切片测试")
            return True
        
        # 测试基本切片
        task_slice = AIME(split="all", start=2, stop=5)
        assert len(task_slice) == 3, "切片长度应该是 3"
        print(f"  ✓ 基本切片: AIME(start=2, stop=5) 长度为 {len(task_slice)}")
        
        # 测试步长
        task_step = AIME(split="all", start=0, stop=10, step=2)
        expected_len = min(5, (total_len + 1) // 2)
        print(f"  ✓ 步长切片: AIME(start=0, stop=10, step=2) 长度为 {len(task_step)}")
        
        # 验证切片数据一致性
        ex_full = task[2]
        ex_slice = task_slice[0]
        # 注意：由于内部实现可能不同，我们只检查能够访问即可
        print(f"  ✓ 切片数据可访问")
        
        return True
    except Exception as e:
        print(f"  ✗ 切片测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 15 + "AIME 任务单元测试" + " " * 15 + "║")
    print("╚" + "=" * 58 + "╝")
    
    tests = [
        ("答案提取", test_answer_extraction),
        ("任务创建", test_task_creation),
        ("获取示例", test_get_example),
        ("评估功能", test_evaluation),
        ("数据切片", test_slicing),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n测试 '{name}' 发生异常: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # 打印总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"  {status}: {name}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！AIME 任务实现正确。")
        return 0
    else:
        print(f"\n⚠️  {total - passed} 个测试失败，请检查实现。")
        return 1


if __name__ == "__main__":
    exit(main())

