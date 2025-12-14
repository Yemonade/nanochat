"""
使用强模型生成安全回复，构建 Safety SFT 数据集

流程：
1. 读取上一步生成的有毒 prompts
2. 使用强模型为每个 prompt 生成安全的拒绝回复
3. 组成对话 pairs，格式兼容现有 SFT pipeline
4. 保存为 JSONL 格式
"""

import os
import sys
import json
import requests
import random
from typing import List, Dict, Tuple
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from nanochat.common import get_base_dir


# API 配置
api_key = open("openroutertoken.txt", "r", encoding="utf-8").read().strip()
url = "https://openrouter.ai/api/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}


# 安全回复生成的系统提示
SAFETY_RESPONSE_PROMPT = """你是一个负责任的 AI 助手，需要对用户的请求做出安全、适当的回应。

对于有害、不当或违反道德规范的请求，你需要：
1. 礼貌但坚定地拒绝
2. 简要说明为什么不能提供帮助
3. 在合适的情况下，提供建设性的替代建议
4. 保持专业、尊重的语气
5. 避免说教或居高临下

回复要求：
- 长度：2-4 句话
- 语气：坚定但友善
- 避免使用"作为 AI"、"我不能"等重复性开头（多样化表达）
- 根据请求的具体内容定制回复

现在，请为以下用户请求生成一个安全、适当的拒绝回复：

用户请求：{user_request}

请只输出你的回复，不要包含其他解释。"""


def load_toxic_prompts(prompts_file: str) -> List[Tuple[str, str]]:
    """
    加载有毒 prompts
    
    参数:
        prompts_file: prompts 文件路径
    
    返回:
        prompts: [(attack_type, prompt), ...]
    """
    print(f"正在加载有毒 prompts: {prompts_file}")
    
    prompts = []
    with open(prompts_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # 提取攻击类型和 prompt
            if line.startswith('[') and ']' in line:
                attack_type = line[1:line.index(']')]
                prompt = line[line.index(']')+1:].strip()
            else:
                attack_type = "unknown"
                prompt = line
            
            prompts.append((attack_type, prompt))
    
    print(f"✓ 加载 {len(prompts)} 条有毒 prompts")
    return prompts


def generate_safety_response(
    user_request: str,
    model: str = "openai/gpt-4o",
    temperature: float = 0.7
) -> str:
    """
    使用强模型生成安全回复
    
    参数:
        user_request: 用户的有害请求
        model: 使用的模型
        temperature: 温度参数
    
    返回:
        response: 安全回复
    """
    try:
        # 构建请求
        system_prompt = SAFETY_RESPONSE_PROMPT.format(user_request=user_request)
        
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": system_prompt}
            ],
            "temperature": temperature,
            "max_tokens": 200
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        result = response.json()
        
        # 提取回复
        safety_response = result['choices'][0]['message']['content'].strip()
        
        return safety_response
        
    except Exception as e:
        print(f"  ✗ 生成失败: {e}")
        return None


def generate_safety_pair(
    attack_type: str,
    user_request: str,
    model: str = "openai/gpt-4o"
) -> Dict:
    """
    生成一个完整的 safety 对话 pair
    
    参数:
        attack_type: 攻击类型
        user_request: 用户请求
        model: 使用的模型
    
    返回:
        conversation: 对话字典
    """
    # 生成安全回复
    safety_response = generate_safety_response(user_request, model)
    
    if not safety_response:
        return None
    
    # 构建对话格式（兼容 CustomJSON）
    conversation = {
        "messages": [
            {"role": "user", "content": user_request},
            {"role": "assistant", "content": safety_response}
        ],
        "metadata": {
            "attack_type": attack_type,
            "source": "safety_pipeline"
        }
    }
    
    return conversation


def generate_safety_dataset(
    toxic_prompts: List[Tuple[str, str]],
    output_file: str,
    model: str = "openai/gpt-4o",
    num_workers: int = 4,
    max_samples: int = None
):
    """
    批量生成 safety 数据集
    
    参数:
        toxic_prompts: 有毒 prompts 列表
        output_file: 输出文件
        model: 使用的模型
        num_workers: 并行线程数
        max_samples: 最多生成多少条（None = 全部）
    """
    print("\n" + "=" * 80)
    print("生成 Safety SFT 数据集")
    print("=" * 80)
    print(f"模型: {model}")
    print(f"并行线程: {num_workers}")
    print()
    
    # 限制数量
    if max_samples:
        toxic_prompts = toxic_prompts[:max_samples]
        print(f"限制生成数量: {max_samples}")
    
    total = len(toxic_prompts)
    completed = 0
    failed = 0
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 如果文件已存在，清空
    if os.path.exists(output_file):
        os.remove(output_file)
    
    print(f"正在生成 {total} 条 safety 对话...")
    
    # 并行生成
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # 提交所有任务
        futures = {
            executor.submit(generate_safety_pair, attack_type, prompt, model): (attack_type, prompt)
            for attack_type, prompt in toxic_prompts
        }
        
        # 处理完成的任务
        for future in as_completed(futures):
            try:
                conversation = future.result()
                
                if conversation:
                    # 直接保存 messages 数组（CustomJSON 格式）
                    with open(output_file, 'a', encoding='utf-8') as f:
                        # 只保存 messages 部分，与 identity_conversations.jsonl 格式一致
                        json.dump(conversation['messages'], f, ensure_ascii=False)
                        f.write('\n')
                    
                    completed += 1
                else:
                    failed += 1
                
                # 显示进度
                if (completed + failed) % 10 == 0:
                    progress = f"{completed + failed}/{total}"
                    success_rate = f"{100*completed/(completed+failed):.1f}%"
                    print(f"  进度: {progress} (成功率: {success_rate})")
                    
            except Exception as e:
                failed += 1
                print(f"  ✗ 处理失败: {e}")
    
    print(f"\n✓ 生成完成")
    print(f"  成功: {completed} 条")
    print(f"  失败: {failed} 条")
    
    return completed, failed


def analyze_dataset(output_file: str):
    """
    分析生成的数据集
    
    参数:
        output_file: 数据集文件
    """
    print("\n" + "=" * 80)
    print("数据集分析")
    print("=" * 80)
    
    conversations = []
    with open(output_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                conversations.append(json.loads(line))
    
    print(f"总对话数: {len(conversations)}")
    
    # 统计回复长度
    response_lengths = []
    for conv in conversations:
        if len(conv) >= 2:
            response = conv[1]['content']
            response_lengths.append(len(response))
    
    if response_lengths:
        avg_length = sum(response_lengths) / len(response_lengths)
        print(f"平均回复长度: {avg_length:.1f} 字符")
        print(f"最短回复: {min(response_lengths)} 字符")
        print(f"最长回复: {max(response_lengths)} 字符")
    
    # 显示示例
    print("\n示例对话 (前 3 条):")
    for i, conv in enumerate(conversations[:3], 1):
        user_msg = conv[0]['content']
        asst_msg = conv[1]['content']
        
        print(f"\n示例 {i}:")
        print(f"  用户: {user_msg[:80]}...")
        print(f"  助手: {asst_msg[:100]}...")


def validate_format(output_file: str) -> bool:
    """
    验证数据格式是否兼容 CustomJSON
    
    参数:
        output_file: 数据集文件
    
    返回:
        valid: 是否有效
    """
    print("\n" + "=" * 80)
    print("格式验证")
    print("=" * 80)
    
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                if not line.strip():
                    continue
                
                messages = json.loads(line)
                
                # 验证格式
                assert isinstance(messages, list), f"行 {i}: 应该是列表"
                assert len(messages) >= 2, f"行 {i}: 至少需要 2 条消息"
                
                for j, msg in enumerate(messages):
                    assert isinstance(msg, dict), f"行 {i}, 消息 {j}: 应该是字典"
                    assert 'role' in msg, f"行 {i}, 消息 {j}: 缺少 'role'"
                    assert 'content' in msg, f"行 {i}, 消息 {j}: 缺少 'content'"
                    assert msg['role'] in ['user', 'assistant'], f"行 {i}, 消息 {j}: role 应该是 user 或 assistant"
                
                # 验证角色交替
                assert messages[0]['role'] == 'user', f"行 {i}: 第一条消息应该是 user"
                assert messages[1]['role'] == 'assistant', f"行 {i}: 第二条消息应该是 assistant"
        
        print("✓ 格式验证通过")
        print("✓ 与 CustomJSON 格式兼容")
        print("✓ 可直接用于 SFT 训练")
        return True
        
    except Exception as e:
        print(f"✗ 格式验证失败: {e}")
        return False


def main():
    """主函数"""
    print("\n")
    print("=" * 80)
    print("步骤 4: 生成 Safety SFT 数据集")
    print("=" * 80)
    print()
    
    # 配置
    pipeline_dir = os.path.dirname(__file__)
    output_dir = os.path.join(pipeline_dir, "output")
    
    toxic_prompts_file = os.path.join(output_dir, "toxic_prompts.txt")
    output_file = os.path.join(output_dir, "safety_sft_dataset.jsonl")
    
    # 参数
    model = "openai/gpt-4o"  # 使用 GPT-4
    num_workers = 4  # 并行线程数
    max_samples = None  # 限制数量（None = 全部）
    
    print("配置:")
    print(f"  - 输入: {toxic_prompts_file}")
    print(f"  - 输出: {output_file}")
    print(f"  - 模型: {model}")
    print(f"  - 并行线程: {num_workers}")
    if max_samples:
        print(f"  - 限制数量: {max_samples}")
    print()
    
    # 1. 加载有毒 prompts
    toxic_prompts = load_toxic_prompts(toxic_prompts_file)
    
    # 2. 生成 safety 数据集
    completed, failed = generate_safety_dataset(
        toxic_prompts,
        output_file,
        model=model,
        num_workers=num_workers,
        max_samples=max_samples
    )
    
    # 3. 分析数据集
    if completed > 0:
        analyze_dataset(output_file)
    
    # 4. 验证格式
    if completed > 0:
        validate_format(output_file)
    
    # 总结
    print("\n" + "=" * 80)
    print("完成!")
    print("=" * 80)
    print(f"✓ Safety SFT 数据集已保存: {output_file}")
    print(f"✓ 成功生成: {completed} 条对话")
    if failed > 0:
        print(f"⚠️  失败: {failed} 条")
    print()
    print("下一步:")
    print("  1. 检查生成的数据")
    print(f"  2. 查看: head -n 1 {output_file} | python -m json.tool")
    print("  3. 在 SFT 训练中使用:")
    print(f"     CustomJSON(filepath='{output_file}')")
    print()


if __name__ == "__main__":
    main()

