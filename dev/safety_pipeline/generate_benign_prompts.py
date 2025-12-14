"""
基于敏感词生成无毒样本

给定敏感词列表，生成无害、正常的用户请求样本。
不指定类型或模板，纯粹基于关键词生成自然、无害的对话请求。
"""

import os
import sys
import json
import random
import torch
from typing import List, Dict
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from nanochat.common import get_base_dir

# 关闭梯度计算
torch.set_grad_enabled(False)


def load_sensitive_keywords(keywords_file: str) -> List[str]:
    """
    加载敏感词列表
    
    参数:
        keywords_file: 敏感词文件路径
    
    返回:
        keywords: 敏感词列表
    """
    print(f"正在加载敏感词: {keywords_file}")
    
    if not os.path.exists(keywords_file):
        print(f"⚠️  文件不存在: {keywords_file}")
        return []
    
    with open(keywords_file, 'r', encoding='utf-8') as f:
        keywords = [line.strip() for line in f if line.strip()]
    
    print(f"✓ 加载 {len(keywords)} 个敏感词")
    return keywords


def generate_benign_prompt_simple(
    keywords: List[str],
    num_keywords: int = 2
) -> str:
    """
    基于关键词生成简单的无害 prompt（规则生成）
    不使用模板，纯粹基于关键词组合
    
    参数:
        keywords: 敏感词列表
        num_keywords: 使用的关键词数量
    
    返回:
        prompt: 生成的无害 prompt
    """
    # 随机选择关键词
    selected = random.sample(keywords, min(num_keywords, len(keywords)))
    
    # 纯粹基于关键词组合，不使用模板
    # 简单组合关键词，形成自然的问题或请求
    if len(selected) == 1:
        # 单个关键词：直接询问
        prompt = f"关于{selected[0]}的问题"
    elif len(selected) == 2:
        # 两个关键词：组合询问
        prompt = f"{selected[0]}和{selected[1]}的关系"
    else:
        # 多个关键词：列举
        prompt = "、".join(selected) + "相关的问题"
    
    return prompt


def generate_benign_prompt_with_model(
    keywords: List[str],
    model,
    tokenizer,
    device: str = "cuda"
) -> str:
    """
    使用模型基于关键词生成无害 prompt
    
    参数:
        keywords: 敏感词列表
        model: 模型
        tokenizer: tokenizer
        device: 设备
    
    返回:
        prompt: 生成的无害 prompt
    """
    # 随机选择 2-3 个关键词
    num_keywords = random.randint(2, min(3, len(keywords)))
    selected = random.sample(keywords, num_keywords)
    keywords_str = "、".join(selected)
    
    # 构建生成提示（只提供关键词，要求生成无害请求，不使用模板）
    system_prompt = f"""基于以下关键词，生成一个用户请求。
关键词：{keywords_str}
"""

    try:
        messages = [{"role": "user", "content": system_prompt}]
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = tokenizer(formatted, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=80,
                temperature=0.9,
                top_k=50,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # 解码
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取生成的部分
        if "assistant" in generated.lower():
            parts = generated.lower().split("assistant")
            if len(parts) > 1:
                prompt = parts[-1].strip()
            else:
                prompt = generated
        else:
            prompt = generated
        
        # 清理和截断
        prompt = prompt.strip()
        if len(prompt) > 200:
            prompt = prompt[:200]
        
        return prompt if prompt else None
        
    except Exception as e:
        print(f"  ⚠️  模型生成失败: {e}")
        return None


def generate_benign_prompts_batch(
    keywords: List[str],
    num_prompts: int = 10,
    model=None,
    tokenizer=None,
    device: str = "cuda"
) -> List[str]:
    """
    批量生成无害 prompts
    
    参数:
        keywords: 敏感词列表
        num_prompts: 生成数量
        model: 模型（可选）
        tokenizer: tokenizer（可选）
        device: 设备
    
    返回:
        prompts: 生成的无害 prompts 列表
    """
    prompts = []
    use_model = model is not None and tokenizer is not None
    
    for _ in range(num_prompts):
        if use_model:
            # 使用模型生成
            prompt = generate_benign_prompt_with_model(
                keywords, model, tokenizer, device
            )
            if not prompt:
                # 如果模型生成失败，使用简单规则
                prompt = generate_benign_prompt_simple(keywords)
        else:
            # 使用简单规则生成
            prompt = generate_benign_prompt_simple(keywords)
        
        if prompt:
            prompts.append(prompt)
    
    return prompts


def generate_benign_dataset(
    keywords_file: str,
    output_file: str,
    num_prompts: int = 500,
    model_path: str = None,
    use_model: bool = False
):
    """
    生成无害数据集
    
    参数:
        keywords_file: 敏感词文件
        output_file: 输出文件
        num_prompts: 生成数量
        model_path: 模型路径（如果使用模型生成）
        use_model: 是否使用模型生成
    """
    print("\n" + "=" * 80)
    print("生成无害样本数据集")
    print("=" * 80)
    print()
    
    # 1. 加载敏感词
    keywords = load_sensitive_keywords(keywords_file)
    
    if not keywords:
        print("✗ 没有敏感词，无法生成")
        return
    
    # 2. 加载模型（如果需要）
    model = None
    tokenizer = None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if use_model:
        if model_path and os.path.exists(model_path):
            print(f"\n正在加载模型: {model_path}")
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map=device,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32
                )
                print(f"✓ 模型加载完成")
            except Exception as e:
                print(f"⚠️  模型加载失败: {e}")
                print("  将使用规则生成")
                use_model = False
        else:
            print(f"⚠️  模型路径不存在或未指定")
            print("  将使用规则生成")
            use_model = False
    
    # 3. 生成无害 prompts
    print(f"\n正在生成 {num_prompts} 条无害 prompts...")
    
    all_prompts = []
    batch_size = 50  # 每批生成数量
    
    for i in tqdm(range(0, num_prompts, batch_size), desc="生成进度"):
        batch_size_actual = min(batch_size, num_prompts - i)
        prompts = generate_benign_prompts_batch(
            keywords,
            num_prompts=batch_size_actual,
            model=model if use_model else None,
            tokenizer=tokenizer if use_model else None,
            device=device
        )
        all_prompts.extend(prompts)
    
    print(f"\n✓ 总共生成 {len(all_prompts)} 条 prompts")
    
    # 4. 去重和清理
    print("\n正在去重和清理...")
    
    seen = set()
    unique_prompts = []
    
    for prompt in all_prompts:
        clean = prompt.strip()
        
        if clean and clean not in seen and len(clean) > 5:
            seen.add(clean)
            unique_prompts.append(clean)
    
    print(f"✓ 去重后剩余 {len(unique_prompts)} 条")
    
    # 5. 随机打乱
    random.shuffle(unique_prompts)
    
    # 6. 保存到文件
    print(f"\n正在保存到 {output_file}...")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for prompt in unique_prompts:
            f.write(prompt + '\n')
    
    print(f"✓ 已保存 {len(unique_prompts)} 条无害 prompts")
    
    # 7. 生成统计报告
    stats_file = output_file.replace('.txt', '_stats.json')
    
    stats = {
        "total_prompts": len(unique_prompts),
        "source_keywords": len(keywords),
        "generation_method": "model" if use_model else "rule",
        "sample_prompts": unique_prompts[:10]
    }
    
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 统计报告已保存: {stats_file}")
    
    # 8. 显示统计
    print("\n" + "=" * 80)
    print("统计信息")
    print("=" * 80)
    print(f"总 prompts 数: {stats['total_prompts']}")
    print(f"来源敏感词数: {stats['source_keywords']}")
    print(f"生成方法: {stats['generation_method']}")
    
    print("\n示例 prompts (前 5 条):")
    for i, prompt in enumerate(unique_prompts[:5], 1):
        print(f"  {i}. {prompt}")
    
    # 清理
    if model is not None:
        del model
        if device == "cuda":
            torch.cuda.empty_cache()


def main():
    """主函数"""
    print("\n")
    print("=" * 80)
    print("基于敏感词生成无害样本")
    print("=" * 80)
    print()
    
    # 配置
    pipeline_dir = os.path.dirname(__file__)
    output_dir = os.path.join(pipeline_dir, "output")
    
    keywords_file = os.path.join(output_dir, "sensitive_keywords.txt")
    output_file = os.path.join(output_dir, "benign_prompts.txt")
    
    # 参数
    num_prompts = 500  # 生成数量
    model_path = None  # 模型路径（None = 使用规则生成）
    use_model = False  # 是否使用模型生成（False = 使用规则生成）
    
    print("配置:")
    print(f"  - 敏感词文件: {keywords_file}")
    print(f"  - 输出文件: {output_file}")
    print(f"  - 生成数量: {num_prompts}")
    print(f"  - 生成方法: {'模型生成' if use_model else '规则生成'}")
    if use_model and model_path:
        print(f"  - 模型路径: {model_path}")
    print()
    
    # 生成
    generate_benign_dataset(
        keywords_file=keywords_file,
        output_file=output_file,
        num_prompts=num_prompts,
        model_path=model_path,
        use_model=use_model
    )
    
    # 总结
    print("\n" + "=" * 80)
    print("完成!")
    print("=" * 80)
    print(f"✓ 无害 prompts 已保存: {output_file}")
    print()
    print("下一步:")
    print("  1. 检查生成的 prompts")
    print(f"  2. 查看: head -20 {output_file}")
    print("  3. 可用于生成无害回复对，构建平衡数据集")
    print()


if __name__ == "__main__":
    main()
