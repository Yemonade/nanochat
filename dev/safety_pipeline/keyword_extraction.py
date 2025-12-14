"""
基于 LXT 的迭代数据集生成模块

对 SFT 模型未成功拒答的 prompts 进行分析：
1. 使用 LXT (Layer-wise Relevance Propagation) 提取注意力分数
2. 识别高注意力分数的 tokens 作为新的敏感词
3. 将新敏感词用于下一轮数据生成

"""

import os
import sys
import json
import torch
from typing import List, Dict, Tuple
from collections import Counter
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.qwen2 import modeling_qwen2

from lxt.efficient import monkey_patch
from lxt.utils import clean_tokens

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from nanochat.common import get_base_dir


class LXTSensitiveWordExtractor:
    """
    使用 LXT 提取敏感词的类
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        stopwords: List[str] = None
    ):
        """
        初始化
        
        参数:
            model_path: 模型路径
            device: 设备
            stopwords: 停用词列表
        """
        print(f"正在加载模型: {model_path}")
        
        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            padding_side='left'
        )
        
        # Monkey patch Qwen2 模型以支持 LRP
        monkey_patch(modeling_qwen2, verbose=False)
        
        # 加载模型
        self.model = modeling_qwen2.Qwen2ForCausalLM.from_pretrained(
            model_path,
            device_map=device,
            torch_dtype=torch.bfloat16
        )
        
        # 设置为训练模式并启用梯度检查点以节省内存
        self.model.train()
        self.model.gradient_checkpointing_enable()
        
        # 禁用参数梯度以节省内存
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.device = device
        
        # 停用词
        if stopwords is None:
            self.stopwords = {' ', '\n', '\t', '', '<', '>', '|', 'user', 'assistant'}
        else:
            self.stopwords = set(stopwords)
        
        print(f"✓ 模型加载完成")
    
    def extract_sensitive_tokens(
        self,
        prompt: str,
        top_k: int = 10,
        threshold: float = 0.3
    ) -> List[Tuple[str, float]]:
        """
        从单个 prompt 中提取敏感 tokens
        
        参数:
            prompt: 用户请求
            top_k: 返回前 k 个高相关性 tokens
            threshold: 相关性阈值（归一化后）
        
        返回:
            [(token, relevance_score), ...]
        """
        try:
            # 构建对话
            messages = [{"role": "user", "content": prompt}]
            
            # 应用对话模板
            texts = self.tokenizer.apply_chat_template(
                [messages],
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize
            input_ids = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True
            ).input_ids.to(self.device)
            
            # 获取输入 embeddings（需要计算梯度）
            input_embeds = self.model.get_input_embeddings()(input_ids)
            
            # 前向传播，获取最后位置的最大 logit
            output_logits = self.model(
                inputs_embeds=input_embeds.requires_grad_(),
                use_cache=False
            ).logits
            
            max_logits, max_indices = torch.max(output_logits[0, -1, :], dim=-1)
            
            # 反向传播（LRP 计算）
            max_logits.backward()
            
            # 计算相关性：Gradient * Input
            relevance = (
                input_embeds.grad * input_embeds
            ).float().sum(-1).detach().cpu()[0]
            
            # 归一化到 [-1, 1]
            relevance = relevance / relevance.abs().max()
            
            # 获取 tokens
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
            tokens = clean_tokens(tokens)
            
            # 找到 prompt 在 tokens 中的位置
            # 通常需要跳过系统提示和特殊标记
            prompt_start_idx = self._find_prompt_start(tokens, prompt)
            
            # 提取高相关性的 tokens
            sensitive_tokens = []
            for i in range(prompt_start_idx, len(tokens)):
                token = tokens[i]
                score = relevance[i].item()
                
                # 过滤停用词和低相关性 tokens
                if (abs(score) >= threshold and 
                    token not in self.stopwords and
                    len(token.strip()) > 0):
                    sensitive_tokens.append((token, score))
            
            # 按相关性排序并返回 top_k
            sensitive_tokens.sort(key=lambda x: abs(x[1]), reverse=True)
            
            return sensitive_tokens[:top_k]
            
        except Exception as e:
            print(f"  ✗ 提取失败: {e}")
            return []
    
    def _find_prompt_start(self, tokens: List[str], prompt: str) -> int:
        """
        在 tokens 中找到 prompt 的起始位置
        
        参数:
            tokens: token 列表
            prompt: 原始 prompt
        
        返回:
            start_idx: 起始索引
        """
        # 简单策略：找到第一个 prompt 中的词
        prompt_words = prompt.split()[:3]  # 取前3个词
        
        for i, token in enumerate(tokens):
            token_clean = token.strip().lower()
            for word in prompt_words:
                if word.lower() in token_clean or token_clean in word.lower():
                    return max(0, i - 2)  # 稍微提前一点
        
        # 如果找不到，返回一个合理的默认值
        return len(tokens) // 2


def load_failed_prompts(dataset_file: str) -> List[Dict]:
    """
    加载验证失败的 prompts
    
    参数:
        dataset_file: 验证报告文件
    
    返回:
        failed_prompts: 失败的 prompt 列表
    """
    print(f"正在加载验证失败的 prompts: {dataset_file}")
    
    with open(dataset_file, 'r', encoding='utf-8') as f:
        report = json.load(f)
    
    failed_samples = report.get('failed_samples', [])
    
    prompts = []
    for sample in failed_samples:
        prompts.append({
            'user_request': sample['user'],
            'assistant_response': sample['assistant'],
            'validation': sample['validation']
        })
    
    print(f"✓ 加载 {len(prompts)} 个失败的 prompts")
    return prompts


def extract_sensitive_words_batch(
    extractor: LXTSensitiveWordExtractor,
    prompts: List[str],
    top_k: int = 10
) -> List[str]:
    """
    批量提取敏感词
    
    参数:
        extractor: LXT 提取器
        prompts: prompt 列表
        top_k: 每个 prompt 提取的 top_k tokens
    
    返回:
        sensitive_words: 敏感词列表（去重）
    """
    print(f"\n正在使用 LXT 分析 {len(prompts)} 个 prompts...")
    
    all_tokens = []
    
    for prompt in tqdm(prompts, desc="提取敏感词"):
        tokens_with_scores = extractor.extract_sensitive_tokens(
            prompt,
            top_k=top_k
        )
        
        # 只保留 token
        tokens = [token for token, score in tokens_with_scores]
        all_tokens.extend(tokens)
    
    # 统计词频
    token_counts = Counter(all_tokens)
    
    # 按频率排序
    sorted_tokens = sorted(token_counts.items(), key=lambda x: -x[1])
    
    print(f"\n✓ 提取到 {len(sorted_tokens)} 个唯一敏感词")
    
    # 显示最常见的词
    print("\n最常见的敏感词 (Top 20):")
    for token, count in sorted_tokens[:20]:
        print(f"  {token}: {count} 次")
    
    return [token for token, count in sorted_tokens]


def merge_with_existing_keywords(
    new_keywords: List[str],
    existing_file: str
) -> List[str]:
    """
    合并新提取的关键词与现有关键词
    
    参数:
        new_keywords: 新提取的关键词
        existing_file: 现有关键词文件
    
    返回:
        merged_keywords: 合并后的关键词列表
    """
    print(f"\n正在合并关键词...")
    
    # 加载现有关键词
    existing_keywords = []
    if os.path.exists(existing_file):
        with open(existing_file, 'r', encoding='utf-8') as f:
            existing_keywords = [line.strip() for line in f if line.strip()]
        print(f"  现有关键词: {len(existing_keywords)} 个")
    
    # 合并并去重
    all_keywords = list(set(existing_keywords + new_keywords))
    
    print(f"  新增关键词: {len(new_keywords)} 个")
    print(f"  合并后总计: {len(all_keywords)} 个")
    
    return all_keywords


def save_keywords(keywords: List[str], output_file: str):
    """
    保存关键词到文件
    
    参数:
        keywords: 关键词列表
        output_file: 输出文件
    """
    print(f"\n正在保存关键词到 {output_file}...")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for keyword in keywords:
            f.write(keyword + '\n')
    
    print(f"✓ 已保存 {len(keywords)} 个关键词")


def main():
    """主函数"""
    print("\n")
    print("=" * 80)
    print("步骤 6: 基于 LXT 的迭代数据集生成")
    print("=" * 80)
    print()
    
    # 配置
    pipeline_dir = os.path.dirname(__file__)
    output_dir = os.path.join(pipeline_dir, "output")
    
    # 文件路径
    model_path = "Qwen/Qwen2.5-7B-Instruct"  # SFT 后的模型路径
    validation_report = os.path.join(output_dir, "validation_report.json")
    existing_keywords_file = os.path.join(output_dir, "sensitive_keywords.txt")
    new_keywords_file = os.path.join(output_dir, "sensitive_keywords_iteration.txt")
    
    # 参数
    top_k = 10  # 每个 prompt 提取的 top_k tokens
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("配置:")
    print(f"  - 模型: {model_path}")
    print(f"  - 验证报告: {validation_report}")
    print(f"  - 现有关键词: {existing_keywords_file}")
    print(f"  - 输出: {new_keywords_file}")
    print(f"  - Top-K: {top_k}")
    print(f"  - 设备: {device}")
    print()
    
    # 1. 加载验证失败的 prompts
    failed_data = load_failed_prompts(validation_report)
    
    if not failed_data:
        print("✓ 没有失败的 prompts，无需迭代")
        return
    
    # 提取 prompts
    failed_prompts = [item['user_request'] for item in failed_data]
    
    # 2. 初始化 LXT 提取器
    extractor = LXTSensitiveWordExtractor(
        model_path=model_path,
        device=device
    )
    
    # 3. 提取敏感词
    new_keywords = extract_sensitive_words_batch(
        extractor,
        failed_prompts,
        top_k=top_k
    )
    
    # 4. 合并关键词
    merged_keywords = merge_with_existing_keywords(
        new_keywords,
        existing_keywords_file
    )
    
    # 5. 保存关键词
    save_keywords(merged_keywords, new_keywords_file)
    
    # 总结
    print("\n" + "=" * 80)
    print("完成!")
    print("=" * 80)
    print(f"✓ 新关键词文件: {new_keywords_file}")
    print(f"✓ 总关键词数: {len(merged_keywords)}")
    print()
    print("下一步:")
    print("  1. 查看新提取的关键词")
    print(f"  2. 使用新关键词重新生成数据:")
    print(f"     python -m dev.safety_pipeline.generate_toxic_prompts")
    print("  3. 重复整个流程直到收敛")
    print()
    
    # 清理
    del extractor
    if device == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

