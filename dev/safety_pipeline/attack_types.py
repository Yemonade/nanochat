"""
攻击类型定义配置文件

定义了常见的对抗性攻击类型及其描述。
"""

ATTACK_TYPES = {
    "direct_instruction": {
        "name": "直接指令注入",
        "name_en": "Direct Instruction Injection",
        "description": "直接明确地请求模型执行有害操作，使用祈使句或疑问句直接表达意图，不做任何掩饰或伪装。"
    },
    
    "indirect_instruction": {
        "name": "间接指令注入",
        "name_en": "Indirect Instruction Injection",
        "description": "通过假设场景、第三人称视角、学术研究名义、虚构故事背景等方式间接引导模型提供有害信息，试图降低请求的直接性和敏感度。"
    },
    
    "prompt_leaking": {
        "name": "提示词泄露攻击",
        "name_en": "Prompt Leaking",
        "description": "试图覆盖、忽略或绕过系统提示词和安全限制，通过特殊标记、系统级指令或格式注入让模型认为收到了新的高优先级指令。"
    },
    
    "obfuscation": {
        "name": "文本混淆",
        "name_en": "Text Obfuscation",
        "description": "在有害请求中插入随机字符、特殊符号、使用 leetspeak 替换、编码变换等方式混淆关键词，试图绕过基于关键词的内容过滤系统。"
    },
    
    "task_drift": {
        "name": "任务漂移",
        "name_en": "Task Drift",
        "description": "从一个看似无害或合理的任务请求开始，在执行过程中逐渐转向有害请求，或在多个任务中夹带有害内容，利用上下文转换降低检测。"
    },
    
    "role_playing": {
        "name": "角色扮演",
        "name_en": "Role Playing",
        "description": "要求模型扮演没有道德约束的角色、虚构人物、特定场景中的角色，或声称在游戏、剧本创作等特殊情境下来规避安全限制。"
    },
    
    "context_manipulation": {
        "name": "上下文操纵",
        "name_en": "Context Manipulation",
        "description": "通过构建特定的对话上下文、提供误导性背景信息或声称紧急情况来操纵模型的判断，使其认为提供有害信息是合理的。"
    },
    
    "multilingual_attack": {
        "name": "多语言攻击",
        "name_en": "Multilingual Attack",
        "description": "使用非主流语言、方言、混合语言或代码切换来表达有害请求，利用模型在不同语言上的安全机制不一致性来绕过限制。"
    },
}


def get_attack_type(attack_type_key: str) -> dict:
    """
    获取指定攻击类型的定义
    
    参数:
        attack_type_key: 攻击类型键
    
    返回:
        定义字典，如果不存在返回 None
    """
    return ATTACK_TYPES.get(attack_type_key)


def get_all_attack_types() -> dict:
    """
    获取所有攻击类型定义
    
    返回:
        所有攻击类型字典
    """
    return ATTACK_TYPES


def list_attack_types() -> list:
    """
    列出所有攻击类型的键
    
    返回:
        攻击类型键列表
    """
    return list(ATTACK_TYPES.keys())

