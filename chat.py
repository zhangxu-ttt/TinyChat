import argparse
import torch
from transformers import AutoTokenizer
from enum import Enum
import sys
import os

# 将当前目录添加到 Python 路径，以便导入自定义模型
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import TransformerModel, ModelConfig


def chat(model, tokenizer, args):
    """聊天模式"""
    messages = []
    print("=" * 80)
    print("聊天模式已启动！输入 'exit' 退出，'clear' 清空对话历史")
    print("=" * 80)
    
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() == "exit":
            print("再见！")
            break
        elif user_input.lower() == "clear":
            messages = []
            print("=" * 80)
            print("对话历史已清空")
            print("=" * 80)
            continue

        messages.append({"role": "user", "content": user_input})
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # 生成回复
        response = generate_text(model, tokenizer, prompt, args)
        
        messages.append({"role": "assistant", "content": response})
        print(f"\nAssistant: {response}")
        print("-" * 80)


def generate(model, tokenizer, args):
    """生成模式"""
    print("=" * 80)
    print("生成模式已启动！输入 'exit' 退出")
    print("=" * 80)
    
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() == "exit":
            print("再见！")
            break
        
        # 生成回复
        response = generate_text(model, tokenizer, user_input, args)
        print(f"\nAssistant: {response}")
        print("-" * 80)


def generate_text(model, tokenizer, prompt, args):
    """生成文本的核心函数"""
    # Tokenize 输入
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)
    
    # 生成
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=True if args.temperature > 0 else False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # 解码（只返回新生成的部分）
    generated_ids = output_ids[0][len(input_ids[0]):]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return response


class GenerateEnum(Enum):
    chat = chat
    generate = generate

    def __call__(self, *args, **kwargs):
        return self.value(*args, **kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Chat with the ZX-LLM model")
    parser.add_argument("--model", type=str, required=True, help="Path to the ZX-LLM model")
    parser.add_argument("--tokenizer", type=str, default=None, help="Path to the tokenizer")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p (nucleus) sampling")
    parser.add_argument("--mode", type=str, default="chat", help="Mode: 'chat' or 'generate'")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run the model on: 'cpu' or 'cuda'")
    args = parser.parse_args()

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"使用设备: {device}")
    
    # 加载模型和 tokenizer
    print(f"正在加载模型: {args.model}")
    model = TransformerModel.from_pretrained(args.model)
    model.to(device)
    model.eval()
    
    if args.tokenizer is None:
        args.tokenizer = args.model
    print(f"正在加载 tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    
    # 设置特殊 token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"模型加载完成！参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # 运行对应模式
    if args.mode == "chat":
        GenerateEnum.chat(model, tokenizer, args)
    elif args.mode == "generate":
        GenerateEnum.generate(model, tokenizer, args)
    else:
        raise ValueError("Invalid mode. Please choose 'chat' or 'generate'.")


