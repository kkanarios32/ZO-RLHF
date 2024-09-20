# train_online_dpo.py
from datasets import load_dataset
from trl import OnlineDPOConfig, OnlineDPOTrainer
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
reward_model = AutoModelForSequenceClassification.from_pretrained(
    "trl-lib/Qwen2-0.5B-Reward", num_labels=1
)
train_dataset = load_dataset("trl-lib/ultrafeedback-prompt", split="train")

training_args = OnlineDPOConfig(output_dir="online-dpo-qwen2", logging_steps=10)
trainer = OnlineDPOTrainer(
    model=model,
    reward_model=reward_model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
)
trainer.train()
