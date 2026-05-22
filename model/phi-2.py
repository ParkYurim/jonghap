import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer, SFTConfig

def main():
    # 1. 모델 및 출력 디렉토리 설정
    model_id = "microsoft/phi-2"
    output_dir = "./phi-2-lima-lora"

    # 2. LIMA 데이터셋 로드 및 전처리
    print("데이터셋을 로드하는 중...")
    # 수정 후 코드: json 파일 직접 불러오기
    dataset = load_dataset(
        "json",
        data_files="https://huggingface.co/datasets/GAIR/lima/resolve/main/train.jsonl",
        split="train" 
    )
    def format_prompts(examples):
        texts = []
        for conv in examples['conversations']:
            # LIMA 데이터셋의 conversations는 [User, Assistant] 형태의 텍스트 리스트입니다.
            if len(conv) >= 2:
                user_msg = conv[0]
                assistant_msg = conv[1]
                # Phi-2 모델에 맞게 프롬프트 구성
                text = f"Instruct: {user_msg}\nOutput: {assistant_msg}<|endoftext|>"
                texts.append(text)
        return {"text": texts}

    # 데이터셋 맵핑
    dataset = dataset.map(format_prompts, batched=True, remove_columns=dataset.column_names)

    # 3. 4-bit 양자화 설정 (QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # 4. 토크나이저 로드
    print("토크나이저 및 모델을 로드하는 중...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    # Phi-2는 기본 패딩 토큰이 없으므로 eos_token을 지정해줍니다.
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 5. 모델 로드
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 모델을 양자화 학습에 맞게 준비
    model = prepare_model_for_kbit_training(model)

    # 6. LoRA 설정
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"], 
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # model = get_peft_model(model, peft_config)
    # model.print_trainable_parameters()

    # 7. 학습 인자 설정
# 7. 학습 인자 설정 (A5000 24GB 맞춤형 세팅)
    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=4,    # VRAM이 넉넉하므로 배치 사이즈를 1 -> 4로 4배 증가!
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        save_strategy="epoch",
        logging_steps=10,
        num_train_epochs=3,
        optim="paged_adamw_8bit",
        bf16=True,
        max_grad_norm=0.3,
        warmup_steps=50,           
        dataset_text_field="text",        
        max_length=2048,              # LIMA의 긴 대화를 다 담기 위해 시퀀스 길이 1024 -> 2048로 2배 확장! 헿 교수님 감사합니다1!!!
    )

    # 8. SFT Trainer 설정
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,         
        processing_class=tokenizer,
        args=training_args,       

    )

    # 9. 학습 시작
    print("학습을 시작합니다...")
    trainer.train()

    # 10. 어댑터 저장
    print("어댑터 가중치를 저장합니다...")
    trainer.model.save_pretrained(f"{output_dir}-final")

if __name__ == "__main__":
    main()