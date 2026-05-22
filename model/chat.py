import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def main():
    # 1. 모델 경로 설정
    base_model_id = "microsoft/phi-2"
    adapter_path = "./phi-2-lima-lora-final"  # 방금 학습이 완료되어 저장된 폴더

    print("로딩 중... (약 1~2분 정도 소요될 수 있습니다)")
    
    # 2. 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. 원본 모델을 bfloat16으로 빠르게 로드 (A5000 최적화)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    # 4. 학습된 LoRA 어댑터 가중치를 원본 모델에 융합
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()  # 평가(추론) 모드로 전환
    
    print("\n✅ 모델 로딩 완료! 대화를 시작합니다. (종료하려면 'quit' 입력)")
    print("-" * 50)

    # 5. 무한 대화 루프
    while True:
        user_input = input("\n👤 질문을 입력하세요: ")
        
        if user_input.lower() == 'quit':
            print("대화를 종료합니다.")
            break
            
        # ⭐️ 핵심: 학습할 때 사용했던 프롬프트 양식과 똑같이 맞춰주어야 모델이 알아듣습니다.
        prompt = f"Instruct: {user_input}\nOutput: "
        
        # 입력값을 텐서로 변환하여 GPU로 전송
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        # 답변 생성
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,       # 최대 생성 단어 수
                temperature=0.7,          # 창의성 조절 (낮을수록 보수적, 높을수록 창의적)
                top_p=0.9,                # 샘플링 확률
                repetition_penalty=1.1,   # 같은 말 반복 방지
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # 생성된 토큰을 텍스트로 디코딩 (입력 프롬프트 부분은 잘라내고 출력)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = response.split("Output: ")[-1]
        
        print(f"\n🤖 Phi-2의 답변:\n{answer}")
        print("-" * 50)

if __name__ == "__main__":
    main()