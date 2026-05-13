"""
RoboQA — 로보틱스 논문 QA 챗봇
================================
Gemini API로 로보틱스·VLA 논문 개념을 쉽게 풀어주는 QA 챗봇.
트랙 A: LLM API 챗봇 | HF Spaces (Docker 모드) 배포
"""

import os
import gradio as gr
from google import genai

# Gemini API 설정
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise SystemExit(
        "GEMINI_API_KEY 환경변수가 비어 있습니다.\n"
        "HF Space: Settings > Secrets에 GEMINI_API_KEY 등록"
    )

client = genai.Client(api_key=GEMINI_API_KEY)
MODEL = "gemini-2.0-flash"

SYSTEM_PROMPT = """너는 로보틱스·VLA(Vision-Language-Action) 논문 전문가 AI 조교다.

전문 분야:
- VLA 모델 (OpenVLA, TwinVLA, RT-2, Octo 등)
- Embodied AI, 로봇 학습 (Imitation Learning, Reinforcement Learning)
- World Models (DreamerV3, IRIS 등)
- Diffusion Policy, Action Chunking
- 로봇 매니퓰레이션, 내비게이션

규칙:
1. 한국어로 답변한다.
2. 전문 용어는 영어 원문을 괄호에 병기한다. 예: 모방 학습(Imitation Learning)
3. 어려운 개념은 직관적 비유를 곁들여 설명한다.
4. 답변 끝에 관련 논문이 있으면 1~3개 추천한다.
5. 모르는 내용은 솔직히 "확실하지 않다"고 말한다.
6. 마크다운 포맷으로 깔끔하게 정리한다.

모드:
- 사용자가 "용어:" 로 시작하면 → 용어 사전 모드: 정의 + 관련 논문 + 직관적 비유
- 사용자가 "요약:" 로 시작하면 → 논문 요약 모드: 3줄 요약 + 핵심 기여(contribution)
- 그 외 → 일반 QA 모드: 질문에 대해 상세 답변
"""


def respond(message, history):
    """Gradio ChatInterface 콜백"""
    if not message.strip():
        return ""

    # 대화 히스토리 구성
    contents = []

    # 시스템 프롬프트를 첫 턴에 포함
    if len(history) == 0:
        contents.append({"role": "user", "parts": [{"text": SYSTEM_PROMPT + "\n\n사용자 질문: " + message}]})
    else:
        # 이전 대화 히스토리 추가
        for user_msg, assistant_msg in history:
            contents.append({"role": "user", "parts": [{"text": user_msg}]})
            contents.append({"role": "model", "parts": [{"text": assistant_msg}]})
        contents.append({"role": "user", "parts": [{"text": message}]})

    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=contents,
        )
        return response.text
    except Exception as e:
        return f"⚠️ 오류가 발생했습니다: {str(e)[:200]}"


# Gradio UI
demo = gr.ChatInterface(
    fn=respond,
    title="🤖 RoboQA — 로보틱스 논문 QA 챗봇",
    description=(
        "로보틱스·VLA 논문에 대해 무엇이든 물어보세요!\n\n"
        "💡 **사용법:**\n"
        "- 일반 질문: `OpenVLA가 뭐야?`\n"
        "- 용어 사전: `용어: Action Chunking`\n"
        "- 논문 요약: `요약: TwinVLA`"
    ),
    examples=[
        "VLA 모델이 뭐야? 쉽게 설명해줘",
        "용어: Imitation Learning",
        "요약: TwinVLA",
        "OpenVLA랑 RT-2 차이가 뭐야?",
        "Diffusion Policy가 왜 로봇에 좋아?",
    ],
    theme=gr.themes.Soft(),
)

if __name__ == "__main__":
    is_space = bool(os.getenv("SPACE_ID"))
    demo.launch(
        server_name="0.0.0.0" if is_space else "127.0.0.1",
        server_port=int(os.getenv("PORT", 7860)),
        show_api=False,
        ssr_mode=False,
    )
