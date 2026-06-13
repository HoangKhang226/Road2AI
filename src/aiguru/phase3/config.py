"""Configuration for Phase 3 batched Unsloth generation."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class GenerationConfig:
    model_name: str = os.getenv(
        "AIGURU_MODEL_NAME",
        "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    )
    max_seq_length: int = int(os.getenv("AIGURU_MAX_SEQ_LENGTH", "8192"))
    max_new_tokens: int = int(os.getenv("AIGURU_MAX_NEW_TOKENS", "1024"))
    max_context_chars: int = int(os.getenv("AIGURU_MAX_CONTEXT_CHARS", "18000"))
    batch_size: int = int(os.getenv("AIGURU_GENERATION_BATCH_SIZE", "4"))
    temperature: float = float(os.getenv("AIGURU_TEMPERATURE", "0.0"))
    top_p: float = float(os.getenv("AIGURU_TOP_P", "0.9"))
    repetition_penalty: float = float(os.getenv("AIGURU_REPETITION_PENALTY", "1.1"))
    load_in_4bit: bool = os.getenv("AIGURU_LOAD_IN_4BIT", "1") != "0"


SYSTEM_PROMPT = """Bạn là trợ lý pháp lý AI chuyên về pháp luật Việt Nam cho doanh nghiệp SME.

QUY TẮC BẮT BUỘC:
1. Chỉ sử dụng thông tin trong [CONTEXT]. Không bịa đặt điều luật, số hiệu văn bản hoặc dữ kiện.
2. Chỉ được trích dẫn các mục [TRÍCH DẪN HỢP LỆ] xuất hiện trong context.
3. Khi trích dẫn căn cứ pháp lý, phải nêu rõ Điều X và tên văn bản tương ứng.
4. Nếu context không đủ căn cứ, nói rõ hệ thống dữ liệu chưa ghi nhận quy định cụ thể.
5. Trả lời rõ ràng, thực tiễn, phù hợp với người đọc không chuyên.
6. Kết thúc bằng cảnh báo giới hạn chuẩn."""


ANSWER_FORMAT = """Trả lời theo cấu trúc:
1. **Căn cứ pháp lý**: Liệt kê rõ Điều X của văn bản nào.
2. **Phân tích**: Áp dụng căn cứ vào tình huống được hỏi.
3. **Tư vấn sơ bộ**: Nêu hướng xử lý thực tế cho doanh nghiệp.
4. **Cảnh báo**: Cảnh báo giới hạn: Đây là tư vấn sơ bộ từ AI, doanh nghiệp cần đối chiếu văn bản gốc hoặc tham khảo chuyên gia pháp lý trước khi áp dụng."""
