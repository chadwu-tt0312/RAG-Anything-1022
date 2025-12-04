import argparse
import asyncio
import os

from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

from raganything import RAGAnything, RAGAnythingConfig


async def main():
    # 設定 API 配置
    api_key = os.getenv("LLM_BINDING_API_KEY")  # 從環境變量獲取 API 金鑰
    base_url = os.getenv("LLM_BINDING_HOST")  # 可選

    # 建立 RAGAnything 配置
    config = RAGAnythingConfig(
        working_dir="./rag_storage",
        parser="mineru",  # 選擇解析器：mineru 或 docling
        parse_method="auto",  # 解析方法：auto, ocr 或 txt
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
    )

    # 定義 LLM 模型函數
    def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        return openai_complete_if_cache(
            os.getenv("LLM_MODEL", "gpt-4o-mini"),
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )

    # 定義視覺模型函數用於影象處理
    def vision_model_func(
        prompt, system_prompt=None, history_messages=[], image_data=None, messages=None, **kwargs
    ):
        # 如果提供了messages格式（用於多模態VLM增強查詢），直接使用
        if messages:
            return openai_complete_if_cache(
                "gpt-4o",
                "",
                system_prompt=None,
                history_messages=[],
                messages=messages,
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            )
        # 傳統單圖片格式
        elif image_data:
            return openai_complete_if_cache(
                "gpt-4o",
                "",
                system_prompt=None,
                history_messages=[],
                messages=[
                    {"role": "system", "content": system_prompt} if system_prompt else None,
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                            },
                        ],
                    }
                    if image_data
                    else {"role": "user", "content": prompt},
                ],
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            )
        # 純文字格式
        else:
            return llm_model_func(prompt, system_prompt, history_messages, **kwargs)

    # 定義嵌入函數
    embedding_func = EmbeddingFunc(
        embedding_dim=1536,  # text-embedding-3-small 的維度是 1536
        max_token_size=8192,
        func=lambda texts: openai_embed(
            texts,
            model="text-embedding-3-small",
            api_key=api_key,
            base_url=base_url,
        ),
    )

    # 初始化 RAGAnything
    rag = RAGAnything(
        config=config,
        llm_model_func=llm_model_func,
        vision_model_func=vision_model_func,
        embedding_func=embedding_func,
    )

    # 處理多個文檔
    await rag.process_folder_complete(
        folder_path="./eval/docs",
        output_dir="./output",
        file_extensions=[".pdf", ".png", ".docx", ".pptx"],
        recursive=True,
        max_workers=4,
    )

    # 查詢處理後的內容
    # 純文字查詢 - 基本知識庫搜尋
    text_result = await rag.aquery("文件的主要內容是什麼？", mode="hybrid")
    print("文字查詢結果:", text_result)


if __name__ == "__main__":
    asyncio.run(main())
