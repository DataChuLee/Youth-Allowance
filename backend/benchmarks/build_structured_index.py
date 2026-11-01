"""구조 기반 청킹(실험 8) 인덱스를 만든다.

기준선과 같은 OCR JSONL·임베딩 모델을 쓰고 청커만 바꾼다. 서비스 인덱스(faiss_rag_final)는 건드리지 않는다.
backend/에서 `python -m benchmarks.build_structured_index --execute`로 실행한다.
측정할 때는 FAISS_INDEX_DIR=backend/faiss_rag_structured_v<버전>으로 바꾼다.
"""

from __future__ import annotations

import argparse
import hashlib
import json

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from app.core.config import BACKEND_DIR, PROJECT_ROOT, get_settings
from app.indexing.index_pdf import MANIFEST_FILENAME, assign_chunk_metadata, build_manifest, load_ocr_pages
from app.indexing.structured_chunker import CHUNKER_CONFIG, chunk_pages

# 버전마다 폴더를 나눠 이전 버전 측정 결과와 짝지을 수 있게 한다(v1: 실험 8, v2: 부모 절 추가 8b).
OUTPUT_DIR = BACKEND_DIR / f"faiss_rag_structured_v{CHUNKER_CONFIG['version']}"


def main() -> None:
    parser = argparse.ArgumentParser(description="구조 기반 청킹 실험 인덱스 생성")
    parser.add_argument("--execute", action="store_true", help="임베딩 API를 호출해 인덱스를 만든다")
    args = parser.parse_args()

    settings = get_settings()
    jsonl_path = PROJECT_ROOT / settings.ocr_jsonl_path
    chunks = assign_chunk_metadata(chunk_pages(load_ocr_pages(jsonl_path)), jsonl_path.name)
    # 기준선 sem_ ID와 섞이지 않게 접두어를 바꾼다.
    for chunk in chunks:
        chunk.metadata["chunk_id"] = chunk.metadata["chunk_id"].replace("sem_", "st_")
    lengths = sorted(len(chunk.page_content) for chunk in chunks)
    print(f"청크 {len(chunks)}개, 길이 최소 {lengths[0]} / 중앙 {lengths[len(lengths) // 2]} / 최대 {lengths[-1]}")
    if not args.execute:
        print("임베딩하려면 --execute를 붙이세요.")
        return

    embeddings = OpenAIEmbeddings(api_key=settings.openai_api_key, model=settings.openai_embedding_model)
    store = FAISS.from_documents(chunks, embeddings, ids=[str(c.metadata["chunk_id"]) for c in chunks])
    OUTPUT_DIR.mkdir(exist_ok=True)
    store.save_local(str(OUTPUT_DIR))
    manifest = build_manifest(
        chunks, hashlib.sha256(jsonl_path.read_bytes()).hexdigest(), settings.openai_embedding_model
    )
    manifest["chunker"] = CHUNKER_CONFIG
    # 청크 본문이 같아도 부모 절이 다르면 다른 인덱스다. 해시에 부모 절을 포함한다.
    fingerprint = "\n".join(
        "\t".join([str(c.metadata["chunk_id"]), c.page_content, c.metadata.get("parent_id", ""),
                   c.metadata.get("parent_text", "")])
        for c in chunks
    )
    manifest["chunks_sha256"] = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()
    (OUTPUT_DIR / MANIFEST_FILENAME).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장: {OUTPUT_DIR} chunks_sha256={manifest['chunks_sha256']}")


if __name__ == "__main__":
    main()
