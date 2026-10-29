"""기준선 고정 청크에 표 분리만 적용한 실험 인덱스를 만든다.

SemanticChunker는 다시 돌리면 청크가 2~4개 달라지므로 쓰지 않는다. 비교하는 변수가 표 분리 하나가 되도록
현재 고정 인덱스(backend/faiss_rag_final)의 청크를 그대로 가져와 표만 떼어 다시 임베딩한다.
backend/에서 `python -m benchmarks.build_table_split_index --execute`로 실행한다.
"""

from __future__ import annotations

import argparse
import json
import re

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from app.core.config import BACKEND_DIR, get_settings
from app.indexing.index_pdf import MANIFEST_FILENAME, build_manifest
from app.rag.retrieval import load_documents_from_faiss_store, load_trusted_faiss_store
from benchmarks.run_eval_v1 import BASELINE_CHUNKS_SHA256

OUTPUT_DIR = BACKEND_DIR / "faiss_rag_table_split"
TABLE_CONTEXT_CHARS = 200
_TABLE_SEPARATOR = re.compile(r"^\|(?:\s*:?-{3,}:?\s*\|)+\s*$")


def split_table_blocks(chunk: Document) -> list[Document]:
    """마크다운 표를 별도 청크로 떼고, 표 바로 앞 문맥을 머리말로 붙인다.

    표가 설명 문장과 한 청크에 섞이면 임베딩이 흐려져 표 안의 항목(예: 사용 불가 업종)이
    검색되지 않는다. 표 앞 문맥은 원래 청크에도 남겨 둔다.
    """
    lines = chunk.page_content.split("\n")
    prose: list[str] = []
    tables: list[str] = []
    # 직전 표 이후의 문장만 다음 표의 머리말로 쓴다(앞 표의 설명이 섞이지 않게).
    context_start = 0
    index = 0
    while index < len(lines):
        is_table_start = (
            index + 1 < len(lines) and "|" in lines[index] and _TABLE_SEPARATOR.match(lines[index + 1])
        )
        if not is_table_start:
            prose.append(lines[index])
            index += 1
            continue
        # OCR 결과는 표 머리행이 설명 문장과 같은 줄에서 시작하기도 한다.
        lead, header = lines[index].split("|", 1)
        if lead.strip():
            prose.append(lead)
        block = ["|" + header, lines[index + 1]]
        index += 2
        while index < len(lines) and lines[index].lstrip().startswith("|"):
            block.append(lines[index])
            index += 1
        context = " ".join(" ".join(prose[context_start:]).split())[-TABLE_CONTEXT_CHARS:]
        context_start = len(prose)
        table = "\n".join(block)
        tables.append(f"{context}\n{table}" if context else table)

    if not tables:
        return [chunk]
    chunk_id = str(chunk.metadata["chunk_id"])
    pieces = []
    if "".join(prose).strip():
        pieces.append(Document(page_content="\n".join(prose), metadata=dict(chunk.metadata)))
    for number, table in enumerate(tables, 1):
        pieces.append(
            Document(page_content=table, metadata={**chunk.metadata, "chunk_id": f"{chunk_id}_t{number}"})
        )
    return pieces


def main() -> None:
    parser = argparse.ArgumentParser(description="표 분리 실험 인덱스 생성")
    parser.add_argument("--execute", action="store_true", help="임베딩 API를 호출해 인덱스를 만든다")
    args = parser.parse_args()

    settings = get_settings()
    source_dir = BACKEND_DIR / "faiss_rag_final"
    source_manifest = json.loads((source_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    if source_manifest["chunks_sha256"] != BASELINE_CHUNKS_SHA256:
        raise SystemExit("backend/faiss_rag_final이 기준선 고정 인덱스가 아닙니다.")

    embeddings = OpenAIEmbeddings(api_key=settings.openai_api_key, model=settings.openai_embedding_model)
    base_chunks = sorted(
        load_documents_from_faiss_store(load_trusted_faiss_store(source_dir, embeddings)),
        key=lambda document: str(document.metadata["chunk_id"]),
    )
    chunks = [piece for chunk in base_chunks for piece in split_table_blocks(chunk)]
    print(f"청크 {len(base_chunks)}개 → {len(chunks)}개")
    if not args.execute:
        print("임베딩하려면 --execute를 붙이세요.")
        return

    store = FAISS.from_documents(chunks, embeddings, ids=[str(c.metadata["chunk_id"]) for c in chunks])
    OUTPUT_DIR.mkdir(exist_ok=True)
    store.save_local(str(OUTPUT_DIR))
    manifest = build_manifest(chunks, source_manifest["source_sha256"], settings.openai_embedding_model)
    manifest["chunker"] = {**source_manifest["chunker"], "table_split": {"context_chars": TABLE_CONTEXT_CHARS}}
    manifest["derived_from_chunks_sha256"] = BASELINE_CHUNKS_SHA256
    (OUTPUT_DIR / MANIFEST_FILENAME).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장: {OUTPUT_DIR} chunks_sha256={manifest['chunks_sha256']}")


if __name__ == "__main__":
    main()
