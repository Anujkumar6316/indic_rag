"""Gradio Studio — interactive playground for the Indic RAG microservice.

Tabs:
1. 🔍 Hybrid Search   — dense+BM25/RRF retrieval with lang/domain filters
2. 💬 Ask (RAG)       — full LangGraph query pipeline incl. grounded answer
3. 📥 Ingest          — paste/upload documents, audit & batch ingestion
4. 📊 Corpus          — collection stats & health of the wiring

SOLID notes: SRP (UI only — delegates to RagService ports), DIP (the service
container is injected via ``build_studio(service=...)`` so tests can pass a
pre-seeded in-memory container).
"""
from __future__ import annotations

import json
from typing import Optional

import gradio as gr

from indic_rag.config import get_settings
from indic_rag.schemas import RawDocument
from indic_rag.service import RagService, get_service

_LANG_LABELS = {
    "auto": "Auto-detect",
    "sa": "Sanskrit (संस्कृतम्)",
    "hi": "Hindi (हिन्दी)",
    "mr": "Marathi (मराठी)",
    "gu": "Gujarati (ગુજરાતી)",
    "en": "English",
}


def _fmt_results(results) -> str:
    if not results:
        return "_No chunks retrieved._"
    lines = []
    for r in results:
        lines.append(
            f"**#{r.rank} · [{r.lang}] {r.title}**  \n"
            f"score `{r.score}` · sources `{', '.join(r.sources) or '-'}`\n\n"
            f"> {r.text}\n"
        )
    return "\n\n---\n\n".join(lines)


def build_studio(service: Optional[RagService] = None) -> gr.Blocks:
    svc = service or get_service()

    def do_search(query: str, lang_pick: str, domain: str, top_k: int):
        if not query.strip():
            return "*Type a query first.*"
        lang = None if lang_pick == "auto" else lang_pick
        results = svc.retriever.search(query, lang=lang, domain=domain or None, final_k=int(top_k))
        return _fmt_results(results)

    def do_ask(query: str, lang_pick: str, domain: str, top_k: int):
        if not query.strip():
            return "*Type a question first.*", ""
        lang = None if lang_pick == "auto" else lang_pick
        state = svc.query_service.ask(query, lang=lang, domain=domain or None, final_k=int(top_k))
        answer = state.get("answer", "")
        cites = _fmt_results(state.get("results", []))
        meta = (
            f"_detected language: **{state.get('detected_lang')}** · "
            f"provider: **{state.get('provider')}**_"
        )
        return answer, f"{meta}\n\n{cites}"

    def do_ingest_text(text: str, doc_id: str, title: str, lang_pick: str, domain: str):
        if not text.strip():
            return "⚠️ Paste some Indic text first."
        lang = None if lang_pick == "auto" else lang_pick
        doc = RawDocument(
            doc_id=doc_id or "studio-doc", title=title or "none",
            lang=lang, domain=domain or "general", raw_text=text,
        )
        chunks = svc.pipeline.process_document(doc)
        upserted = svc.pipeline.embed_and_store(chunks)
        preview = "\n\n".join(f"- `#{c.chunk_index}` ({c.token_count} tok) {c.text[:160]}…"
                              for c in chunks[:8])
        return (
            f"✅ Indexed **{upserted}** chunk(s) from `{doc.doc_id}` "
            f"(lang={chunks[0].lang if chunks else lang})\n\n{preview}"
        )

    def do_upload(file, doc_id: str, lang_pick: str, domain: str):
        if file is None:
            return "⚠️ Upload a .txt/.md file first."
        path = getattr(file, "name", str(file))
        text = open(path, encoding="utf-8", errors="replace").read()
        stem = path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        return do_ingest_text(text, doc_id or stem, stem, lang_pick, domain)

    def do_audit():
        reports = svc.pipeline.run_audit()
        if not reports:
            return "_data/raw is empty — nothing to audit._"
        rows = [
            {"file": r.path, "action": r.action, "font": r.detected_font or "-",
             "suspicious_ratio": round(r.suspicious_ratio, 3)}
            for r in reports
        ]
        return json.dumps(rows, ensure_ascii=False, indent=2)

    def do_batch_ingest(run_audit: bool):
        out = svc.pipeline.run_full() if run_audit else {"stats": svc.pipeline.run_ingest().model_dump()}
        return json.dumps(out, ensure_ascii=False, indent=2, default=str)

    def do_stats():
        health = svc.health()
        chunks = svc.store.scroll_all()
        by_lang: dict[str, int] = {}
        for c in chunks:
            by_lang[c.lang] = by_lang.get(c.lang, 0) + 1
        return (
            "```json\n"
            + json.dumps({"health": health, "chunks_by_lang": by_lang},
                         ensure_ascii=False, indent=2)
            + "\n```"
        )

    lang_choices = list(_LANG_LABELS.values())
    lang_codes = list(_LANG_LABELS.keys())

    def to_code(label: str) -> str:
        return lang_codes[lang_choices.index(label)]

    with gr.Blocks(title="Indic Multilingual RAG Studio", theme="soft") as demo:
        gr.Markdown(
            "# 🕉️ Indic Multilingual RAG Studio\n"
            "**sa · hi · mr · gu · en** — hybrid retrieval "
            "(EmbeddingGemma-2 / BGE-M3 dense + BM25 sparse, RRF fusion) over Qdrant."
        )

        with gr.Tab("🔍 Hybrid Search"):
            q = gr.Textbox(label="Query", placeholder="उदा॥ अग्निम… या Who controls the water of Ayodhya?")
            with gr.Row():
                l1 = gr.Dropdown(lang_choices, value="Auto-detect", label="Language filter")
                d1 = gr.Textbox(label="Domain (optional)", placeholder="e.g. ramayana")
                k1 = gr.Slider(1, 20, value=5, step=1, label="Top-K")
            b1 = gr.Button("Search", variant="primary")
            o1 = gr.Markdown()
            b1.click(lambda *a: do_search(a[0], to_code(a[1]), a[2], a[3]),
                     inputs=[q, l1, d1, k1], outputs=o1)

        with gr.Tab("💬 Ask (RAG graph)"):
            qa = gr.Textbox(label="Question", lines=2)
            with gr.Row():
                l2 = gr.Dropdown(lang_choices, value="Auto-detect", label="Language filter")
                d2 = gr.Textbox(label="Domain (optional)")
                k2 = gr.Slider(1, 20, value=5, step=1, label="Retrieved chunks")
            b2 = gr.Button("Ask", variant="primary")
            ans = gr.Markdown(label="Answer")
            srcs = gr.Markdown(label="Sources")
            b2.click(lambda *a: do_ask(a[0], to_code(a[1]), a[2], a[3]),
                     inputs=[qa, l2, d2, k2], outputs=[ans, srcs])

        with gr.Tab("📥 Ingest"):
            with gr.Row():
                with gr.Column():
                    txt = gr.Textbox(label="Paste document text", lines=8)
                    with gr.Row():
                        did = gr.Textbox(label="Doc ID", value="studio-doc")
                        ttl = gr.Textbox(label="Title", value="none")
                    with gr.Row():
                        l3 = gr.Dropdown(lang_choices, value="Auto-detect", label="Language")
                        d3 = gr.Textbox(label="Domain", value="general")
                    ing_btn = gr.Button("Embed & Index", variant="primary")
                with gr.Column():
                    upl = gr.File(label="Or upload .txt / .md")
                    aud_btn = gr.Button("Audit data/raw (font gatekeeper)")
                    bat_btn = gr.Button("Run full batch (audit + ingest)")
            out3 = gr.Markdown()
            ing_btn.click(lambda *a: do_ingest_text(a[0], a[1], a[2], to_code(a[3]), a[4]),
                          inputs=[txt, did, ttl, l3, d3], outputs=out3)
            upl.upload(lambda f, i, l, d: do_upload(f, i, to_code(l), d),
                       inputs=[upl, did, l3, d3], outputs=out3)
            aud_btn.click(do_audit, outputs=out3)
            bat_btn.click(do_batch_ingest.__wrapped__ if hasattr(do_batch_ingest, "__wrapped__")
                          else lambda: do_batch_ingest(True), outputs=out3)

        with gr.Tab("📊 Corpus & Health"):
            st_btn = gr.Button("Refresh")
            st_out = gr.Markdown()
            st_btn.click(do_stats, outputs=st_out)
            demo.load(do_stats, outputs=st_out)

    return demo


def launch(settings=None, service: Optional[RagService] = None) -> None:
    cfg = settings or get_settings()
    build_studio(service).launch(
        server_name=cfg.studio.host,
        server_port=cfg.studio.port,
        share=cfg.studio.share,
    )


if __name__ == "__main__":
    launch()
