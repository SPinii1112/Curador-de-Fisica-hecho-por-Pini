import streamlit as st
import tempfile
from pathlib import Path

import engine

# ---------------------------------------------------------------------------
# Config de pagina
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Curador de Pini",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS personalizado — paleta verde pizarra igual que el proyecto original
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&family=Source+Serif+4:opsz,wght@8..60,600&display=swap');

html, body, [class*="css"] { font-family: Manrope, sans-serif; }

.brand { font-size: 26px; font-weight: 800; color: #17221e; }
.brand span { color: #4b7514; }
.eyebrow { font-size: 11px; letter-spacing: .08em; color: #527416; font-weight: 700; text-transform: uppercase; }
.score-big { font-size: 52px; font-weight: 800; color: #547716; font-family: "Source Serif 4", serif; }
.match-card { background: #fff; border: 1px solid #d9ddd4; padding: 16px; border-radius: 4px; margin-bottom: 10px; }
.match-top { display: flex; justify-content: space-between; font-weight: 700; font-size: 14px; }
.match-score { color: #557a12; }
.match-text { color: #59645e; font-size: 13px; line-height: 1.6; margin-top: 10px; }
.pill { display: inline-block; padding: 4px 10px; background: #e7ebe3; border-radius: 99px; font-size: 12px; }
.hist-row { border-bottom: 1px solid #d9ddd4; padding: 10px 0; font-size: 13px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Cabecera
# ---------------------------------------------------------------------------
st.markdown('<p class="brand"><span>✦</span> Curador de Pini</p>', unsafe_allow_html=True)
st.markdown('<p class="eyebrow">Analisis semantico de videos y PDFs</p>', unsafe_allow_html=True)
st.divider()

# ---------------------------------------------------------------------------
# Sidebar — Biblioteca de referencia
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📚 Biblioteca de referencia")
    st.caption("Estos PDFs son la base contra la que se compara cada archivo analizado.")

    uploaded_refs = st.file_uploader(
        "Agregar PDFs a la biblioteca",
        type=["pdf"],
        accept_multiple_files=True,
        key="ref_uploader",
        help="Podes subir varios PDFs a la vez.",
    )

    if uploaded_refs:
        if st.button("➕ Incorporar a la biblioteca", use_container_width=True):
            with st.spinner("Procesando PDFs..."):
                resultados = []
                for f in uploaded_refs:
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                            tmp.write(f.read())
                            tmp_path = tmp.name
                        info = engine.add_reference_pdf(tmp_path, f.name)
                        Path(tmp_path).unlink(missing_ok=True)
                        resultados.append(f"✅ **{info['documento']}** — {info['fragmentos']} fragmentos")
                    except Exception as e:
                        resultados.append(f"❌ **{f.name}**: {e}")
                for r in resultados:
                    st.markdown(r)

    st.divider()

    # Lista de documentos en biblioteca
    refs = engine.references_summary()
    if refs:
        st.markdown(f"**{len(refs)} documento(s) en biblioteca:**")
        for r in refs:
            col1, col2 = st.columns([3, 1])
            col1.markdown(f"📄 {r['documento']}")
            col2.caption(f"{r['fragmentos']} frags.")
        st.divider()
        if st.button("🗑️ Vaciar biblioteca", use_container_width=True, type="secondary"):
            engine.clear_references()
            st.success("Biblioteca vaciada.")
            st.rerun()
    else:
        st.info("La biblioteca esta vacia. Subi al menos un PDF de referencia para empezar.")

# ---------------------------------------------------------------------------
# Tabs principales
# ---------------------------------------------------------------------------
tab_video, tab_pdf, tab_historial = st.tabs([
    "🎬 Analizar Video",
    "📄 Analizar PDF",
    "📊 Historial",
])

# ===========================================================================
# TAB 1 — Video local
# ===========================================================================
with tab_video:
    st.markdown("### Subi un video para analizar")
    st.caption(
        "El audio se transcribe con Whisper y se compara semanticamente contra tu biblioteca. "
        "Formatos soportados: MP4, MOV, MKV, AVI, WEBM, M4V."
    )

    video_file = st.file_uploader(
        "Elegir video",
        type=["mp4", "mov", "mkv", "avi", "webm", "m4v"],
        key="video_uploader",
    )

    calidad_label = st.radio(
        "Calidad de transcripcion",
        options=list(engine.WHISPER_SIZES.keys()),
        horizontal=True,
        help=(
            "**Rapido**: mucho mas veloz, ideal para CPU. "
            "**Equilibrado**: mejor calidad. "
            "**Preciso**: mas lento, no recomendado en Streamlit Cloud."
        ),
    )
    calidad = engine.WHISPER_SIZES[calidad_label]

    if st.button("▶ Transcribir y comparar", disabled=video_file is None, type="primary", key="btn_video"):
        if not engine.references_summary():
            st.error("Primero carga al menos un PDF de referencia en la barra lateral.")
        else:
            suffix = Path(video_file.name).suffix.lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(video_file.read())
                tmp_path = tmp.name

            try:
                with st.spinner(f"Transcribiendo con Whisper ({calidad_label})... esto puede tardar."):
                    texto = engine.transcribe_video(tmp_path, calidad)

                with st.spinner("Calculando similitud semantica..."):
                    resultado = engine.compare_text(texto)

                engine.save_result("Video", video_file.name, resultado)

                # --- Mostrar resultado ---
                st.success("Analisis completado")
                col_score, col_info = st.columns([1, 3])
                with col_score:
                    st.markdown(f'<p class="score-big">{resultado["promedio"]}%</p>', unsafe_allow_html=True)
                    st.caption("similitud media")
                with col_info:
                    st.markdown(f"**Archivo:** {video_file.name}")
                    st.markdown(f"**Fragmentos comparados:** {len(resultado['coincidencias'])}")

                st.divider()
                st.markdown("#### Mejores coincidencias")
                for m in resultado["coincidencias"]:
                    with st.expander(f"📄 {m['documento']} — Fragmento {m['fragmento']} · **{m['puntaje']}%**"):
                        st.write(m["texto"])

                with st.expander("Ver transcripcion completa"):
                    st.text(resultado["texto"])

            except Exception as e:
                st.error(str(e))
            finally:
                Path(tmp_path).unlink(missing_ok=True)

# ===========================================================================
# TAB 2 — PDF
# ===========================================================================
with tab_pdf:
    st.markdown("### Subi un PDF para analizar")
    st.caption(
        "Se extrae el texto del PDF y se compara contra tu biblioteca. "
        "Si el PDF es un escaneo sin OCR, el resultado puede ser incorrecto."
    )

    pdf_file = st.file_uploader(
        "Elegir PDF",
        type=["pdf"],
        key="pdf_uploader",
    )

    if st.button("🔍 Analizar PDF", disabled=pdf_file is None, type="primary", key="btn_pdf"):
        if not engine.references_summary():
            st.error("Primero carga al menos un PDF de referencia en la barra lateral.")
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(pdf_file.read())
                tmp_path = tmp.name

            try:
                with st.spinner("Extrayendo texto del PDF..."):
                    texto = engine.extract_pdf_text(tmp_path)

                with st.spinner("Calculando similitud semantica..."):
                    resultado = engine.compare_text(texto)

                engine.save_result("PDF", pdf_file.name, resultado)

                # --- Mostrar resultado ---
                st.success("Analisis completado")
                col_score, col_info = st.columns([1, 3])
                with col_score:
                    st.markdown(f'<p class="score-big">{resultado["promedio"]}%</p>', unsafe_allow_html=True)
                    st.caption("similitud media")
                with col_info:
                    st.markdown(f"**Archivo:** {pdf_file.name}")
                    st.markdown(f"**Fragmentos comparados:** {len(resultado['coincidencias'])}")

                st.divider()
                st.markdown("#### Mejores coincidencias")
                for m in resultado["coincidencias"]:
                    with st.expander(f"📄 {m['documento']} — Fragmento {m['fragmento']} · **{m['puntaje']}%**"):
                        st.write(m["texto"])

                with st.expander("Ver texto extraido del PDF"):
                    st.text(resultado["texto"][:3000] + ("..." if len(resultado["texto"]) > 3000 else ""))

            except Exception as e:
                st.error(str(e))
            finally:
                Path(tmp_path).unlink(missing_ok=True)

# ===========================================================================
# TAB 3 — Historial
# ===========================================================================
with tab_historial:
    st.markdown("### Historial de analisis")

    historial = engine.load_historial()
    if not historial:
        st.info("Todavia no hay analisis guardados. Analiza un video o PDF para empezar.")
    else:
        for entry in historial:
            fecha = entry["fecha"][:16].replace("T", " ")
            tipo_emoji = "🎬" if entry["tipo"] == "Video" else "📄"
            with st.expander(f"{tipo_emoji} **{entry['archivo']}** — {entry['promedio']}% similitud · {fecha} UTC"):
                st.caption(f"Tipo: {entry['tipo']} | Similitud media: **{entry['promedio']}%**")
                st.markdown("**Coincidencias:**")
                for m in entry["coincidencias"]:
                    st.markdown(f"- 📄 `{m['documento']}` Frag. {m['fragmento']} — **{m['puntaje']}%**")
