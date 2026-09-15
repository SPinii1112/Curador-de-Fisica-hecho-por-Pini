# Curador de Pini ✦

Aplicacion de analisis semantico de **videos** y **PDFs** comparados contra una biblioteca de referencia propia.

Desarrollado con [Streamlit](https://streamlit.io/), [Whisper](https://github.com/openai/whisper) y [sentence-transformers](https://www.sbert.net/).

---

## ¿Que hace?

1. **Cargás PDFs** de referencia en tu biblioteca personal (apuntes, libros, papers).
2. **Subís un video** (MP4, MOV, MKV, AVI, WEBM) o **un PDF** a analizar.
3. La app **transcribe o extrae el texto**, lo compara semanticamente y te muestra los fragmentos de tu biblioteca mas relacionados, con porcentaje de similitud.

Todo el procesamiento ocurre en tu maquina (o en Streamlit Cloud). Los archivos no se almacenan permanentemente.

---

## Correr en local

### Requisitos previos

- Python 3.10 o superior
- [ffmpeg](https://ffmpeg.org/download.html) instalado y en PATH (necesario para transcribir videos)

### Instalacion

```bash
# Clonar el repositorio
git clone https://github.com/TU_USUARIO/curador-de-pini.git
cd curador-de-pini

# Crear entorno virtual (opcional pero recomendado)
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Mac/Linux

# Instalar dependencias
pip install -r requirements.txt

# Correr la app
streamlit run app.py
```

Abrí `http://localhost:8501` en el navegador.

---

## Deploy en Streamlit Cloud

1. Subí el repositorio a GitHub (los archivos de `data/` se ignoran automaticamente).
2. En [share.streamlit.io](https://share.streamlit.io), conecta tu repo y elegí `app.py` como archivo principal.
3. Streamlit Cloud instalara ffmpeg automaticamente gracias al archivo `packages.txt`.
4. ¡Listo! Compartí el link publico con quien quieras.

> **Nota:** En Streamlit Cloud la biblioteca de referencia se reinicia con cada sesion porque el servidor es efimero. Para uso persistente, corra la app localmente.

---

## Estructura del proyecto

```
curador-de-pini/
├── app.py              # Interfaz Streamlit
├── engine.py           # Motor de analisis (PDF, video, embeddings)
├── requirements.txt    # Dependencias Python
├── packages.txt        # Dependencias del sistema (ffmpeg para Streamlit Cloud)
├── .streamlit/
│   └── config.toml     # Tema visual
└── data/               # Carpeta local para base de conocimiento (ignorada por git)
```

---

## Calidad de transcripcion

| Modo | Modelo Whisper | Velocidad | Calidad |
|---|---|---|---|
| Rapido | `tiny` | ⚡⚡⚡ | Buena |
| Equilibrado | `base` | ⚡⚡ | Muy buena |
| Preciso | `small` | ⚡ | Excelente |

En Streamlit Cloud se recomienda **Rapido** por limitaciones de RAM.

---

## Licencia

MIT — usa y modifica libremente.
