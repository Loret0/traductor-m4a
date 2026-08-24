import streamlit as st
from faster_whisper import WhisperModel
import tempfile
import os

st.set_page_config(page_title="Traductor M4A", page_icon="🎙️")

st.title("🎙️ Traductor de Audio M4A a Texto")
st.write("Sube tu archivo de audio en formato .m4a para procesarlo de forma privada.")

# Carga del modelo (se guarda en caché para no descargarlo cada vez)
@st.cache_resource
def load_model():
    return WhisperModel("small", device="cpu", compute_type="int8")

model = load_model()

uploaded_file = st.file_uploader("Elige un archivo .m4a", type=["m4a", "mp3", "wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/m4a")

    if st.button("Transcribir Audio"):
        with st.spinner("Procesando audio... Esto puede tardar unos minutos según la duración."):
            # Crear un archivo temporal para guardar el audio subido
            with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            try:
                # Transcribir
                segments, info = model.transcribe(tmp_path, beam_size=5)

                st.success(f"¡Listo! Idioma detectado: {info.language.upper()}")

                # Formatear el resultado
                texto_completo = ""
                for segment in segments:
                    linea = f"[{segment.start:.1f}s - {segment.end:.1f}s] {segment.text}\n"
                    texto_completo += linea

                # Mostrar resultado en pantalla
                st.text_area("Resultado de la transcripción", texto_completo, height=350)

                # Botón para descargar el texto
                st.download_button(
                    label="Descargar transcripción (.txt)",
                    data=texto_completo,
                    file_name="transcripcion.txt",
                    mime="text/plain"
                )
            finally:
                # Borrar el archivo temporal
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
