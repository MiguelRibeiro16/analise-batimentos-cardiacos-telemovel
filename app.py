import streamlit as st
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, detrend
import cv2
import tempfile
import os
import matplotlib.pyplot as plt

st.set_page_config(page_title="Analise PPG - Sinais e Sistemas", layout="wide")

st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        background-color: #1f3b57;
        color: white;
        border-radius: 8px;
        padding: 0.6rem 1.4rem;
        font-weight: 600;
        border: none;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        width: 100%;
    }
    .stButton>button:hover { background-color: #14283c; color: white; }
    div.block-container { padding-top: 1.5rem; padding-bottom: 1.5rem; }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        border-left: 4px solid #1f3b57;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("Analise PPG - Frequencia Cardiaca (BPM)")
st.markdown("Projeto de Sinais e Sistemas | Processamento de Sinal de Fotopletismografia")

with st.container():
    st.subheader("Instrucoes para Gravacao do Video")
    st.markdown("""
    Para assegurar a fidelidade do sinal PPG e minimizar ruido por interferencia externa:
    1. **Ative a lanterna (flash)** do telemovel antes de iniciar a gravacao (se gravar com a camara traseira e flash).
    2. **Cubra hermeticamente a camara e o flash** com a ponta do dedo, evitando folgas de luz ambiente.
    3. **Exerca uma pressao suave e constante**, evitando apertar excessivamente (o que restringe a perfusao sanguinea).
    4. **Mantenha o dedo completamente estavel** durante um intervalo de 10 a 30 segundos.
    """)

st.markdown("---")

st.sidebar.subheader("Configuracoes de Leitura")
forcar_fps_manual = st.sidebar.checkbox("Forcar FPS manual", value=False)
fps_manual = st.sidebar.number_input("Valor de FPS", value=30.0, step=0.1)

def extrair_sinal_ppg_path(caminho_video, forcar_manual, valor_manual):
    cap = cv2.VideoCapture(caminho_video)
    Fs = cap.get(cv2.CAP_PROP_FPS)
    
    if forcar_manual or Fs == 0 or np.isnan(Fs) or Fs > 120:
        Fs = valor_manual

    sinal_vermelho = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        h, w, _ = frame.shape
        margem_h = int(round(h * 0.35))
        margem_w = int(round(w * 0.35))
        
        r1 = max(0, int(round(h/2)) - margem_h)
        r2 = min(h, int(round(h/2)) + margem_h + 1)
        c1 = max(0, int(round(w/2)) - margem_w)
        c2 = min(w, int(round(w/2)) + margem_w + 1)

        roi_r = frame[r1:r2, c1:c2, 2].astype(float)
        sinal_vermelho.append(np.mean(roi_r))
    cap.release()

    sinal_vermelho = np.array(sinal_vermelho)
    sinal_r = -sinal_vermelho 
    tempo = np.arange(len(sinal_r)) / Fs
    return tempo, sinal_r, Fs

def processar_ppg_com_toolbox(sinal_bruto, Fs):
    sinal_bruto = np.array(sinal_bruto).flatten()
    N = len(sinal_bruto)
    if N < Fs:
        return 60.0, np.array([0, 1]), np.array([60, 60])

    sinal_detrend = detrend(sinal_bruto, type='linear')

    nyq = 0.5 * Fs
    low = 0.8 / nyq
    high = 3.5 / nyq
    b, a = butter(2, [low, high], btype='band')
    sinal_filt = filtfilt(b, a, sinal_detrend)

    sinal_janelado = sinal_filt * np.hamming(N)
    Nfft = 8192
    X_fft_complex = np.fft.fft(sinal_janelado, Nfft)
    freqs = np.arange(Nfft) * (Fs / Nfft)
    metade = Nfft // 2

    f_pos = freqs[:metade]
    X_pos = np.abs(X_fft_complex[:metade])

    idx_banda_util = np.where((f_pos >= 0.7) & (f_pos <= 3.0))[0]
    if len(idx_banda_util) > 0:
        max_i = np.argmax(X_pos[idx_banda_util])
        idx_real = idx_banda_util[max_i]
        bpm_fft = f_pos[idx_real] * 60
    else:
        bpm_fft = 70.0

    min_dist = max(1, int(round(0.33 * Fs)))
    locs, _ = find_peaks(sinal_filt, distance=min_dist, height=np.mean(sinal_filt))

    if len(locs) > 1:
        intervalos_medios = np.mean(np.diff(locs)) / Fs
        bpm_tempo = 60 / intervalos_medios
        bpm = (bpm_fft + bpm_tempo) / 2
    else:
        bpm = bpm_fft

    t_janela = 5
    t_passo = 1
    t_fim = N / Fs
    
    t_bpm = []
    bpms_hist = []
    
    t = 0
    while t <= (t_fim - t_janela):
        idx_ini = int(round(t * Fs))
        idx_fim = int(round((t + t_janela) * Fs))
        if idx_fim > N:
            break
        
        janela = sinal_filt[idx_ini:idx_fim]
        locs_j, _ = find_peaks(janela, distance=min_dist, height=np.mean(janela))
        
        if len(locs_j) > 1:
            bpm_janela = 60 / (np.mean(np.diff(locs_j)) / Fs)
            if 40 <= bpm_janela <= 220:
                t_bpm.append(t + t_janela / 2)
                bpms_hist.append(bpm_janela)
        t += t_passo
    
    if len(bpms_hist) == 0:
        t_bpm = [0, t_fim]
        bpms_hist = [bpm, bpm]

    return bpm, np.array(t_bpm), np.array(bpms_hist)

# Criacao de Abas para Escolher entre Upload ou Gravacao Direta
aba_upload, aba_gravacao = st.tabs(["📂 Carregar Ficheiros", "🎥 Gravar Vídeo (Webcam/Telemóvel)"])

videos_para_processar = [] # Lista de tuplos: (nome_ficheiro, caminho_temporario_ou_bytes)

with aba_upload:
    st.subheader("Carregamento de Ensaios Existentes")
    uploaded_files = st.file_uploader("Selecione os ficheiros de video (.mp4, .mov, .avi)", type=["mp4", "mov", "avi"], accept_multiple_files=True, key="uploader_ficheiros")
    if uploaded_files:
        for f in uploaded_files:
            videos_para_processar.append({"origem": f, "nome": f.name})

with aba_gravacao:
    st.subheader("Gravação Direta de Vídeo")
    st.markdown("Pode gravar um vídeo diretamente utilizando a câmara do seu dispositivo (computador ou telemóvel).")
    
    # Widget nativo do Streamlit para capturar vídeo da câmara
    video_gravado = st.camera_input("Grave o seu ensaio PPG (cubra a câmara com o dedo e ligue o flash se disponível)")
    
    nome_gravado = st.text_input("Nome/Identificação para este ensaio gravado:", value="Ensaio_Repouso_1")
    
    if video_gravado is not None:
        videos_para_processar.append({"origem": video_gravado, "nome": f"{nome_gravado}.mp4"})
        st.success(p_msg := f"Vídeo '{nome_gravado}' capturado com sucesso e pronto para processamento!")

st.markdown("---")

if len(videos_para_processar) > 0:
    if st.button("Processar Ensaios Selecionados/Gravados"):
        resultados = []
        
        with st.spinner("A processar os videos com o algoritmo sincronizado..."):
            for item in videos_para_processar:
                file_obj = item["origem"]
                file_name = item["nome"]
                
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(file_obj.read())
                tfile.close()
                
                tempo, sinal_bruto, Fs_utilizado = extrair_sinal_ppg_path(tfile.name, forcar_fps_manual, fps_manual)
                bpm, t_bpm, bpms_hist = processar_ppg_com_toolbox(sinal_bruto, Fs_utilizado)
                
                os.unlink(tfile.name)
                
                nome_lower = file_name.lower()
                if "repouso" in nome_lower or "rep" in nome_lower:
                    categoria = "Repouso"
                elif "exercicio" in nome_lower or "exe" in nome_lower or "pos" in nome_lower:
                    categoria = "Pos-Exercicio"
                else:
                    categoria = "Geral"

                resultados.append({
                    "nome": file_name,
                    "categoria": categoria,
                    "bpm_medio": bpm,
                    "fs_lido": Fs_utilizado,
                    "t_bpm": t_bpm,
                    "bpms_hist": bpms_hist
                })

        st.success("Processamento concluido com sucesso.")

        st.subheader("Estatisticas e Medias Globais por Categoria")
        
        categorias_unicas = list(set([r["categoria"] for r in resultados]))
        col_metrics = st.columns(len(categorias_unicas) if len(categorias_unicas) > 0 else 1)
        
        for idx, cat in enumerate(categorias_unicas):
            ensaios_cat = [r for r in resultados if r["categoria"] == cat]
            bpms_cat = [r["bpm_medio"] for r in ensaios_cat]
            media_cat = np.mean(bpms_cat)
            std_cat = np.std(bpms_cat) if len(bpms_cat) > 1 else 0.0
            
            with col_metrics[idx % len(col_metrics)]:
                st.markdown(f"""
                <div class="metric-card">
                    <h4 style="margin:0; color:#1f3b57;">Condicao: {cat}</h4>
                    <p style="margin:5px 0 0 0;color:#1f3b57; font-size:22px; font-weight:bold;">{media_cat:.1f} BPM</p>
                    <p style="margin:0; color:#666; font-size:12px;">Desvio Padrao: ±{std_cat:.1f} | Ensaios: {len(ensaios_cat)}</p>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        st.subheader("Evolucao da Frequencia Cardiaca ao Longo do Tempo")
        fig, ax = plt.subplots(figsize=(10, 4.5))
        
        for res in resultados:
            ax.plot(res["t_bpm"], res["bpms_hist"], linewidth=1.5, label=f"{res['nome']} ({res['bpm_medio']:.1f} BPM)")

        ax.set_title("Frequencia Cardiaca (BPM) - Janelas Moveis", fontsize=11, fontweight='bold')
        ax.set_xlabel("Tempo (segundos)", fontsize=10)
        ax.set_ylabel("Batimentos por Minuto (BPM)", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='best', fontsize=9)
        plt.tight_layout()
        
        st.pyplot(fig, use_container_width=True)

        st.subheader("Resumo Detalhado por Ensaio")
        for res in resultados:
            st.write(f"- **{res['nome']}** [{res['categoria']}]: **{res['bpm_medio']:.1f} BPM** (FPS utilizado: {res['fs_lido']:.1f})")
else:
    st.info("Carregue ficheiros de vídeo ou grave um novo vídeo na aba ao lado para iniciar a análise.")