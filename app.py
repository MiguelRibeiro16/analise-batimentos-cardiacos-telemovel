import streamlit as st
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, detrend
import cv2
import tempfile
import os
import matplotlib.pyplot as plt

st.set_page_config(page_title="Análise PPG - Sinais e Sistemas", page_icon="❤️", layout="wide")

# --- DESIGN SYSTEM & CUSTOM CSS ---
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main { 
        background-color: #f4f6f9; 
    }
    
    /* Global Container Padding */
    div.block-container { 
        padding-top: 2rem; 
        padding-bottom: 2rem; 
        max-width: 1200px;
    }

    /* Modern Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        color: white;
        border-radius: 10px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        border: none;
        box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.2);
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton>button:hover { 
        background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%);
        box-shadow: 0 6px 8px -1px rgba(37, 99, 235, 0.3);
        transform: translateY(-1px);
        color: white; 
    }

    /* Metric & Info Cards */
    .metric-card {
        background: white;
        padding: 1.2rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        border: 1px solid #e5e7eb;
        border-left: 5px solid #2563eb;
        margin-bottom: 1rem;
        transition: transform 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05);
    }

    /* Instructions Box */
    .instruction-box {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        border: 1px solid #e5e7eb;
        margin-bottom: 2rem;
    }

    /* Headers styling */
    h1, h2, h3 {
        color: #1e293b;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER SECTION ---
st.title("❤️ Análise PPG — Frequência Cardíaca (BPM)")
st.markdown("##### **Projeto de Sinais e Sistemas** | Processamento Avançado de Sinal de Fotopletismografia")
st.markdown("---")

# --- INSTRUCTIONS CONTAINER ---
with st.container():
    st.markdown("""
    <div class="instruction-box">
        <h4 style="margin-top:0; color:#1e293b; font-weight:600;">📋 Instruções para Gravação Externa do Vídeo</h4>
        <ol style="margin-bottom:0; color:#475569; line-height: 1.6;">
            <li><b>Ative a lanterna (flash)</b> do telemóvel antes de iniciar a gravação na câmara nativa.</li>
            <li><b>Cubra hermeticamente a câmara e o flash</b> com a ponta do dedo, evitando entradas de luz ambiente.</li>
            <li><b>Exerça uma pressão suave e constante</b> (evite apertar demasiado para não cortar a perfusão sanguínea).</li>
            <li><b>Mantenha o dedo completamente estável</b> durante um intervalo de <b>10 a 30 segundos</b>.</li>
            <li><b>Transfira o ficheiro</b> e carregue-o na aplicação abaixo.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

# --- SIDEBAR CONFIGURATIONS ---
st.sidebar.subheader("⚙️ Configurações")
st.sidebar.markdown("Ajuste os parâmetros de leitura do sinal.")
forcar_fps_manual = st.sidebar.checkbox("Forçar FPS manual", value=False)
fps_manual = st.sidebar.number_input("Valor de FPS", value=30.0, step=0.1)

# --- FUNCTIONS ---
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

# --- FILE UPLOADER SECTION ---
st.subheader("📂 Carregar Ficheiros de Vídeo")
uploaded_files = st.file_uploader(
    "Arraste ou selecione os ficheiros de vídeo (.mp4, .mov, .avi)", 
    type=["mp4", "mov", "avi"], 
    accept_multiple_files=True, 
    key="uploader_ficheiros"
)

videos_para_processar = []
if uploaded_files:
    for f in uploaded_files:
        videos_para_processar.append({"origem": f, "nome": f.name})

st.markdown("<br>", unsafe_allow_html=True)

if len(videos_para_processar) > 0:
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        processar_clicado = st.button("🚀 Processar Ensaios Selecionados")
    
    if processar_clicado:
        resultados = []
        
        with st.spinner("A processar sinais PPG (Filtragem Butterworth + FFT + Picos)..."):
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
                    categoria = "Pós-Exercício"
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

        st.success("✨ Processamento concluído com sucesso!")
        st.markdown("---")

        # --- METRICS SECTION ---
        st.subheader("📊 Estatísticas e Médias Globais por Categoria")
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
                    <p style="margin:0; color:#64748b; font-size:13px; font-weight:600; text-transform: uppercase;">Condição: {cat}</p>
                    <p style="margin:8px 0 4px 0; color:#1e293b; font-size:28px; font-weight:700;">{media_cat:.1f} <span style="font-size:16px; font-weight:500; color:#64748b;">BPM</span></p>
                    <p style="margin:0; color:#64748b; font-size:12px;">Desvio Padrão: ±{std_cat:.1f} | Ensaios: {len(ensaios_cat)}</p>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # --- GRAPH SECTION ---
        st.subheader("📈 Evolução da Frequência Cardíaca ao Longo do Tempo")
        
        fig, ax = plt.subplots(figsize=(10, 4.5))
        
        # Style matplotlib to look clean & modern
        ax.set_facecolor('#ffffff')
        fig.patch.set_facecolor('#ffffff')
        
        for res in resultados:
            ax.plot(res["t_bpm"], res["bpms_hist"], linewidth=2, label=f"{res['nome']} ({res['bpm_medio']:.1f} BPM)", alpha=0.85)

        ax.set_title("Frequência Cardíaca (BPM) - Janelas Móveis", fontsize=12, fontweight='bold', color='#1e293b', pad=15)
        ax.set_xlabel("Tempo (segundos)", fontsize=10, color='#475569', labelpad=10)
        ax.set_ylabel("Batimentos por Minuto (BPM)", fontsize=10, color='#475569', labelpad=10)
        ax.grid(True, linestyle='--', alpha=0.4, color='#cbd5e1')
        ax.legend(loc='best', fontsize=9, frameon=True, facecolor='#f8fafc', edgecolor='#e2e8f0')
        
        # Clean spines
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_color('#cbd5e1')

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # --- DETAILED SUMMARY ---
        st.subheader("📋 Resumo Detalhado por Ensaio")
        for res in resultados:
            st.markdown(f"• **{res['nome']}** &nbsp;|&nbsp; Categoria: `{res['categoria']}` &nbsp;|&nbsp; Média: **{res['bpm_medio']:.1f} BPM** &nbsp;|&nbsp; *FPS utilizado:* {res['fs_lido']:.1f}")
else:
    st.info("👆 Carregue ficheiros de vídeo (.mp4, .mov, .avi) acima para iniciar a análise visual e numérica.")