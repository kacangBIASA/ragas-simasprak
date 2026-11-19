import argparse, json, os, re, sys, time
import pandas as pd
from ragas import evaluate, EvaluationDataset, RunConfig
from ragas.metrics import (Faithfulness, ResponseRelevancy, FactualCorrectness,
                           LLMContextPrecisionWithReference, LLMContextRecall)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

ap = argparse.ArgumentParser()
ap.add_argument("hasil", nargs="?", default="hasil_chatbot.json")
ap.add_argument("--awalan", default="", help="nama pendek model, mis. gptoss120b")
a = ap.parse_args()
suf = f"_{a.awalan}" if a.awalan else ""
HASIL, RINGKAS = f"ragas{suf}_per_pertanyaan.csv", f"ragas{suf}_ringkasan.csv"

def muat_env(path):
    isi = {}
    if os.path.isfile(path):
        for baris in open(path, encoding="utf-8-sig"):
            baris = baris.strip()
            if baris and not baris.startswith("#") and "=" in baris:
                k, v = baris.split("=", 1)
                v = v.strip().strip("\"'")
                if v:
                    isi[k.strip()] = v
                    os.environ.setdefault(k.strip(), v)
    return isi

AKAR = os.path.dirname(os.path.abspath(__file__))
lokal = muat_env(os.path.join(AKAR, ".env"))
if not os.getenv("GOOGLE_API_KEY") and lokal.get("GEMINI_API_KEY"):  
    os.environ["GOOGLE_API_KEY"] = lokal["GEMINI_API_KEY"]
if lokal.get("GOOGLE_API_KEY") or lokal.get("GEMINI_API_KEY"):
    print("Kunci penilai: dari ragas-eval/.env")
elif os.getenv("GOOGLE_API_KEY"):
    print("Kunci penilai: dari variabel terminal")
if not os.getenv("GOOGLE_API_KEY"):
    sys.exit("GOOGLE_API_KEY tidak ditemukan. Isi di ragas-eval/.env (GOOGLE_API_KEY=... atau GEMINI_API_KEY=...) "
             "atau atur variabel terminal. Kunci sistem di backend/.env sengaja tidak dipakai.")

JUDGE = os.getenv("JUDGE_MODEL", "gemini-3.8-flash")
EMB = os.getenv("EMB_MODEL", "models/gemini-embedding-001")
BATCH = int(os.getenv("BATCH", "4"))
JEDA = int(os.getenv("JEDA_BATCH", "45"))

df = pd.DataFrame(json.load(open(a.hasil, encoding="utf-8")))
df["jawaban_chatbot"] = df.jawaban_chatbot.str.replace(r"<think>.*?</think>", "", regex=True, flags=re.S).str.strip()
ERR = ("[ERROR]", "[GALAT]")

METRICS = [Faithfulness(), ResponseRelevancy(), FactualCorrectness(),
           LLMContextPrecisionWithReference(), LLMContextRecall()]
METRIK = [m.name for m in METRICS]

normal = df[(df.jenis == "normal") & (~df.jawaban_chatbot.str.startswith(ERR))].copy()
normal = normal[normal.konteks.map(len) > 0]
normal["id"] = normal["id"].astype(int)

lama = pd.read_csv(HASIL, encoding="utf-8-sig") if os.path.isfile(HASIL) else pd.DataFrame(columns=["id"] + METRIK)
for m in METRIK:
    if m not in lama.columns:
        lama[m] = float("nan")
selesai = set(lama[lama[METRIK].notna().all(axis=1)]["id"].astype(int))
sisa = normal[~normal["id"].isin(selesai)]
print(f"Soal dalam cakupan: {len(normal)} | sudah lengkap: {len(selesai & set(normal['id']))} | akan dinilai: {len(sisa)}")

if os.getenv("JUDGE_PROVIDER", "gemini").lower() == "openai":
    from langchain_openai import ChatOpenAI
    chat = ChatOpenAI(model=JUDGE, base_url=os.environ["JUDGE_BASE_URL"],
                      api_key=os.environ["JUDGE_API_KEY"], temperature=0)
else:
    from langchain_core.rate_limiters import InMemoryRateLimiter
    RPM = float(os.getenv("JUDGE_RPM", "12"))
    chat = ChatGoogleGenerativeAI(model=JUDGE, temperature=0,
                                  rate_limiter=InMemoryRateLimiter(requests_per_second=RPM / 60, max_bucket_size=1))
print("Penilai:", os.getenv("JUDGE_PROVIDER", "gemini"), JUDGE)
try:  
    llm = LangchainLLMWrapper(chat, bypass_n=True)
except TypeError:  
    llm = LangchainLLMWrapper(chat)

try:
    chat.invoke("Balas dengan satu kata: OK")
    GoogleGenerativeAIEmbeddings(model=EMB).embed_query("uji")
    print("Uji koneksi penilai dan embedding: OK")
except Exception as e:
    sys.exit(f"Uji koneksi GAGAL ({type(e).__name__}): {str(e)[:300]}\nPeriksa kunci di ragas-eval/.env, nama model (JUDGE_MODEL/EMB_MODEL), atau kuota.")
emb = LangchainEmbeddingsWrapper(GoogleGenerativeAIEmbeddings(model=EMB))
skor = {int(r["id"]): {m: r[m] for m in METRIK} for _, r in lama.iterrows()}

def simpan():
    tmp = normal[["id", "peran", "kategori", "pertanyaan"]].copy()
    for m in METRIK:
        tmp[m] = tmp["id"].map(lambda i: skor.get(int(i), {}).get(m, float("nan")))
    tmp.to_csv(HASIL, index=False, encoding="utf-8-sig")
    return tmp

for mulai in range(0, len(sisa), BATCH):
    b = sisa.iloc[mulai:mulai + BATCH]
    ds = EvaluationDataset.from_list([{
        "user_input": r.pertanyaan, "response": r.jawaban_chatbot,
        "retrieved_contexts": r.konteks, "reference": r.jawaban_acuan} for r in b.itertuples()])
    kurang = [m for m in METRICS if any(pd.isna(skor.get(int(i), {}).get(m.name, float("nan"))) for i in b["id"])]
    try:
        res = evaluate(ds, metrics=kurang, llm=llm, embeddings=emb,
                       run_config=RunConfig(max_workers=1, timeout=600, max_retries=8))
        sk = res.to_pandas()
        if mulai == 0:
            print("Kolom keluaran RAGAS:", list(sk.columns))
        def kolom(nama):  
            if nama in sk.columns:
                return nama
            return next((c for c in sk.columns if str(c).startswith(nama)), None)
        for i, (_, r) in enumerate(b.iterrows()):
            for m in kurang:
                c = kolom(m.name)
                v = sk.iloc[i][c] if c else float("nan")
                if pd.notna(v):
                    skor.setdefault(int(r["id"]), {})[m.name] = v
    except Exception as e:  
        print("Batch gagal:", type(e).__name__, str(e)[:120])
    tmp = simpan()
    print(f"Batch {mulai // BATCH + 1} selesai; terskor lengkap: {int(tmp[METRIK].notna().all(axis=1).sum())}/{len(tmp)}")
    if mulai + BATCH < len(sisa):
        time.sleep(JEDA)

out = simpan()
n_ok = out[METRIK].notna().sum()
print("\n=== Jumlah soal yang BERHASIL dinilai per metrik (dari %d) ===" % len(out))
print(n_ok.to_string())
if (n_ok < len(out)).any():
    print("PERINGATAN: sebagian skor kosong (batas laju/timeout). Jalankan ulang skrip ini sampai semua terisi sebelum dilaporkan.")
print("\n=== Rata-rata RAGAS ===")
agg = out[METRIK].agg(["mean", "std"]).round(3).T
agg["n"] = n_ok
print(agg)
print("\n=== Per kategori (rata-rata) ===")
print(out.groupby("kategori")[METRIK].mean().round(3))
agg.to_csv(RINGKAS, encoding="utf-8-sig")

luar = df[df.jenis == "di_luar_cakupan"]
salah = df[df.jenis == "salah_peran"]
acc_luar = luar.di_luar_cakupan.mean() if len(luar) else float("nan")
if len(salah) and "salah_peran" in salah.columns:
    acc_salah = salah.salah_peran.astype(bool).mean()
elif len(salah):
    acc_salah = salah.jawaban_chatbot.str.startswith("Maaf, saya tidak dapat menjawabnya").mean()
else:
    acc_salah = float("nan")
print(f"\nPenolakan benar, di luar cakupan: {acc_luar:.2%} ({len(luar)} soal)")
print(f"Penolakan benar, salah peran    : {acc_salah:.2%} ({len(salah)} soal)")
print("Soal dengan error API:", int(df.jawaban_chatbot.str.startswith(ERR).sum()))
