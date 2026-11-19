"""Context relevance (Es et al., 2024) untuk konteks hasil retrieval chatbot.

CR = jumlah kalimat konteks yang relevan dan berhasil diekstraksi / jumlah seluruh kalimat dalam konteks c(q)

Metrik ini tidak memakai jawaban chatbot maupun jawaban acuan, hanya pertanyaan dan konteks hasil retrieval.
Konteks pada ketiga model identik (39 dari 39 soal), sehingga cukup dijalankan SATU kali pada salah satu berkas hasil.

Penilai: sama dengan evaluate_ragas.py (Gemini dari ragas-eval/.env, atau JUDGE_PROVIDER=openai). Pakai JUDGE_MODEL yang sama.

python evaluate_context_relevance.py hasil_gptoss120b.json
Hasil: context_relevance_per_pertanyaan.csv dan context_relevance_ringkasan.csv.
Aman dijalankan berulang: soal yang sudah terisi dilewati.
"""
import argparse, json, os, re, sys, time
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("hasil", nargs="?", default="hasil_gptoss120b.json")
ap.add_argument("--id", nargs="*", default=[], help="hitung ulang hanya soal dengan id ini (untuk audit), mis. --id 1 5 11 23")
a = ap.parse_args()
HASIL, RINGKAS = "context_relevance_per_pertanyaan.csv", "context_relevance_ringkasan.csv"
JEDA = float(os.getenv("JEDA_PANGGILAN", "6")) 

def muat_env(path):
    if os.path.isfile(path):
        for baris in open(path, encoding="utf-8-sig"):
            baris = baris.strip()
            if baris and not baris.startswith("#") and "=" in baris:
                k, v = baris.split("=", 1)
                v = v.strip().strip("\"'")
                if v:
                    os.environ.setdefault(k.strip(), v)


muat_env(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
if not os.getenv("GOOGLE_API_KEY") and os.getenv("GEMINI_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

JUDGE = os.getenv("JUDGE_MODEL", "gemini-3.8-flash")
if os.getenv("JUDGE_PROVIDER", "gemini").lower() == "openai":
    from langchain_openai import ChatOpenAI
    chat = ChatOpenAI(model=JUDGE, base_url=os.environ["JUDGE_BASE_URL"],
                      api_key=os.environ["JUDGE_API_KEY"], temperature=0)
else:
    if not os.getenv("GOOGLE_API_KEY"):
        sys.exit("GOOGLE_API_KEY tidak ditemukan (isi ragas-eval/.env seperti pada evaluate_ragas.py).")
    from langchain_google_genai import ChatGoogleGenerativeAI
    chat = ChatGoogleGenerativeAI(model=JUDGE, temperature=0)
print("Penilai:", os.getenv("JUDGE_PROVIDER", "gemini"), JUDGE)

PROMPT = """Please extract relevant sentences from the provided context that can potentially help answer the following question.
While extracting candidate sentences you're not allowed to make any changes to sentences from the given context.
Write each extracted sentence on its own line, with no numbering and no other text.
If no relevant sentences are found, or if you believe the question cannot be answered from the given context, return the phrase "Insufficient Information".

question: {q}
context: {c}"""


def kalimat(teks):
    """Pecah teks menjadi kalimat: batas pada baris baru atau tanda . ! ? yang diikuti spasi."""
    out = []
    for bagian in re.split(r"[\r\n]+", teks):
        for k in re.split(r"(?<=[.!?])\s+", bagian.strip()):
            k = k.strip()
            if len(k) > 1:
                out.append(k)
    return out


def teks_jawaban(r):
    c = r.content
    if isinstance(c, list):  
        c = "".join(x.get("text", "") if isinstance(x, dict) else str(x) for x in c)
    return c


data = [r for r in json.load(open(a.hasil, encoding="utf-8")) if r.get("jenis") == "normal" and r.get("konteks")]
if a.id:
    data = [r for r in data if str(r["id"]) in set(a.id)]
lama = pd.read_csv(HASIL, encoding="utf-8-sig") if os.path.isfile(HASIL) else pd.DataFrame(
    columns=["id", "kategori", "pertanyaan", "kalimat_konteks", "kalimat_terambil", "context_relevance"])
sudah = {int(i) for i in lama.loc[lama.context_relevance.notna(), "id"]} - {int(x) for x in a.id}
if a.id:
    data_semua = None
baris = {int(r["id"]): r for _, r in lama.iterrows()}
print(f"Soal dalam cakupan: {len(data)} | sudah terisi: {len(sudah)} | akan dinilai: {len([d for d in data if int(d['id']) not in sudah])}")


def simpan():
    df = pd.DataFrame(sorted(baris.values(), key=lambda r: int(r["id"])))
    df.to_csv(HASIL, index=False, encoding="utf-8-sig")
    return df


for d in data:
    i = int(d["id"])
    if i in sudah:
        continue
    ctx = "\n".join(d["konteks"])
    semua = kalimat(ctx)
    for percobaan in range(6):
        try:
            hasil = teks_jawaban(chat.invoke(PROMPT.format(q=d["pertanyaan"], c=ctx))).strip()
            break
        except Exception as e:
            print(f"  soal {i}: percobaan {percobaan + 1} gagal ({type(e).__name__}); tunggu")
            time.sleep(20 * (percobaan + 1))
    else:
        print(f"Soal {i} dilewati (tetap gagal); jalankan ulang nanti.")
        continue
    ambil = [] if "insufficient information" in hasil.lower() else kalimat(hasil)
    cr = min(len(ambil) / len(semua), 1.0) if semua else float("nan")
    baris[i] = {"id": i, "kategori": d["kategori"], "pertanyaan": d["pertanyaan"],
                "kalimat_konteks": len(semua), "kalimat_terambil": len(ambil), "context_relevance": round(cr, 4),
                "kalimat_terpilih": " || ".join(ambil)}
    simpan()
    print(f"soal {i}: {len(ambil)}/{len(semua)} kalimat -> CR = {cr:.3f}")
    time.sleep(JEDA)

df = simpan()
ok = df[df.context_relevance.notna()]
ring = pd.DataFrame([{"jumlah_soal": len(ok), "rata_rata": ok.context_relevance.mean(), "sd": ok.context_relevance.std(),
                      "median": ok.context_relevance.median(), "min": ok.context_relevance.min(), "maks": ok.context_relevance.max()}])
ring.to_csv(RINGKAS, index=False, encoding="utf-8-sig")
print("\nRingkasan:\n", ring.round(3).to_string(index=False))
print("Per kategori:\n", ok.groupby("kategori").context_relevance.agg(["count", "mean"]).round(3).to_string())
