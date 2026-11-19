import json, sys
import pandas as pd

baris = []
for arg in sys.argv[1:]:
    n, _, awalan = arg.partition(":")
    awalan = awalan or n
    d = pd.DataFrame(json.load(open(f"hasil_{n}.json", encoding="utf-8")))
    r = pd.read_csv(f"ragas_{awalan}_ringkasan.csv", encoding="utf-8-sig", index_col=0)
    err = d.jawaban_chatbot.str.startswith(("[ERROR]", "[GALAT]"))
    luar, salah = d[d.jenis == "di_luar_cakupan"], d[d.jenis == "salah_peran"]
    ok = d[(d.jenis == "normal") & ~err]
    baris.append({
        "model": d["model"].iloc[0] if "model" in d else n,
        "n_terskor": int(r["n"].min()),
        "faithfulness": r.loc["faithfulness", "mean"],
        "answer_relevancy": r.loc["answer_relevancy", "mean"],
        "factual_correctness": r.loc[[i for i in r.index if i.startswith("factual_correctness")][0], "mean"],
        "context_precision": r.loc["llm_context_precision_with_reference", "mean"],
        "context_recall": r.loc["context_recall", "mean"],
        "penolakan_luar_cakupan": luar.di_luar_cakupan.mean() if len(luar) else float("nan"),
        "penolakan_salah_peran": (salah.salah_peran.astype(bool).mean() if "salah_peran" in salah.columns
                                  else salah.jawaban_chatbot.str.startswith("Maaf, saya tidak dapat menjawabnya").mean())
                                 if len(salah) else float("nan"),
        "waktu_llm_rata2_detik": ok.waktu_ms.mean() / 1000 if "waktu_ms" in ok else float("nan"),
        "token_keluar_rata2": ok.token_keluar.mean() if "token_keluar" in ok else float("nan"),
        "soal_error_api": int(err.sum()),
    })
t = pd.DataFrame(baris).set_index("model").round(3)
print(t.T.to_string())
t.to_csv("perbandingan_model.csv", encoding="utf-8-sig")
