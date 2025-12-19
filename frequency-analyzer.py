import spacy
import numpy as np
import ipywidgets as w
from IPython.display import display
from xml.etree.cElementTree import XML
import zipfile
import os
from collections import Counter

NS = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
P_TAG = NS + 'p'
T_TAG = NS + 't'

YRS = list(range(1944, 2025, 4))
UP_WIDGETS = {}

for y in YRS:
    for p in ['R', 'D']:
        lbl = f"{y} - {p}"
        UP_WIDGETS[lbl] = w.FileUpload(accept='.docx', multiple=False, description=lbl)

def get_docx_txt(path):
    try:
        z = zipfile.ZipFile(path)
        xml = z.read('word/document.xml')
        z.close()
        tree = XML(xml)
        paras = []
        for p in tree.iter(P_TAG):
            frags = [t.text for t in p.iter(T_TAG) if t.text]
            if frags:
                paras.append(''.join(frags))
        return '\n'.join(paras)
    except Exception as e:
        print(f"ERROR:FILE_READ:{path}:{e}")
        return None

try:
    nlp = spacy.load("en_core_web_md")
except:
    nlp = None

def clean(txt):
    if not nlp: return []
    d = nlp(txt.lower())
    return [t.lemma_ for t in d if not t.is_stop and not t.is_punct and t.is_alpha]

out = w.Output()
btn = w.Button(description="ANALYZE CORPUS", button_style='success', layout=w.Layout(width='auto'))

def reset_and_limit(change):
    u = change['owner']
    val = change['new']
    if len(val) > 1:
        u.unobserve(reset_and_limit, names='value')
        last_k = list(val.keys())[-1]
        u.value = {last_k: val[last_k]}
        u.observe(reset_and_limit, names='value')

for u in UP_WIDGETS.values():
    u.observe(reset_and_limit, names='value')

def run_analysis(b):
    docs = {}
    with out:
        out.clear_output()
        print("STATUS:STARTING_ANALYSIS")

        for lbl, u in UP_WIDGETS.items():
            if not u.value: continue

            inf = next(iter(u.value.values()))
            fname = inf['metadata']['name']
            
            with open(fname, 'wb') as f:
                f.write(inf['content'])

            txt = get_docx_txt(fname)
            os.remove(fname)

            if txt:
                docs[lbl] = txt
                print(f"STATUS:PROCESSED:{lbl}")
            
            try:
                u._counter = 0 
            except:
                pass

        if not docs:
            print("ERROR:NO_DOCUMENTS_UPLOADED")
            return

        if not nlp:
            print("ERROR: Cannot perform analysis, spaCy model not ready.")
            return

        all_tokens = []
        proc_map = {}
        for name, text in docs.items():
            tks = clean(text)
            proc_map[name] = tks
            all_tokens.extend(tks)

        if not all_tokens:
            print("\nNo meaningful words found after cleaning and tokenization.")
            return

        counts = Counter(all_tokens)
        top_word, tot_cnt = counts.most_common(1)[0]
        overall_pct = (tot_cnt / len(all_tokens)) * 100

        print("\n--- CORPUS ANALYSIS RESULTS ---")
        print(f"\nMost Common Word Across All Documents: '{top_word.upper()}' ({overall_pct:.2f}%)")

        print("\n--- FREQUENCY BREAKDOWN BY DOCUMENT ---")
        for d_name, tks in proc_map.items():
            d_tot = len(tks)
            d_cnt = tks.count(top_word)
            d_pct = (d_cnt / d_tot * 100) if d_tot > 0 else 0.0
            print(f"  {d_name}: {d_cnt} mentions ({d_pct:.2f}%)")

        print("\n--- TOP 10 WORDS IN ENTIRE CORPUS (LEMMA) ---")
        for r, (wrd, c) in enumerate(counts.most_common(10), 1):
            print(f"  {r}. {wrd} (Total Count: {c})")

        print("\nSTATUS:ANALYSIS_COMPLETE")

btn.on_click(run_analysis)

rows = []
all_ups = list(UP_WIDGETS.values())
for i in range(0, len(all_ups), 4):
    rows.append(w.HBox(all_ups[i:i+4]))

head = w.HTML("<h2>Corpus Frequency Analyzer (1944-2024)</h2>")
display(w.VBox([head, w.VBox(rows), btn, out]))
