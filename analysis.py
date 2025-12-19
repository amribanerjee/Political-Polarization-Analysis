import nltk
import spacy
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import ipywidgets as w
from IPython.display import display
from xml.etree.cElementTree import XML
import zipfile
import os
import math

NS = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
P_TAG = NS + 'p'
T_TAG = NS + 't'

def extract(f):
    try:
        z = zipfile.ZipFile(f)
        xml = z.read('word/document.xml')
        z.close()
        tree = XML(xml)
        paras = []
        for p in tree.iter(P_TAG):
            frags = [t.text for t in p.iter(T_TAG) if t.text]
            if frags:
                paras.append(''.join(frags))
        return '\n'.join(paras)
    except:
        return None

nlp = spacy.load("en_core_web_md")
vader = SentimentIntensityAnalyzer()

def clean(t):
    doc = nlp(t.lower())
    return [tok.lemma_ for tok in doc if not tok.is_stop and not tok.is_punct and tok.is_alpha]

u1 = w.FileUpload(accept='.docx', multiple=False)
u2 = w.FileUpload(accept='.docx', multiple=False)
out = w.Output()

display(u1, u2, out)

def run(change):
    if u1.value and u2.value:
        with out:
            out.clear_output()
            
            i1 = next(iter(u1.value.values()))
            i2 = next(iter(u2.value.values()))
            
            n1, n2 = i1['metadata']['name'], i2['metadata']['name']
            
            with open(n1, 'wb') as f: f.write(i1['content'])
            with open(n2, 'wb') as f: f.write(i2['content'])

            d_txt = extract(n1)
            r_txt = extract(n2)

            os.remove(n1)
            os.remove(n2)

            if d_txt and r_txt:
                t1 = clean(d_txt)
                t2 = clean(r_txt)
                
                s1, s2 = set(t1), set(t2)
                
                jac = len(s1 & s2) / len(s1 | s2) if len(s1 | s2) > 0 else 0
                
                all_w = s1 | s2
                v_map = {word: i for i, word in enumerate(all_w)}
                v_size = len(all_w)
                
                idf = {}
                for word in all_w:
                    ct = (1 if word in s1 else 0) + (1 if word in s2 else 0)
                    idf[word] = 1 + math.log(2 / ct)

                def get_vec(tokens):
                    v = np.zeros(v_size)
                    tf = {}
                    for t in tokens: tf[t] = tf.get(t, 0) + 1
                    for t, c in tf.items():
                        if t in v_map:
                            v[v_map[t]] = c * idf.get(t, 0)
                    return v

                v1 = get_vec(t1)
                v2 = get_vec(t2)
                
                n_v1 = np.linalg.norm(v1)
                n_v2 = np.linalg.norm(v2)
                cos = np.dot(v1, v2) / (n_v1 * n_v2) if n_v1 > 0 and n_v2 > 0 else 0

                print("\n--- Detailed Analysis Report ---")
                print(f"\nJaccard Similarity (Vocabulary Overlap): {jac:.4f}")
                print(f"Cosine Similarity (Statistical Divergence): {cos:.4f}")

                print("\nSentiment Analysis (VADER Scores):")
                for p_label, text in [("democrat", d_txt), ("republican", r_txt)]:
                    sc = vader.polarity_scores(text)
                    print(f"  {p_label:<10} | Compound: {sc['compound']:.4f}, Pos: {sc['pos']:.4f}, Neu: {sc['neu']:.4f}, Neg: {sc['neg']:.4f}")

u1.observe(run, names='value')
u2.observe(run, names='value')
