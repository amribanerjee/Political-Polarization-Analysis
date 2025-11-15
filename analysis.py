nltk.download('vader_lexicon')
!python -m spacy download en_core_web_md
!python -m spacy download en_core_web_md
import nltk
import spacy
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import ipywidgets as widgets
from IPython.display import display
from xml.etree.cElementTree import XML
import zipfile
import os
import math

WORD_NAMESPACE = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
PARA_TAG = WORD_NAMESPACE + 'p'
TEXT_TAG = WORD_NAMESPACE + 't'

def extract_text_from_docx_file(file_path):
    try:
        docx_archive = zipfile.ZipFile(file_path)
        xml_content = docx_archive.read('word/document.xml')
        docx_archive.close()
        
        xml_tree = XML(xml_content)
        all_paragraphs = []
        
        for paragraph_element in xml_tree.iter(PARA_TAG):
            text_fragments = []
            for text_node in paragraph_element.iter(TEXT_TAG):
                if text_node.text:
                    text_fragments.append(text_node.text)
            
            if text_fragments:
                all_paragraphs.append(''.join(text_fragments))
        
        return '\n'.join(all_paragraphs)
    except Exception as e:
        print(f"Error reading DOCX file {file_path}: {e}")
        return None

def start_interactive_upload():
    print("Welcome! Please upload the two speeches you want to compare.")
    
    democrat_uploader = widgets.FileUpload(
        accept='.docx',
        multiple=False,
        description='Upload Democrat Speech DOCX'
    )
    republican_uploader = widgets.FileUpload(
        accept='.docx',
        multiple=False,
        description='Upload Republican Speech DOCX'
    )
    
    analysis_output_area = widgets.Output()

    display(democrat_uploader)
    display(republican_uploader)
    display(analysis_output_area)

    def handle_files_uploaded(change):
        if democrat_uploader.value and republican_uploader.value:
            
            with analysis_output_area:
                analysis_output_area.clear_output()
                print("Got the files! Processing now...")
            
                democrat_file_info = next(iter(democrat_uploader.value.values()))
                republican_file_info = next(iter(republican_uploader.value.values()))
                
                democrat_temp_filename = democrat_file_info['metadata']['name']
                republican_temp_filename = republican_file_info['metadata']['name']
                
                with open(democrat_temp_filename, 'wb') as f:
                    f.write(democrat_file_info['content'])
                with open(republican_temp_filename, 'wb') as f:
                    f.write(republican_file_info['content'])

                democrat_text = extract_text_from_docx_file(democrat_temp_filename)
                republican_text = extract_text_from_docx_file(republican_temp_filename)

                os.remove(democrat_temp_filename)
                os.remove(republican_temp_filename)

                if democrat_text and republican_text:
                    speech_analyzer = PoliticalSpeechAnalyzer(democrat_text, republican_text)
                    all_results = speech_analyzer.execute_full_analysis()
                    
                    print("\n--- Detailed Analysis Report ---")
                    
                    vocab_metrics = all_results['vocabulary_metrics']
                    semantic_metrics = all_results['semantic_metrics']
                    print(f"\nJaccard Similarity (Vocabulary Overlap): {vocab_metrics['jaccard_similarity']:.4f}")
                    print(f"Cosine Similarity (Statistical Divergence): {semantic_metrics['cosine_similarity']:.4f}")

                    sentiment_data = all_results['sentiment_analysis']
                    print("\nSentiment Analysis (VADER Scores):")
                    for party_name, scores in sentiment_data.items():
                        print(f"  {party_name.capitalize():<10} | Compound: {scores['compound']:.4f}, Pos: {scores['pos']:.4f}, Neu: {scores['neu']:.4f}, Neg: {scores['neg']:.4f}")
                                        
                else:
                    print("Uh oh. Couldn't extract text from one or both DOCX files. Check the file integrity!")

    democrat_uploader.observe(handle_files_uploaded, names='value')
    republican_uploader.observe(handle_files_uploaded, names='value')


class PoliticalSpeechAnalyzer:
    def __init__(self, democrat_speech_text, republican_speech_text):
        self.raw_speeches = {
            'democrat': democrat_speech_text, 
            'republican': republican_speech_text
        }
        
        self.nlp = spacy.load("en_core_web_md") 
        self.vader_sentiment_tool = SentimentIntensityAnalyzer()
        
        self.processed_tokens = {}
        self.processed_tokens['democrat'] = self._clean_tokens(self.raw_speeches['democrat'])
        self.processed_tokens['republican'] = self._clean_tokens(self.raw_speeches['republican'])
        
        self.unique_word_sets = {
            'democrat': set(self.processed_tokens['democrat']),
            'republican': set(self.processed_tokens['republican'])
        }
        
        self.idf_weights = self._calculate_idf()
        self.vocab_map, self.vocab_size = self._create_vocabulary_map()

    def _clean_tokens(self, text_to_process):
        doc = self.nlp(text_to_process.lower())
        
        clean_tokens = [
            token.lemma_ for token in doc 
            if not token.is_stop and not token.is_punct and token.is_alpha
        ]
        return clean_tokens

    def _create_vocabulary_map(self):
        all_unique_words = self.unique_word_sets['democrat'].union(self.unique_word_sets['republican'])
        vocab_map = {word: i for i, word in enumerate(all_unique_words)}
        return vocab_map, len(all_unique_words)

    def _calculate_idf(self):
        corpus_size = 2
        
        df = {}
        all_unique_words = self.unique_word_sets['democrat'].union(self.unique_word_sets['republican'])
        
        for word in all_unique_words:
            doc_count = 0
            if word in self.unique_word_sets['democrat']:
                doc_count += 1
            if word in self.unique_word_sets['republican']:
                doc_count += 1
            df[word] = doc_count
            
        idf_weights = {}
        for word, count in df.items():
            idf_weights[word] = 1 + math.log(corpus_size / count)
            
        return idf_weights

    def _calculate_pure_tfidf_vector(self, party):
        tokens = self.processed_tokens[party]
        vector = np.zeros(self.vocab_size)
        
        if not tokens:
            return vector

        tf = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1
            
        for token, count in tf.items():
            if token in self.vocab_map:
                tfidf_score = count * self.idf_weights.get(token, 0)
                index = self.vocab_map[token]
                vector[index] = tfidf_score
            
        return vector

    def analyze_sentiment(self):
        sentiment_results = {}
        for party_key, raw_text in self.raw_speeches.items():
            sentiment_results[party_key] = self.vader_sentiment_tool.polarity_scores(raw_text)
        return sentiment_results

    def calculate_jaccard_similarity(self):
        democrat_vocab = self.unique_word_sets['democrat']
        republican_vocab = self.unique_word_sets['republican']
        
        intersection_size = len(democrat_vocab.intersection(republican_vocab))
        union_size = len(democrat_vocab.union(republican_vocab))
        
        return intersection_size / union_size if union_size != 0 else 0

    def calculate_semantic_similarity(self):
        democrat_vector = self._calculate_pure_tfidf_vector('democrat')
        republican_vector = self._calculate_pure_tfidf_vector('republican')

        norm_dem = np.linalg.norm(democrat_vector)
        norm_rep = np.linalg.norm(republican_vector)

        if norm_dem == 0 or norm_rep == 0:
            return 0.0
        
        dot_product = np.dot(democrat_vector, republican_vector)
        cosine_similarity = dot_product / (norm_dem * norm_rep)
        
        return cosine_similarity

    def execute_full_analysis(self):
        final_results = {
            "sentiment_analysis": self.analyze_sentiment(),
            "vocabulary_metrics": {"jaccard_similarity": self.calculate_jaccard_similarity()},
            "semantic_metrics": {"cosine_similarity": self.calculate_semantic_similarity()},
        }
        return final_results

start_interactive_upload()
