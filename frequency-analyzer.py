import spacy
import numpy as np
import ipywidgets as widgets
from IPython.display import display
from xml.etree.cElementTree import XML
import zipfile
import os
from collections import Counter

# --- DOCX Extraction Utilities ---
WORD_NAMESPACE = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
PARA_TAG = WORD_NAMESPACE + 'p'
TEXT_TAG = WORD_NAMESPACE + 't'

# AMENDMENT: Updated ELECTION_YEARS to include every four years from 1944 through 2024
ELECTION_YEARS = list(range(1944, 2024 + 1, 4))
UPLOADER_WIDGETS = {}

# Initialize FileUpload widgets for each corpus slot
for year in ELECTION_YEARS:
    for party_abbr in ['R', 'D']:
        label = f"{year} - {party_abbr}"
        UPLOADER_WIDGETS[label] = widgets.FileUpload(
            accept='.docx',
            multiple=False,
            description=label
        )

def extract_text_from_docx_file(file_path):
    # Function to robustly read text from a DOCX file
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
        # Prints a specific error if file reading fails
        print(f"ERROR:FILE_READ:{file_path}:{e}")
        return None

# --- Main Analyzer Class ---
class CorpusAnalyzer:
    def __init__(self, document_map):
        self.raw_documents = document_map
        # Load the medium-sized English model from spaCy
        # NOTE: Make sure to run 'python -m spacy download en_core_web_md' if this fails
        try:
            self.nlp = spacy.load("en_core_web_md")
        except OSError:
            print("ERROR: spaCy model 'en_core_web_md' not found. Please install it.")
            self.nlp = None

        self.processed_tokens = {}
        if self.nlp:
            for name, text in self.raw_documents.items():
                self.processed_tokens[name] = self._clean_tokens(text)

        self.unique_word_sets = {
            name: set(tokens) for name, tokens in self.processed_tokens.items()
        }

    def _clean_tokens(self, text_to_process):
        # Tokenize, lemmatize, remove stopwords and punctuation
        doc = self.nlp(text_to_process.lower())
        clean_tokens = [
            token.lemma_ for token in doc
            if not token.is_stop and not token.is_punct and token.is_alpha
        ]
        return clean_tokens

    def get_corpus_word_frequency(self, n=10):
        all_tokens = []
        for tokens in self.processed_tokens.values():
            all_tokens.extend(tokens)

        total_tokens = len(all_tokens)

        if total_tokens == 0:
            return {
                "most_common_word": "N/A",
                "overall_percentage": 0.0,
                "document_frequencies": {},
                "top_n_corpus_words": []
            }

        word_counts = Counter(all_tokens)

        # Data point 1: Overall percentage of the single most common word
        # Handle case where word_counts might be empty if input was only stop/punct
        if not word_counts:
            return {
                "most_common_word": "N/A",
                "overall_percentage": 0.0,
                "document_frequencies": {},
                "top_n_corpus_words": []
            }

        most_common_word, overall_count = word_counts.most_common(1)[0]
        overall_percentage = (overall_count / total_tokens) * 100

        # Data point 2: Top N words across the entire corpus
        top_n_corpus_words = word_counts.most_common(n)

        # Data point 3: Breakdown of the single most common word by document
        doc_frequencies = {}
        for doc_name, tokens in self.processed_tokens.items():
            doc_total = len(tokens)
            doc_count = tokens.count(most_common_word)
            doc_percentage = (doc_count / doc_total) * 100 if doc_total > 0 else 0.0
            doc_frequencies[doc_name] = (doc_count, doc_percentage)

        return {
            "most_common_word": most_common_word,
            "overall_percentage": overall_percentage,
            "document_frequencies": doc_frequencies,
            "top_n_corpus_words": top_n_corpus_words
        }

    def execute_full_analysis(self):
        # Ensure NLP model loaded before proceeding
        if not self.nlp:
            return {"error": "NLP model not loaded. Cannot analyze."}

        final_results = {
            "overall_frequency": self.get_corpus_word_frequency(n=10)
        }
        return final_results

# --- Frontend and Widget Logic ---
class CorpusFrontend:
    def __init__(self):
        self.widget_list = list(UPLOADER_WIDGETS.values())
        self.analyze_button = widgets.Button(
            description="ANALYZE CORPUS",
            button_style='success',
            layout=widgets.Layout(width='auto')
        )
        self.analysis_output = widgets.Output()

        # Attach observer to every uploader widget
        for uploader in self.widget_list:
            uploader.observe(self._enforce_single_file_state, names='value')

    def _enforce_single_file_state(self, change):
        uploader = change['owner']
        new_value = change['new']

        # Clear output before printing debug/error messages
        # with self.analysis_output:
        #     self.analysis_output.clear_output()

        if len(new_value) > 1:
            try:
                # Get the latest uploaded file data (the last key in the dictionary)
                last_file_name = list(new_value.keys())[-1]
                last_file_data = new_value[last_file_name]
                single_file_dict = {last_file_name: last_file_data}

                # Temporarily unobserve to avoid recursion
                uploader.unobserve(self._enforce_single_file_state, names='value')

                # Force the internal value to only contain the single, latest file.
                # NOTE: We can only set 'value' here because the user is performing the action.
                # The read-only restriction mostly applies to clearing it later.
                uploader.value = single_file_dict

                # Re-observe
                uploader.observe(self._enforce_single_file_state, names='value')

            except Exception as e:
                with self.analysis_output:
                    print(f"ERROR:VISUAL_CORRECTION_FAILED:{type(e).__name__}:{e}")
                uploader.unobserve(self._enforce_single_file_state, names='value')
                # Clearing value via exception path still risks TraitError,
                # but we'll leave it as is since the primary fix is for analysis end.
                try:
                    uploader.value = {}
                except TraitError:
                    pass
                uploader.observe(self._enforce_single_file_state, names='value')

        elif len(new_value) == 1:
            # Normal single file upload
            pass # Suppressing debug spam for normal upload

        elif len(new_value) == 0:
            # The widget was cleared after analysis
            pass # Suppressing debug spam for clearing

    def _execute_analysis(self, b):
        document_map = {}

        with self.analysis_output:
            self.analysis_output.clear_output()
            print("STATUS:STARTING_ANALYSIS")

            for label, uploader in UPLOADER_WIDGETS.items():
                if not uploader.value:
                    continue

                # Retrieve the single file data (guaranteed to be length 1 by the observer)
                file_info = next(iter(uploader.value.values()))
                temp_filename = file_info['metadata']['name']

                # Save file locally for spaCy/XML extraction
                with open(temp_filename, 'wb') as f:
                    f.write(file_info['content'])

                text = extract_text_from_docx_file(temp_filename)

                # Cleanup the temporary file immediately
                os.remove(temp_filename)

                if text:
                    document_map[label] = text
                    print(f"STATUS:PROCESSED:{label}")
                else:
                    print(f"ERROR:TEXT_EXTRACTION:{label}")

                # --- WIDGET RESET FIX ---
                # Rely on resetting the internal file counter to visually clear the widget
                try:
                    uploader._counter = 0 
                except Exception as e:
                    print(f"ERROR:WIDGET_RESET_FAILED: Could not reset uploader {label}. {e}")
                # --- END FIX ---


            if len(document_map) < 1:
                print("ERROR:NO_DOCUMENTS_UPLOADED")
                return

            # --- Analysis execution block ---
            try:
                analyzer = CorpusAnalyzer(document_map)
                
                if "error" in analyzer.execute_full_analysis():
                    print("ERROR: Cannot perform analysis, spaCy model not ready.")
                    return

                all_results = analyzer.execute_full_analysis()

                print("\n--- CORPUS ANALYSIS RESULTS ---")

                overall_freq = all_results['overall_frequency']
                word = overall_freq['most_common_word']
                percentage = overall_freq['overall_percentage']
                top_corpus_words = overall_freq['top_n_corpus_words']

                if word == "N/A":
                     print("\nNo meaningful words found after cleaning and tokenization.")
                     return

                print(f"\nMost Common Word Across All Documents: '{word.upper()}' ({percentage:.2f}%)")

                print("\n--- FREQUENCY BREAKDOWN BY DOCUMENT ---")
                for doc_name, (count, doc_percentage) in overall_freq['document_frequencies'].items():
                    print(f"  {doc_name}: {count} mentions ({doc_percentage:.2f}%)")

                # Display the top 10 words across the entire combined corpus
                print("\n--- TOP 10 WORDS IN ENTIRE CORPUS (LEMMA) ---")
                for rank, (word, count) in enumerate(top_corpus_words, 1):
                    print(f"  {rank}. {word} (Total Count: {count})")

                print("\nSTATUS:ANALYSIS_COMPLETE")

            except Exception as e:
                # This will print the actual error if the analysis fails.
                print(f"ERROR:ANALYSIS_FAILED:{type(e).__name__}:{e}")


    def run(self):
        self.analyze_button.on_click(self._execute_analysis)

        # Layout the uploaders in two columns for better visibility
        upload_rows = []
        all_uploaders = list(UPLOADER_WIDGETS.values())
        # Since there are now many years, we'll use 4 columns for a more compact display
        NUM_COLUMNS = 4
        for i in range(0, len(all_uploaders), NUM_COLUMNS):
            row = widgets.HBox(all_uploaders[i:i+NUM_COLUMNS])
            upload_rows.append(row)

        header = widgets.HTML(value="<h2 style='color: #4B0082;'>Corpus Frequency Analyzer (1944-2024)</h2><p>Upload a .docx file for each corpus segment you wish to compare. The visual counter may show >1, but the system only retains the last file. **NOTE:** The upload widgets will clear automatically after successful analysis.</p>")

        display(widgets.VBox([
            header,
            widgets.VBox(upload_rows),
            self.analyze_button,
            self.analysis_output
        ]))

# Run the application
CorpusFrontend().run()
