import streamlit as st
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, GPT2LMHeadModel, GPT2Tokenizer
import re
import language_tool_python
from time import time
import textwrap
from nltk.translate.bleu_score import sentence_bleu

class CustomParaphraseGenerator:
    def __init__(self, model_name='t5-base'):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

    def paraphrase(self, text, max_length=512, min_length_ratio=0.8, num_return_sequences=1):
        words = re.findall(r'\b\w+\b', text)
        num_words = len(words)
        if num_words < 20 or num_words > 400:
            raise ValueError("Word limit should be between 20 and 400.")

        if len(text) > max_length:
            chunks = []
            chunk_size = max_length - 20
            for i in range(0, len(text), chunk_size):
                chunk = text[i:i + chunk_size]
                chunks.append(chunk)

            paraphrases = []
            for chunk in chunks:
                paraphrase_chunk = self._paraphrase_chunk(chunk, max_length, min_length_ratio, num_return_sequences)
                paraphrases.append(paraphrase_chunk)

            return ' '.join(paraphrases)
        else:
            return self._paraphrase_chunk(text, max_length, min_length_ratio, num_return_sequences)

    def _paraphrase_chunk(self, text, max_length, min_length_ratio, num_return_sequences):
        preprocess_text = "paraphrase: " + text + " </s>"
        tokenized_text = self.tokenizer.encode(preprocess_text, return_tensors="pt", max_length=max_length, truncation=True).to(device)

        min_length = int(len(tokenized_text[0]) * min_length_ratio)
        summary_ids = self.model.generate(
            tokenized_text,
            max_length=max_length,
            min_length=min_length,
            num_return_sequences=num_return_sequences,
            no_repeat_ngram_size=2,
            early_stopping=True,
            repetition_penalty=2.5,
            length_penalty=1.0,
            num_beams=4
        )

        output = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        output = self.correct_english(output)

        return output

    def correct_english(self, text):
        text = re.sub(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s*', lambda x: x.group(0).capitalize(), text)
        text = re.sub(r'\s([.,!?;:"](?:\s|$))', r'\1', text)
        return text

def correct_text(text):
    tool = language_tool_python.LanguageTool('en-US')
    matches = tool.check(text)
    corrected_text = tool.correct(text)
    return corrected_text

def calculate_bleu(reference, candidate):
    reference = [reference.split()]
    candidate = candidate.split()
    score = sentence_bleu(reference, candidate)
    return score

class GPT2ParaphraseGenerator:
    def __init__(self, model_name='gpt2', device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
        self.device = device

    def paraphrase(self, text, max_length=512, num_return_sequences=1):
        input_ids = self.tokenizer.encode(text, return_tensors="pt", max_length=max_length, truncation=True).to(self.device)

        output = self.model.generate(input_ids, max_length=max_length, num_return_sequences=num_return_sequences, no_repeat_ngram_size=2, num_beams=4)
        decoded_output = self.tokenizer.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

        return decoded_output

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cpg_generator = CustomParaphraseGenerator()
gpt2_generator = GPT2ParaphraseGenerator()

st.title("AI Text Transformer: Custom Paraphrasing Solutions")

user_input = st.text_area("Enter text here (200-400 words):", height=200)
generate_button = st.button("Generate Paraphrases")

if 'prev_input' not in st.session_state:
    st.session_state.prev_input = ""

if generate_button or st.session_state.prev_input:
    if not user_input:
        user_input = st.session_state.prev_input
    st.session_state.prev_input = user_input

    try:
        cpg_output = cpg_generator.paraphrase(user_input)
        cpg_corrected_output = correct_text(cpg_output)
        cpg_wrapped_text = textwrap.fill(cpg_corrected_output.strip(), width=100)

        gpt2_output = gpt2_generator.paraphrase(user_input)
        gpt2_wrapped_text = textwrap.fill(gpt2_output.strip(), width=100)

        st.subheader("Custom Paraphrase Generator Output (CPG):")
        st.text(cpg_wrapped_text)

        st.subheader("GPT-2 Based Paraphrase Generator Output:")
        st.text(gpt2_wrapped_text)

    except ValueError as ve:
        st.warning(f"Warning: {ve}")
