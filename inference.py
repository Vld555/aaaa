import torch
import torch.nn.functional as F
import re
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, recall_score, precision_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class Analyzer:
    def __init__(self, model_path="/Users/vladharcenko/Desktop/ML/РСК/task4/results-3/mini_lm_model_v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval() 

    def clean_text(self, text):
        text = re.sub(r'<br\s*/?>', ' ', text)
        text = re.sub(r'[^a-zA-Z0-9\s.,!?\']', '', text)
        return re.sub(r'\s+', ' ', text).strip()

    def predict(self, raw_text, max_len=256, stride=64, top_k_chunks=3):
        text = self.clean_text(raw_text)
        inputs = self.tokenizer(text, return_tensors="pt", truncation=False)
        input_ids = inputs["input_ids"][0]
        
        chunks = []
        start = 0
        while start < len(input_ids):
            end = start + max_len
            chunk = input_ids[start:end]
            if len(chunk) < max_len:
                pad_tensor = torch.tensor([self.tokenizer.pad_token_id] * (max_len - len(chunk)))
                chunk = torch.cat([chunk, pad_tensor])
            chunks.append(chunk)
            start += (max_len - stride)
            if end >= len(input_ids): break

        chunk_probs = []
        with torch.no_grad(): 
            for chunk in chunks:
                chunk_input = chunk.unsqueeze(0)
                outputs = self.model(chunk_input)
                probs = F.softmax(outputs.logits, dim=-1)
                chunk_probs.append(probs[0][1].item()) 

        
        final_score = sum(chunk_probs) / len(chunk_probs)

        if final_score > 0.6:
            pred_label = 1
        else:
            pred_label = 0
        
        return {
            "pred_label": pred_label,
            "final_score": round(final_score, 4),
            "total_chunks": len(chunks),
            # "analyzed_top_k": top_k,
            "raw_chunk_probs": [round(p, 3) for p in chunk_probs]
        }

def test_longest_reviews(analyzer, df, top_n=3):
    print(f"{top_n} LONGEST TEXTS")
    
    df['text_len'] = df['text'].apply(len)
    longest_df = df.sort_values(by='text_len', ascending=False).head(top_n)
    
    for idx, row in longest_df.iterrows():
        text_snippet = row['text'][:2000] + "..."

        result = analyzer.predict(row['text'])
        
        print(f"\nДлина: {row['text_len']} символов")
        print(f"Реальный класс: {row['label']}")
        print(f"Скор: {result['final_score']})")
        print(f"Разбито на чанков: {result['total_chunks']}")
        print(f"Сырые вероятности чанков: {result['raw_chunk_probs']}")
        print(f"Текст (начало): {text_snippet}")
        print("-" * 60)

def evaluate_full_dataset(analyzer, df_test):
    print("Test")
    
    y_true = []
    y_pred = []
    for text, label in tqdm(zip(df_test['text'], df_test['label']), total=len(df_test), desc="Анализ текстов"):
        result = analyzer.predict(text)
        y_true.append(label)
        y_pred.append(result['pred_label'])
        
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc:.4f}")

    print(classification_report(y_true, y_pred))


import random
import textwrap
import torch
import torch.nn.functional as F

def inspect_random_chunks(analyzer, df, num_examples=3, max_len=256, stride=64, threshold=0.9):
    print(f'Random text analyze')
    
    sample_df = df.sample(num_examples)
    
    for idx, row in sample_df.iterrows():
        text = analyzer.clean_text(row['text'])
        
        inputs = analyzer.tokenizer(text, return_tensors="pt", truncation=False)
        input_ids = inputs["input_ids"][0]
        
        chunks = []
        start = 0
        while start < len(input_ids):
            end = start + max_len
            chunk = input_ids[start:end]
            if len(chunk) < max_len:
                pad_tensor = torch.tensor([analyzer.tokenizer.pad_token_id] * (max_len - len(chunk)))
                chunk = torch.cat([chunk, pad_tensor])
            chunks.append(chunk)
            start += (max_len - stride)
            if end >= len(input_ids): break
            
        batch_input = torch.stack(chunks)
        with torch.no_grad(): 
            outputs = analyzer.model(batch_input)
            probs = F.softmax(outputs.logits, dim=-1)
            chunk_probs = probs[:, 1].tolist()
            
        # 4. Итоги
        final_score = sum(chunk_probs) / len(chunk_probs)
        pred_label = final_score > threshold
        
        print(f"\n📝 ТЕКСТ (Индекс в датасете: {idx})")
        print(f"Реальный класс: {row['label']}")
        print(f"Предсказание:   {pred_label} (Средний скор: {final_score:.4f})")
        print(f"Всего чанков:   {len(chunks)}")
        print("-" * 60)
        
        for i, (chunk_tensor, prob) in enumerate(zip(chunks, chunk_probs)):
            chunk_text = analyzer.tokenizer.decode(chunk_tensor, skip_special_tokens=True)
            
            if len(chunk_text) > 150:
                display_text = chunk_text
            else:
                display_text = chunk_text
                
            if prob > 0.9:
                marker = "🟢 Явный ПОЗИТИВ"
            elif prob < 0.6:
                marker = "🔴 Явный НЕГАТИВ"
            else:
                marker = "🟡 НЕУВЕРЕННОСТЬ (Вода/Сюжет)"
                
            print(f"  Чанк {i+1}/{len(chunks)} | Скор: {prob:.4f} | {marker}")
            print(f"  Текст: {display_text}")
            print("  " + "."*58)
        print("="*80)

if __name__ == "__main__":
    analyzer = Analyzer()
    dataset_path = '/Users/vladharcenko/Desktop/ML/РСК/task4/results-3/test_data_2.csv' 
    df = pd.read_csv(dataset_path)
    
    test_longest_reviews(analyzer, df, top_n=4)

    # evaluate_full_dataset(analyzer, df)
    # find_best_threshold(analyzer, df)
    inspect_random_chunks(analyzer, df, num_examples=3, threshold=0.9)