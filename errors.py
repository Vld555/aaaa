import torch
import torch.nn.functional as F
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm


class ErrorAnalyzer:
    def __init__(self, model_path):
        print(f"⏳ Загрузка модели из {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path)

        # Автовыбор железа (MPS для твоего Mac, CUDA или CPU)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.model.to(self.device)
        self.model.eval()
        print(f"🚀 Устройство: {self.device}")

    def get_probabilities(self, texts, batch_size=16, max_len=512, head_ratio=0.25):
        head_len = int((max_len - 2) * head_ratio)
        tail_len = (max_len - 2) - head_len

        all_probs = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Сбор вероятностей"):
            batch_texts = texts[i:i+batch_size]
            input_ids_list = []
            attention_mask_list = []

            # Тот же самый трюк Head-Tail, на котором училась модель
            for text in batch_texts:
                token_ids = self.tokenizer.encode(
                    str(text), add_special_tokens=False)
                if len(token_ids) > (max_len - 2):
                    token_ids = token_ids[:head_len] + token_ids[-tail_len:]

                input_ids = [self.tokenizer.cls_token_id] + \
                    token_ids + [self.tokenizer.sep_token_id]
                attention_mask = [1] * len(input_ids)

                pad_len = max_len - len(input_ids)
                input_ids.extend([self.tokenizer.pad_token_id] * pad_len)
                attention_mask.extend([0] * pad_len)

                input_ids_list.append(input_ids)
                attention_mask_list.append(attention_mask)

            input_ids_tensor = torch.tensor(
                input_ids_list, dtype=torch.long).to(self.device)
            mask_tensor = torch.tensor(
                attention_mask_list, dtype=torch.long).to(self.device)

            with torch.no_grad():
                outputs = self.model(
                    input_ids_tensor, attention_mask=mask_tensor)
                # Берем вероятности (Softmax) вместо жестких классов
                probs = F.softmax(outputs.logits, dim=-1)
                # Сохраняем вероятность принадлежности к Классу 1 (Positive)
                all_probs.extend(probs[:, 1].cpu().numpy())

        return all_probs


def print_worst_errors(df, top_n=5):
    print("\n" + "="*80)
    print("🔴 ТОП ЛОЖНОПОЛОЖИТЕЛЬНЫХ (False Positives)")
    print("На самом деле НЕГАТИВ (0), но модель уверена, что ПОЗИТИВ (1)")
    print("="*80)

    fp_df = df[(df['label'] == 0) & (df['pred_prob'] > 0.5)].copy()
    fp_df = fp_df.sort_values(by='pred_prob', ascending=False).head(top_n)

    for idx, row in fp_df.iterrows():
        print(
            f"\nУверенность модели в Позитиве: {row['pred_prob']:.4f} (А должно быть 0!)")
        # Выводим первую 1000 символов
        print(f"Текст:\n{row['text'][:1000]}...")
        print("-" * 40)

    print("\n" + "="*80)
    print("🔵 ТОП ЛОЖНООТРИЦАТЕЛЬНЫХ (False Negatives)")
    print("На самом деле ПОЗИТИВ (1), но модель уверена, что НЕГАТИВ (0)")
    print("="*80)

    fn_df = df[(df['label'] == 1) & (df['pred_prob'] < 0.5)].copy()
    fn_df = fn_df.sort_values(by='pred_prob', ascending=True).head(top_n)

    for idx, row in fn_df.iterrows():
        print(
            f"\nУверенность модели в Позитиве: {row['pred_prob']:.4f} (А должно быть 1!)")
        print(f"Текст:\n{row['text'][:1000]}...")
        print("-" * 40)


if __name__ == "__main__":
    # 1. Укажи путь к обученной модели (DistilBERT или MiniLM)
    MODEL_PATH = "/Users/vladharcenko/Desktop/ML/РСК/task4/distilbert"

    # 2. Укажи путь к отложенной выборке (2000 текстов)
    TEST_DATA_PATH = "/Users/vladharcenko/Desktop/ML/РСК/task4/distilbert/test_data_distilbert.csv"

    analyzer = ErrorAnalyzer(MODEL_PATH)
    df_test = pd.read_csv(TEST_DATA_PATH)

    # Получаем вероятности класса 1 для каждого текста
    df_test['pred_prob'] = analyzer.get_probabilities(
        df_test['text'].tolist(), batch_size=16)

    # Запускаем анализ
    print_worst_errors(df_test, top_n=5)
