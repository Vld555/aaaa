import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import time


full_data_path = "/Users/vladharcenko/Desktop/ML/РСК/task4/11.csv" 
df = pd.read_csv(full_data_path)


df_train, df_test = train_test_split(df, test_size=2000, random_state=42)
print(f'train shpe: {len(df_train)}\nTest shape {len(df_test)}')

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<br\s*/?>', ' ', text)
    text = re.sub(r'[^a-z0-9\s.,!?\']', '', text)
    return re.sub(r'\s+', ' ', text).strip()

start_time = time.time()
train_texts = [clean_text(t) for t in df_train['text']]
test_texts = [clean_text(t) for t in df_test['text']]
print(f'Cleaning: {time.time() - start_time:.1f} s')


start_time = time.time()
vectorizer = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
)

X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)

y_train = df_train['label'].values
y_test = df_test['label'].values
print(f'vectorization time {time.time() - start_time:.1f} s')
print(f'Train shape{X_train.shape}')


print("LogReg traiing......")
start_time = time.time()

model = LogisticRegression(C=2.0, max_iter=1000, solver='liblinear', random_state=42)
model.fit(X_train, y_train)

print(f'Training {time.time() - start_time:.1f} s')


print('EVAL')
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f'Accuracy: {acc:.4f}\n')
print(classification_report(y_test, y_pred))