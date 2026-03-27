import pandas as pd
import requests
import json
from typing import List, Dict, Tuple
import time
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
import tiktoken

API_URL = "http://localhost:8000/llm/classify"
CSV_FILE = "../../data/test_data.csv"

# Маппинг названий фильтров на колонки в CSV
FILTER_TO_COLUMN = {
    "Политические новости России": "polit_ru",
    "Военные конфликты": "war",
    "Экономика и энергетика": "economics",
    "Технологии и интернет": "tech",
    "Только о США": "usa",
    "Только негативные новости": "negative",
    "Дипломатия и переговоры": "negotiations",
    "Социальные вопросы": "social",
    "Очень специфичный фильтр": "specific",
    "Абсурдный фильтр": "absurd",
    "Противоречивый фильтр": "contradiction",
    "Пустой контекст": "empty"
}

TEST_PROMPTS = [
    {
        "name": "Политические новости России",
        "prompt": "Определи, относится ли новость к внутренней политике России (законы, Госдума, министерства, внутренние инициативы)"
    },
    {
        "name": "Военные конфликты",
        "prompt": "Является ли эта новость о военных действиях, атаках, ударах или вооруженных конфликтах?"
    },
    {
        "name": "Экономика и энергетика",
        "prompt": "Связана ли новость с экономикой, энергоресурсами, торговлей или финансами?"
    },
    {
        "name": "Технологии и интернет",
        "prompt": "Относится ли новость к технологиям, интернету, мессенджерам или цифровым сервисам?"
    },
    {
        "name": "Только о США",
        "prompt": "Новость должна быть исключительно о событиях внутри США, без упоминания других стран"
    },
    {
        "name": "Только негативные новости",
        "prompt": "Является ли это новостью о негативных событиях (конфликты, катастрофы, преступления)?"
    },
    {
        "name": "Дипломатия и переговоры",
        "prompt": "Касается ли новость дипломатических отношений, переговоров между странами или международных соглашений?"
    },
    {
        "name": "Социальные вопросы",
        "prompt": "Относится ли новость к социальным вопросам: права человека, протесты, дискриминация, аборты?"
    },
    {
        "name": "Очень специфичный фильтр",
        "prompt": "Новость должна упоминать одновременно: Россию, законодательство и детей или женщин"
    },
    {
        "name": "Абсурдный фильтр",
        "prompt": "Новость о космических пришельцах, атакующих подводные базы"
    },
    {
        "name": "Противоречивый фильтр",
        "prompt": "Новость должна быть одновременно о мире и о войне"
    },
    {
        "name": "Пустой контекст",
        "prompt": "Релевантная новость"
    }
]


def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """Подсчёт токенов с использованием tiktoken"""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def load_news_data(csv_file: str) -> pd.DataFrame:
    """Загрузка данных с метками"""
    df = pd.read_csv(csv_file, sep=';')
    return df


def classify_content(content: str, filter_prompt: str, temperature: float = 0) -> Dict:
    """Классификация контента через API"""
    payload = {
        "content": content,
        "filter_prompt": filter_prompt,
        "temperature": temperature
    }

    # Подсчёт входных токенов
    input_text = content + " " + filter_prompt
    input_tokens = count_tokens(input_text)

    try:
        start_time = time.time()
        response = requests.post(API_URL, json=payload, timeout=30)
        response.raise_for_status()
        end_time = time.time()

        result = response.json()
        return {
            "success": True,
            "is_relevant": result.get("is_relevant"),
            "error": None,
            "status_code": response.status_code,
            "response_time": end_time - start_time,
            "input_tokens": input_tokens
        }
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "is_relevant": None,
            "error": "Timeout",
            "status_code": None,
            "response_time": None,
            "input_tokens": input_tokens
        }
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "is_relevant": None,
            "error": str(e),
            "status_code": getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None,
            "response_time": None,
            "input_tokens": input_tokens
        }


def run_tests(df: pd.DataFrame, prompts: List[Dict], sample_size: int = 10) -> pd.DataFrame:
    """Запуск тестов с сохранением ground truth меток"""
    results = []

    test_df = df.head(sample_size)

    total_tests = len(test_df) * len(prompts)
    current_test = 0

    for news_idx, row in test_df.iterrows():
        news_content = row['content']

        for prompt_config in prompts:
            current_test += 1
            filter_name = prompt_config['name']
            column_name = FILTER_TO_COLUMN.get(filter_name)

            # Получаем ground truth метку
            ground_truth = row.get(column_name, None)
            if ground_truth is not None:
                ground_truth = int(ground_truth)

            print(f"Тест {current_test}/{total_tests}: {filter_name} - Новость #{news_idx + 1}")

            result = classify_content(news_content, prompt_config['prompt'])

            # Преобразуем предсказание в int для сравнения
            prediction = None
            if result['is_relevant'] is not None:
                prediction = 1 if result['is_relevant'] else 0

            results.append({
                "news_index": news_idx + 1,
                "content": news_content[:100] + "..." if len(news_content) > 100 else news_content,
                "full_content": news_content,
                "filter_name": filter_name,
                "filter_prompt": prompt_config['prompt'],
                "prediction": prediction,
                "ground_truth": ground_truth,
                "success": result['success'],
                "error": result['error'],
                "status_code": result['status_code'],
                "response_time_sec": result['response_time'],
                "input_tokens": result['input_tokens']
            })

            time.sleep(0.1)

    return pd.DataFrame(results)


def calculate_metrics(y_true: List, y_pred: List) -> Dict:
    """Расчёт метрик классификации"""
    # Фильтруем None значения
    valid_indices = [i for i in range(len(y_true)) if y_true[i] is not None and y_pred[i] is not None]

    if len(valid_indices) == 0:
        return {
            "f1": None,
            "precision": None,
            "recall": None,
            "accuracy": None,
            "support": 0,
            "tp": 0, "fp": 0, "tn": 0, "fn": 0
        }

    y_true_filtered = [y_true[i] for i in valid_indices]
    y_pred_filtered = [y_pred[i] for i in valid_indices]

    # Проверяем, есть ли оба класса
    unique_true = set(y_true_filtered)
    unique_pred = set(y_pred_filtered)

    metrics = {
        "support": len(valid_indices),
        "positive_samples": sum(y_true_filtered),
        "negative_samples": len(y_true_filtered) - sum(y_true_filtered),
        "predicted_positive": sum(y_pred_filtered),
        "predicted_negative": len(y_pred_filtered) - sum(y_pred_filtered)
    }

    try:
        metrics["f1"] = f1_score(y_true_filtered, y_pred_filtered, zero_division=0)
        metrics["precision"] = precision_score(y_true_filtered, y_pred_filtered, zero_division=0)
        metrics["recall"] = recall_score(y_true_filtered, y_pred_filtered, zero_division=0)
        metrics["accuracy"] = accuracy_score(y_true_filtered, y_pred_filtered)

        # Confusion matrix
        if len(unique_true) == 2 or len(unique_pred) == 2:
            tn, fp, fn, tp = confusion_matrix(y_true_filtered, y_pred_filtered, labels=[0, 1]).ravel()
        else:
            tn, fp, fn, tp = 0, 0, 0, 0
            if 1 in unique_true and 1 in unique_pred:
                tp = sum(1 for t, p in zip(y_true_filtered, y_pred_filtered) if t == 1 and p == 1)
            if 0 in unique_true and 0 in unique_pred:
                tn = sum(1 for t, p in zip(y_true_filtered, y_pred_filtered) if t == 0 and p == 0)

        metrics.update({"tp": tp, "fp": fp, "tn": tn, "fn": fn})

    except Exception as e:
        print(f"Ошибка расчёта метрик: {e}")
        metrics.update({
            "f1": None, "precision": None, "recall": None, "accuracy": None,
            "tp": 0, "fp": 0, "tn": 0, "fn": 0
        })

    return metrics


def analyze_results(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Полный анализ результатов с метриками"""

    print("\n" + "=" * 100)
    print("АНАЛИЗ РЕЗУЛЬТАТОВ ТЕСТИРОВАНИЯ")
    print("=" * 100)

    # ==================== ОБЩАЯ СТАТИСТИКА ====================
    print("\n" + "-" * 50)
    print("ОБЩАЯ СТАТИСТИКА")
    print("-" * 50)

    total_tests = len(df)
    successful_tests = df['success'].sum()
    failed_tests = total_tests - successful_tests

    print(f"Всего тестов: {total_tests}")
    print(f"Успешных: {successful_tests} ({successful_tests / total_tests * 100:.1f}%)")
    print(f"Ошибок: {failed_tests} ({failed_tests / total_tests * 100:.1f}%)")

    # Общие метрики
    successful_df = df[df['success'] & df['ground_truth'].notna() & df['prediction'].notna()]

    if len(successful_df) > 0:
        overall_metrics = calculate_metrics(
            successful_df['ground_truth'].tolist(),
            successful_df['prediction'].tolist()
        )

        print(f"\n📊 ОБЩИЕ МЕТРИКИ КАЧЕСТВА:")
        print(f"   F1 Score:  {overall_metrics['f1']:.4f}" if overall_metrics['f1'] else "   F1 Score:  N/A")
        print(f"   Precision: {overall_metrics['precision']:.4f}" if overall_metrics[
            'precision'] else "   Precision: N/A")
        print(f"   Recall:    {overall_metrics['recall']:.4f}" if overall_metrics['recall'] else "   Recall:    N/A")
        print(
            f"   Accuracy:  {overall_metrics['accuracy']:.4f}" if overall_metrics['accuracy'] else "   Accuracy:  N/A")

        print(f"\n📈 СТАТИСТИКА ТОКЕНОВ:")
        print(f"   Всего входных токенов: {df['input_tokens'].sum():,}")
        print(f"   Среднее на запрос: {df['input_tokens'].mean():.1f}")
        print(f"   Мин/Макс: {df['input_tokens'].min()} / {df['input_tokens'].max()}")

        print(f"\n⏱️ СТАТИСТИКА ВРЕМЕНИ:")
        time_stats = df[df['success']]['response_time_sec']
        print(f"   Среднее время: {time_stats.mean():.3f} сек")
        print(f"   Медиана: {time_stats.median():.3f} сек")
        print(f"   Мин/Макс: {time_stats.min():.3f} / {time_stats.max():.3f} сек")
        print(f"   Общее время: {time_stats.sum():.1f} сек")

    # ==================== МЕТРИКИ ПО ФИЛЬТРАМ ====================
    print("\n" + "-" * 50)
    print("МЕТРИКИ ПО ФИЛЬТРАМ")
    print("-" * 50)

    filter_metrics = []

    for filter_name in df['filter_name'].unique():
        filter_df = df[(df['filter_name'] == filter_name) & df['success']]

        if len(filter_df) == 0:
            continue

        y_true = filter_df['ground_truth'].tolist()
        y_pred = filter_df['prediction'].tolist()

        metrics = calculate_metrics(y_true, y_pred)

        filter_metrics.append({
            "filter_name": filter_name,
            "f1": metrics['f1'],
            "precision": metrics['precision'],
            "recall": metrics['recall'],
            "accuracy": metrics['accuracy'],
            "support": metrics['support'],
            "tp": metrics['tp'],
            "fp": metrics['fp'],
            "tn": metrics['tn'],
            "fn": metrics['fn'],
            "positive_samples": metrics['positive_samples'],
            "predicted_positive": metrics['predicted_positive'],
            "avg_tokens": filter_df['input_tokens'].mean(),
            "total_tokens": filter_df['input_tokens'].sum(),
            "avg_time_sec": filter_df['response_time_sec'].mean(),
            "total_time_sec": filter_df['response_time_sec'].sum()
        })

    filter_metrics_df = pd.DataFrame(filter_metrics)

    if len(filter_metrics_df) > 0:
        # Сортируем по F1
        filter_metrics_df = filter_metrics_df.sort_values('f1', ascending=False)

        print("\n" + filter_metrics_df[['filter_name', 'f1', 'precision', 'recall', 'accuracy',
                                        'support', 'avg_tokens', 'avg_time_sec']].to_string(index=False))

    # ==================== МЕТРИКИ ПО НОВОСТЯМ ====================
    print("\n" + "-" * 50)
    print("МЕТРИКИ ПО НОВОСТЯМ")
    print("-" * 50)

    news_metrics = []

    for news_idx in df['news_index'].unique():
        news_df = df[(df['news_index'] == news_idx) & df['success']]

        if len(news_df) == 0:
            continue

        y_true = news_df['ground_truth'].tolist()
        y_pred = news_df['prediction'].tolist()

        metrics = calculate_metrics(y_true, y_pred)

        content_preview = news_df['content'].iloc[0] if len(news_df) > 0 else ""

        news_metrics.append({
            "news_index": news_idx,
            "content_preview": content_preview,
            "f1": metrics['f1'],
            "precision": metrics['precision'],
            "recall": metrics['recall'],
            "accuracy": metrics['accuracy'],
            "support": metrics['support'],
            "tp": metrics['tp'],
            "fp": metrics['fp'],
            "tn": metrics['tn'],
            "fn": metrics['fn'],
            "correct_predictions": metrics['tp'] + metrics['tn'],
            "total_predictions": metrics['support'],
            "avg_tokens": news_df['input_tokens'].mean(),
            "total_tokens": news_df['input_tokens'].sum(),
            "avg_time_sec": news_df['response_time_sec'].mean(),
            "total_time_sec": news_df['response_time_sec'].sum()
        })

    news_metrics_df = pd.DataFrame(news_metrics)

    if len(news_metrics_df) > 0:
        news_metrics_df = news_metrics_df.sort_values('f1', ascending=False)

        print("\n" + news_metrics_df[['news_index', 'f1', 'accuracy', 'correct_predictions',
                                      'total_predictions', 'avg_tokens', 'avg_time_sec']].to_string(index=False))

    # ==================== СВОДНАЯ ТАБЛИЦА (ФИЛЬТРЫ x НОВОСТИ) ====================
    print("\n" + "-" * 50)
    print("МАТРИЦА ПРЕДСКАЗАНИЙ (Фильтры x Новости)")
    print("-" * 50)

    # Создаём сводную таблицу с результатами
    pivot_predictions = df.pivot_table(
        index='news_index',
        columns='filter_name',
        values='prediction',
        aggfunc='first'
    )

    pivot_ground_truth = df.pivot_table(
        index='news_index',
        columns='filter_name',
        values='ground_truth',
        aggfunc='first'
    )

    # Создаём матрицу совпадений
    pivot_match = (pivot_predictions == pivot_ground_truth).astype(int)
    pivot_match = pivot_match.replace({1: '✓', 0: '✗'})

    print("\nСовпадение предсказаний с ground truth (✓ = верно, ✗ = ошибка):")
    print(pivot_match.to_string())

    # ==================== ДЕТАЛЬНЫЙ АНАЛИЗ ОШИБОК ====================
    print("\n" + "-" * 50)
    print("АНАЛИЗ ОШИБОК")
    print("-" * 50)

    errors_df = df[df['success'] & (df['prediction'] != df['ground_truth']) &
                   df['ground_truth'].notna() & df['prediction'].notna()]

    if len(errors_df) > 0:
        print(f"\nВсего ошибок классификации: {len(errors_df)}")

        # False Positives
        fp_df = errors_df[errors_df['prediction'] == 1]
        print(f"\nFalse Positives (предсказано 1, на самом деле 0): {len(fp_df)}")
        if len(fp_df) > 0:
            fp_by_filter = fp_df.groupby('filter_name').size().sort_values(ascending=False)
            print(fp_by_filter.to_string())

        # False Negatives
        fn_df = errors_df[errors_df['prediction'] == 0]
        print(f"\nFalse Negatives (предсказано 0, на самом деле 1): {len(fn_df)}")
        if len(fn_df) > 0:
            fn_by_filter = fn_df.groupby('filter_name').size().sort_values(ascending=False)
            print(fn_by_filter.to_string())
    else:
        print("Ошибок классификации не обнаружено!")

    # API ошибки
    api_errors = df[~df['success']]
    if len(api_errors) > 0:
        print(f"\nОшибки API: {len(api_errors)}")
        print(api_errors[['filter_name', 'news_index', 'error']].to_string(index=False))

    return {
        'filter_metrics': filter_metrics_df,
        'news_metrics': news_metrics_df,
        'pivot_predictions': pivot_predictions,
        'pivot_ground_truth': pivot_ground_truth,
        'errors': errors_df
    }


def save_results(df: pd.DataFrame, metrics: Dict[str, pd.DataFrame]):
    """Сохранение всех результатов"""

    print("\n" + "=" * 50)
    print("СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
    print("=" * 50)

    # 1. Полные результаты
    df.to_csv('test_results_full.csv', index=False, encoding='utf-8-sig')
    print("✅ test_results_full.csv - полные результаты всех тестов")

    # 2. Метрики по фильтрам
    if 'filter_metrics' in metrics and len(metrics['filter_metrics']) > 0:
        metrics['filter_metrics'].to_csv('metrics_by_filter.csv', index=False, encoding='utf-8-sig')
        print("✅ metrics_by_filter.csv - F1, precision, recall по каждому фильтру")

    # 3. Метрики по новостям
    if 'news_metrics' in metrics and len(metrics['news_metrics']) > 0:
        metrics['news_metrics'].to_csv('metrics_by_news.csv', index=False, encoding='utf-8-sig')
        print("✅ metrics_by_news.csv - метрики по каждой новости")

    # 4. Сводная таблица предсказаний
    if 'pivot_predictions' in metrics:
        metrics['pivot_predictions'].to_csv('pivot_predictions.csv', encoding='utf-8-sig')
        print("✅ pivot_predictions.csv - матрица предсказаний")

    # 5. Сводная таблица ground truth
    if 'pivot_ground_truth' in metrics:
        metrics['pivot_ground_truth'].to_csv('pivot_ground_truth.csv', encoding='utf-8-sig')
        print("✅ pivot_ground_truth.csv - матрица истинных меток")

    # 6. Сравнительная таблица
    if 'pivot_predictions' in metrics and 'pivot_ground_truth' in metrics:
        comparison = pd.DataFrame()
        for col in metrics['pivot_predictions'].columns:
            comparison[f'{col}_pred'] = metrics['pivot_predictions'][col]
            comparison[f'{col}_true'] = metrics['pivot_ground_truth'][col]
            comparison[f'{col}_match'] = (
                        metrics['pivot_predictions'][col] == metrics['pivot_ground_truth'][col]).astype(int)
        comparison.to_csv('comparison_detailed.csv', encoding='utf-8-sig')
        print("✅ comparison_detailed.csv - детальное сравнение")

    # 7. Ошибки
    if 'errors' in metrics and len(metrics['errors']) > 0:
        metrics['errors'].to_csv('classification_errors.csv', index=False, encoding='utf-8-sig')
        print("✅ classification_errors.csv - все ошибки классификации")

    # 8. Сводный отчёт
    summary = {
        'total_tests': len(df),
        'successful_tests': df['success'].sum(),
        'total_tokens': df['input_tokens'].sum(),
        'avg_tokens_per_request': df['input_tokens'].mean(),
        'total_time_sec': df[df['success']]['response_time_sec'].sum(),
        'avg_time_per_request': df[df['success']]['response_time_sec'].mean(),
    }

    # Добавляем общие метрики
    successful_df = df[df['success'] & df['ground_truth'].notna() & df['prediction'].notna()]
    if len(successful_df) > 0:
        overall = calculate_metrics(
            successful_df['ground_truth'].tolist(),
            successful_df['prediction'].tolist()
        )
        summary.update({
            'overall_f1': overall['f1'],
            'overall_precision': overall['precision'],
            'overall_recall': overall['recall'],
            'overall_accuracy': overall['accuracy'],
            'total_tp': overall['tp'],
            'total_fp': overall['fp'],
            'total_tn': overall['tn'],
            'total_fn': overall['fn']
        })

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv('summary_report.csv', index=False, encoding='utf-8-sig')
    print("✅ summary_report.csv - сводный отчёт")

    # 9. JSON отчёт для программной обработки
    json_report = {
        'summary': summary,
        'filter_metrics': metrics['filter_metrics'].to_dict('records') if 'filter_metrics' in metrics else [],
        'news_metrics': metrics['news_metrics'].to_dict('records') if 'news_metrics' in metrics else []
    }

    with open('report.json', 'w', encoding='utf-8') as f:
        json.dump(json_report, f, ensure_ascii=False, indent=2, default=str)
    print("✅ report.json - отчёт в JSON формате")


def main():
    print("=" * 100)
    print("ТЕСТИРОВАНИЕ КЛАССИФИКАТОРА С ОЦЕНКОЙ F1")
    print("=" * 100)

    print("\n📂 Загрузка данных...")
    df = load_news_data(CSV_FILE)
    print(f"   Загружено {len(df)} новостей")
    print(f"   Колонки: {list(df.columns)}")

    print("\n🚀 Запуск тестирования...")
    print(f"   Фильтров: {len(TEST_PROMPTS)}")
    print(f"   Новостей: 10")
    print(f"   Всего тестов: {len(TEST_PROMPTS) * 10}")

    results_df = run_tests(df, TEST_PROMPTS, sample_size=10)

    # Анализ результатов
    metrics = analyze_results(results_df)

    # Сохранение
    save_results(results_df, metrics)

    print("\n" + "=" * 100)
    print("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
    print("=" * 100)


if __name__ == "__main__":
    main()