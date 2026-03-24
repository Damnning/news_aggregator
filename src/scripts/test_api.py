import pandas as pd
import requests
import json
from typing import List, Dict
import time

API_URL = "http://localhost:8000/llm/classify"
CSV_FILE = "../../data/test_data.csv"
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


def load_news_data(csv_file: str) -> List[str]:
    df = pd.read_csv(csv_file)
    return df['content'].tolist()


def classify_content(content: str, filter_prompt: str, temperature: float = 0) -> Dict:
    payload = {
        "content": content,
        "filter_prompt": filter_prompt,
        "temperature": temperature
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        return {
            "success": True,
            "is_relevant": result.get("is_relevant"),
            "error": None,
            "status_code": response.status_code,
            "response_time": response.elapsed.total_seconds()
        }
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "is_relevant": None,
            "error": "Timeout",
            "status_code": None,
            "response_time": None
        }
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "is_relevant": None,
            "error": str(e),
            "status_code": getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None,
            "response_time": None
        }


def run_tests(news_list: List[str], prompts: List[Dict], sample_size: int = 10) -> pd.DataFrame:
    results = []

    test_news = news_list[:sample_size]

    total_tests = len(test_news) * len(prompts)
    current_test = 0

    for news_idx, news_content in enumerate(test_news):
        for prompt_config in prompts:
            current_test += 1
            print(f"Тест {current_test}/{total_tests}: {prompt_config['name']} - Новость #{news_idx + 1}")

            result = classify_content(news_content, prompt_config['prompt'])

            results.append({
                "news_index": news_idx + 1,
                "запрос": news_content[:100] + "...",  # Первые 100 символов
                "полный_запрос": news_content,
                "filter_name": prompt_config['name'],
                "filter_prompt": prompt_config['prompt'],
                "результат": result['is_relevant'],
                "success": result['success'],
                "error": result['error'],
                "status_code": result['status_code'],
                "response_time_sec": result['response_time']
            })

            time.sleep(0.1)

    return pd.DataFrame(results)


def analyze_results(df: pd.DataFrame):
    print("\n" + "=" * 80)
    print("АНАЛИЗ РЕЗУЛЬТАТОВ ТЕСТИРОВАНИЯ")
    print("=" * 80)

    print(f"\nВсего тестов выполнено: {len(df)}")
    print(f"Успешных запросов: {df['success'].sum()}")
    print(f"Ошибок: {(~df['success']).sum()}")

    if df['success'].any():
        print(f"\nСреднее время ответа: {df[df['success']]['response_time_sec'].mean():.2f} сек")
        print(f"Максимальное время ответа: {df[df['success']]['response_time_sec'].max():.2f} сек")

    print("\n--- Распределение результатов по фильтрам ---")
    filter_stats = df.groupby('filter_name').agg({
        'результат': lambda x: f"{x.sum()}/{len(x)} релевантных",
        'success': lambda x: f"{x.sum()}/{len(x)} успешных",
        'response_time_sec': 'mean'
    })
    print(filter_stats)

    print("\n--- Ошибки ---")
    errors = df[~df['success']][['filter_name', 'error', 'status_code']]
    if len(errors) > 0:
        print(errors)
    else:
        print("Ошибок не обнаружено")

    print("\n--- Потенциальные проблемы ---")

    for filter_name in df['filter_name'].unique():
        filter_df = df[df['filter_name'] == filter_name]
        if filter_df['success'].all():
            unique_results = filter_df['результат'].unique()
            if len(unique_results) == 1:
                print(f"⚠️  '{filter_name}': всегда возвращает {unique_results[0]}")

    if df['success'].any():
        slow_threshold = 5.0  # секунд
        slow_requests = df[df['response_time_sec'] > slow_threshold]
        if len(slow_requests) > 0:
            print(f"⚠️  Обнаружено {len(slow_requests)} медленных запросов (>{slow_threshold}s)")


def save_results(df: pd.DataFrame):

    df.to_csv('test_results_full.csv', index=False, encoding='utf-8-sig')
    print(f"\n✅ Полные результаты сохранены в test_results_full.csv")

    simple_df = df[['запрос', 'filter_prompt', 'результат']]
    simple_df.to_csv('test_results_simple.csv', index=False, encoding='utf-8-sig')
    print(f"✅ Упрощенная версия сохранена в test_results_simple.csv")

    stats_df = df.groupby('filter_name').agg({
        'результат': ['sum', 'count', lambda x: (x.sum() / len(x) * 100)],
        'success': 'sum',
        'response_time_sec': ['mean', 'max', 'min']
    }).round(2)

    stats_df.columns = [
        'relevant_count', 'total_count', 'relevant_percent',
        'success_count', 'avg_time', 'max_time', 'min_time'
    ]
    stats_df.to_csv('test_statistics.csv', encoding='utf-8-sig')
    print(f"✅ Статистика сохранена в test_statistics.csv")

    errors_df = df[~df['success']][['filter_name', 'запрос', 'error', 'status_code']]
    if len(errors_df) > 0:
        errors_df.to_csv('test_errors.csv', index=False, encoding='utf-8-sig')
        print(f"✅ Ошибки сохранены в test_errors.csv")
    else:
        print(f"✅ Ошибок не обнаружено")

    pivot_df = df.pivot_table(
        index='полный_запрос',
        columns='filter_name',
        values='результат',
        aggfunc='first'
    )
    pivot_df.to_csv('test_results_by_news.csv', encoding='utf-8-sig')
    print(f"✅ Результаты по новостям сохранены в test_results_by_news.csv")


def main():
    print("Загрузка данных...")
    news_list = load_news_data(CSV_FILE)
    print(f"Загружено {len(news_list)} новостей")

    print("\nЗапуск тестирования...")
    print(f"Будет протестировано {len(TEST_PROMPTS)} фильтров на 10 новостях")

    results_df = run_tests(news_list, TEST_PROMPTS, sample_size=10)

    analyze_results(results_df)

    save_results(results_df)


if __name__ == "__main__":
    main()