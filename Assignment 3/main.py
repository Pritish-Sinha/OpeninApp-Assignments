import requests
from bs4 import BeautifulSoup
import json
import csv

def scrape_google_results(query):
    url = f"https://www.google.com/search?q={query}&num=10000"

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    response = requests.get(url, headers=headers)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, 'html.parser')
    search_results = soup.select('.g')

    results = []
    for result in search_results:
        link = result.select_one('a')
        if link:
            href = link['href']
            if "youtube.com/channel/" in href:
                channel_id = href[href.index('/channel/') + 9:]
                results.append({'channel_id': channel_id, 'link': f"https://www.youtube.com/channel/{channel_id}"})

    return results

def save_results_to_json(results, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(results, file, ensure_ascii=False, indent=4)

def save_results_to_csv(results, filename):
    fieldnames = ['channel_id', 'link']

    with open(filename, 'w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

# Example usage
query = 'site:youtube.com openinapp.co'
results = scrape_google_results(query)

save_results_to_json(results, 'results.json')
save_results_to_csv(results, 'results.csv')
