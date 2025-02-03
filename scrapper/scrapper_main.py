import requests
import os
from bs4 import BeautifulSoup


def scrape_documents(url):
    response = requests.get(url)
    print(f"Retrieved page with status code: {response.status_code}")
    if response.status_code != 200:
        print(f"Failed to retrieve the page. Status code: {response.status_code}")
        return False
    soup = BeautifulSoup(response.content, 'html.parser')
    content_elements = soup.find_all(['p', 'pre'])
    content_list = [element.get_text() for element in content_elements]
    document_content = "\n\n".join(content_list)
    name = url.replace("/", "_").replace(":", "_").replace(".","_").replace("https___docs_crewai_com_", "")
    save_dir = "scrapped_docs"
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, f"{name}.txt")
    with open(filepath, "w", encoding="utf-8") as file:
        file.write(document_content)
    print(f"Document saved as '{filepath}'.")


def recusively_scrape(hrefs):
    for href in hrefs:
        print(href)
        fromed_url = "https://docs.crewai.com" + href
        documents = scrape_documents(fromed_url)


def get_urls():
    url = 'https://docs.crewai.com/introduction'
    response = requests.get(url)
    print(f"Retrieved page with status code: {response}")
    if response.status_code != 200:
        print(f"Failed to retrieve the page. Status code: {
              response.status_code}")
        return
    soup = BeautifulSoup(response.content, 'html.parser')
    nav_div = soup.find(id="navigation-items")
    hrefs = [a["href"] for a in nav_div.find_all(
        "a") if a["href"] != "https://community.crewai.com" and a["href"] != "https://github.com/crewAIInc/crewAI/releases"]
    print(hrefs)
    print("Collected the following URLs : \n", hrefs[:5], "\n...")
    return hrefs


if __name__ == "__main__":
    hredfs = get_urls()
    recusively_scrape(hredfs)
