from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

driver = webdriver.Chrome()
driver.get("https://www.moltbook.com/")

top_button = WebDriverWait(driver, 10).until(
    EC.element_to_be_clickable((By.XPATH, "//button[.//span[normalize-space()='Top']]"))
)
driver.execute_script("arguments[0].click();", top_button)

all_posts = set()

while True:
    WebDriverWait(driver, 10).until(
        EC.presence_of_all_elements_located((By.CSS_SELECTOR, "a[href^='/post/']"))
    )

    posts = driver.find_elements(By.CSS_SELECTOR, "a[href^='/post/']")

    for post in posts:
        try:
            href = post.get_attribute("href")
            if href:
                all_posts.add(href)
        except:
            continue

    print(f"Collected {len(all_posts)} posts")

    try:
        load_button = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Load More Posts')]"))
        )

        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", load_button)
        driver.execute_script("arguments[0].click();", load_button)

        time.sleep(2)

    except:
        print("No more Load More button — stopping")
        break

with open("posts.txt", "w", encoding="utf-8") as f:
    for post in all_posts:
        f.write(post + "\n")

print(f"Saved {len(all_posts)} posts")