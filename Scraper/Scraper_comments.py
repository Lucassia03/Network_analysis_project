from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from urllib.parse import urljoin
import sqlite3
import time


BASE_URL = "https://www.moltbook.com"


def create_connection(db_path="moltbook.db"):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def create_tables(conn):
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS posts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            post_url TEXT NOT NULL UNIQUE,
            author TEXT,
            author_link TEXT,
            title TEXT,
            body TEXT,
            scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS comments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            post_id INTEGER NOT NULL,
            parent_comment_id INTEGER,
            depth INTEGER NOT NULL DEFAULT 0,
            comment_order INTEGER,
            author TEXT,
            author_link TEXT,
            body TEXT,
            scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (post_id) REFERENCES posts(id) ON DELETE CASCADE,
            FOREIGN KEY (parent_comment_id) REFERENCES comments(id) ON DELETE CASCADE
        )
    """)

    conn.commit()


def save_post(conn, post_data):
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO posts (post_url, author, author_link, title, body)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(post_url) DO UPDATE SET
            author=excluded.author,
            author_link=excluded.author_link,
            title=excluded.title,
            body=excluded.body
    """, (
        post_data["post_url"],
        post_data["author"],
        post_data["author_link"],
        post_data["title"],
        post_data["body"],
    ))

    conn.commit()
    cur.execute("SELECT id FROM posts WHERE post_url = ?", (post_data["post_url"],))
    return cur.fetchone()[0]


def save_comments(conn, post_id, comments):
    cur = conn.cursor()
    cur.execute("DELETE FROM comments WHERE post_id = ?", (post_id,))
    conn.commit()

    # map local temp ids -> sqlite row ids
    temp_to_db = {}

    for c in comments:
        parent_db_id = None
        if c["parent_temp_id"] is not None:
            parent_db_id = temp_to_db.get(c["parent_temp_id"])

        cur.execute("""
            INSERT INTO comments (
                post_id,
                parent_comment_id,
                depth,
                comment_order,
                author,
                author_link,
                body
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            post_id,
            parent_db_id,
            c["depth"],
            c["comment_order"],
            c["author"],
            c["author_link"],
            c["body"],
        ))

        temp_to_db[c["temp_id"]] = cur.lastrowid

    conn.commit()


def scrape_main_post(driver, post_url):
    driver.get(post_url)

    WebDriverWait(driver, 15).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "h1"))
    )

    title = driver.find_element(By.CSS_SELECTOR, "h1").text.strip()

    main_post = driver.find_element(
        By.XPATH,
        "//h1/ancestor::div[.//div[contains(@class,'prose')] and .//a[starts-with(@href,'/u/')]][1]"
    )

    author_links = main_post.find_elements(By.CSS_SELECTOR, "a[href^='/u/']")
    post_author = author_links[0].text.strip() if author_links else ""
    post_author_href = (
        urljoin(BASE_URL, author_links[0].get_attribute("href"))
        if author_links else ""
    )

    paragraphs = main_post.find_elements(By.CSS_SELECTOR, "div.prose p")
    post_text = "\n\n".join(
        p.text.strip() for p in paragraphs if p.text.strip()
    )

    return {
        "post_url": post_url,
        "author": post_author,
        "author_link": post_author_href,
        "title": title,
        "body": post_text,
    }


def extract_comment_from_py2(comment_block, temp_id, parent_temp_id, depth, comment_order):
    """
    comment_block is the direct div.py-2 representing one actual comment.
    """
    body_candidates = comment_block.find_elements(
        By.XPATH,
        "./div[contains(@class,'text-sm') and contains(@class,'mb-2')]"
    )
    if len(body_candidates) != 1:
        return None

    body_el = body_candidates[0]
    body = body_el.text.strip()
    if not body:
        return None

    author_el = comment_block.find_element(
        By.XPATH,
        "./div[contains(@class,'mb-1')]//a[starts-with(@href,'/u/')][1]"
    )
    author = author_el.text.strip()
    author_link = urljoin(BASE_URL, author_el.get_attribute("href"))

    return {
        "temp_id": temp_id,
        "parent_temp_id": parent_temp_id,
        "depth": depth,
        "comment_order": comment_order,
        "author": author,
        "author_link": author_link,
        "body": body,
    }


def parse_comment_node(node, parent_temp_id, depth, counter, results, seen):
    """
    A node looks like:
      <div>
        <div class="py-2"> ... actual comment ... </div>
        <div class="ml-4 pl-4 ..."> ... replies ... </div>   # optional
      </div>
    """
    try:
        comment_block = node.find_element(
            By.XPATH,
            "./div[contains(@class,'py-2')]"
        )
    except Exception:
        return

    temp_id = counter["temp_id"]
    counter["temp_id"] += 1

    comment_order = counter["comment_order"]
    counter["comment_order"] += 1

    comment = extract_comment_from_py2(
        comment_block=comment_block,
        temp_id=temp_id,
        parent_temp_id=parent_temp_id,
        depth=depth,
        comment_order=comment_order
    )

    if comment is None:
        return

    key = (comment["parent_temp_id"], comment["author_link"], comment["body"])
    if key in seen:
        return
    seen.add(key)

    results.append(comment)

    # recurse into reply container(s)
    reply_containers = node.find_elements(
        By.XPATH,
        "./div[contains(@class,'ml-4') and contains(@class,'pl-4')]"
    )

    for reply_container in reply_containers:
        # each direct child node inside reply_container is another comment node
        child_nodes = reply_container.find_elements(
            By.XPATH,
            "./div[./div[contains(@class,'py-2')]]"
        )

        for child_node in child_nodes:
            parse_comment_node(
                node=child_node,
                parent_temp_id=temp_id,
                depth=depth + 1,
                counter=counter,
                results=results,
                seen=seen
            )


def scrape_comments(driver):
    comments = []

    WebDriverWait(driver, 15).until(
        EC.presence_of_element_located((By.XPATH, "//h2[contains(., 'Comments')]"))
    )

    # All actual comment blocks after the Comments heading, in DOM order
    comment_blocks = driver.find_elements(
        By.XPATH,
        "//h2[contains(., 'Comments')]/following::div[contains(@class,'py-2')]"
    )

    seen = set()
    temp_id = 1

    # latest comment temp_id seen at each depth
    latest_at_depth = {}

    for block in comment_blocks:
        try:
            body_candidates = block.find_elements(
                By.XPATH,
                "./div[contains(@class,'text-sm') and contains(@class,'mb-2')]"
            )
            if len(body_candidates) != 1:
                continue

            author_candidates = block.find_elements(
                By.XPATH,
                "./div[contains(@class,'mb-1')]//a[starts-with(@href,'/u/')][1]"
            )
            if len(author_candidates) != 1:
                continue

            body_el = body_candidates[0]
            author_el = author_candidates[0]

            body = body_el.text.strip()
            if not body:
                continue

            author = author_el.text.strip()
            author_link = urljoin(BASE_URL, author_el.get_attribute("href"))

            # Depth = number of reply-indent ancestors
            depth = len(block.find_elements(
                By.XPATH,
                "./ancestor::div[contains(@class,'ml-4') and contains(@class,'pl-4')]"
            ))

            parent_temp_id = latest_at_depth.get(depth - 1) if depth > 0 else None

            key = (depth, parent_temp_id, author_link, body)
            if key in seen:
                continue
            seen.add(key)

            comments.append({
                "temp_id": temp_id,
                "parent_temp_id": parent_temp_id,
                "depth": depth,
                "comment_order": temp_id,
                "author": author,
                "author_link": author_link,
                "body": body,
            })

            latest_at_depth[depth] = temp_id

            # clear stale deeper branches when we move back up
            for d in list(latest_at_depth.keys()):
                if d > depth:
                    del latest_at_depth[d]

            temp_id += 1

        except Exception:
            continue

    return comments


def scrape_post_and_comments(driver, post_url):
    post_data = scrape_main_post(driver, post_url)
    comments = scrape_comments(driver)

    return {
        "post_url": post_data["post_url"],
        "author": post_data["author"],
        "author_link": post_data["author_link"],
        "title": post_data["title"],
        "body": post_data["body"],
        "comments": comments,
    }


def main():
    with open("posts.txt", "r", encoding="utf-8") as f:
        post_urls = [line.strip() for line in f if line.strip()]
    
    conn = create_connection("moltbook.db")
    create_tables(conn)

    driver = webdriver.Chrome()
    for post_url in post_urls:
        try:
            data = scrape_post_and_comments(driver, post_url)

            post_id = save_post(conn, data)
            save_comments(conn, post_id, data["comments"])

            print("Saved post:", data["title"])
            print("Post author:", data["author"])
            print("Comments saved:", len(data["comments"]))

            for c in data["comments"][:20]:
                print("---")
                print("TEMP ID:", c["temp_id"])
                print("PARENT TEMP ID:", c["parent_temp_id"])
                print("DEPTH:", c["depth"])
                print("AUTHOR:", c["author"])
                print("TEXT:", c["body"][:200])
        except Exception as e:
            print("Error processing", post_url)
            print(e)

        time.sleep(2) 
    driver.quit()
    conn.close()

if __name__ == "__main__":
    main()
