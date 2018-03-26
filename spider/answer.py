"""
    use method search_answer(question) to get answer(list)
    >>> from spider.answer import search_answer
    >>> print(search_answer("hahaha"))
"""

import requests
import threading

lock = threading.Lock()
answers = []


def search_answer(question):
    global answers
    answers = []
    search_yahoo(question)
    #search_stackoverflow(question)
    return answers

def search_yahoo(question):
    search_url = "https://answers.search.yahoo.com/search"
    payload = {'p': question}
    question_pattern = "class=\"\s?lh-17\sfz-m\s?\"\shref=\"(.+?)\""
    questions = extract_question(search_url, payload, question_pattern)[:4]
    answer_pattern = "class=\"ya-q-full-text\">(.+?)</span>"
    multi_thread_crawl(questions, answer_pattern)


def search_stackoverflow(question):
    host = "https://stackoverflow.com"
    search_url = "https://stackoverflow.com/search"
    payload = {'q': question}
    question_pattern = "href=\"(/questions/\d+/.+?)\""
    questions = extract_question(search_url, payload, question_pattern)[:4]
    questions = [host+url for url in questions]
    answer_pattern = "class=\"answercell\">\s*<div.+?>\s*([\s\S]+?)</div>"
    multi_thread_crawl(questions, answer_pattern)


def extract_question(search_url, payload, question_pattern):
    return re_match(search_url, question_pattern, payload)


def multi_thread_crawl(questions, answer_pattern):
    for url in questions:
        t = threading.Thread(target=extract_answer, args=(url, answer_pattern))
        t.start()
        t.join()


def extract_answer(url, answer_pattern):
    global answers
    if len(answers) > 10:
        return
    answer = re_match(url, answer_pattern)
    lock.acquire()
    answers += answer
    lock.release()


def re_match(url, pattern, payload=None):
    import re
    response = requests.get(url, payload).text
    match_list = re.findall(pattern, response)
    return match_list


if __name__ == "__main__":
    test_answer = search_answer("How to map a continent or land")
    print(test_answer)