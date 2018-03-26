from flask import Flask, render_template, jsonify, request
import sortAnswers
import spider
import json

app = Flask(__name__)
app.config.from_object("config")
q, a = sortAnswers.load_ques_ans('question.txt', 'answer.txt')

@app.route("/", methods=["GET", "POST"])
def search():
    return render_template("search.html")

@app.route("/api/test", methods=["GET", "POST"])
def api_test():
    res = sortAnswers.make_answers(q, a)
    return jsonify(str(res))

@app.route("/api/sort_answers", methods=["GET", "POST"])
def api_sort():
    '''
    def test():
        data = {}
        data['question'] = '["HOW AFRICAN AMERICANS WERE IMMIGRATED TO THE US"]'
        data['answer'] = '["African immigration to the United States refers to immigrants to the United States who are or were nationals of Africa .", "The term African in the scope of this article refers to geographical or national origins rather than racial affiliation .", "From the Immigration and Nationality Act of 1965 to 2007 , an estimated total of 0.8 to 0.9 million Africans immigrated to the United States , accounting for roughly 3.3 % of total immigration to the United States during this period .", "African immigrants in the United States come from almost all regions in Africa and do not constitute a homogeneous group .", "They include people from different national , linguistic , ethnic , racial , cultural and social backgrounds .", "As such , African immigrants are to be distinguished from African American people , the latter of whom are descendants of mostly West and Central Africans who were involuntarily brought to the United States by means of the historic Atlantic slave trade .", "None", "None", "None", "None"]'
        r = requests.post('http://127.0.0.1:5000/api/sort_answers', data=data)
        print r.text
    '''
    if request.method == 'POST':
        print(request.form['question'])
        print(request.form['answer'])
        q = json.loads(request.form['question'])
        a = json.loads(request.form['answer'])
        res = sortAnswers.make_answers(q, a)
        return jsonify(str(res))
    return render_template('search.html')

@app.route("/api/get_answers", methods=["GET", "POST"])
def api_get_answer():
    '''
    def test():
        data = {}
        data['question'] = '["HOW AFRICAN AMERICANS WERE IMMIGRATED TO THE US"]'
        r = requests.post('http://127.0.0.1:5000/api/get_answers', data=data)
        print r.text
    '''
    if request.method == 'POST':
        print(request.form['question'])
        q = request.form['question']
        a = spider.search_answer(q)
        while len(a) < 10:
            a.append('None')
        q = [q]
        print(a)
        res = sortAnswers.make_answers(q, a)
        return jsonify(res)
    return render_template('search.html')

if __name__ == "__main__":
    app.run(debug=True)
