from flask import Flask,render_template,request
from predictor import Predictor

prediction=Predictor()
app=Flask(__name__)

@app.route("/")
def home():
    return render_template("sample.html")

@app.route("/predict",methods=["POST","GET"])
def pre():
    if request.method=="POST":
        news=request.form["nm"]
        if str(news).isspace() or news=="":
            result="Enter the news!"
        else:
            stemmed=prediction.stemming(str(news))
            stemmed=prediction.joining(stemmed)
            stemmed=prediction.vectorizer.transform([stemmed])

            result=prediction.predictor.predict(stemmed)

            if result[0]==0:
                result="Looks Fake!"
            elif result[0]==1:
                result="Looks True!"


        return render_template("sample.html",result=result,news=news)
    else:
        return render_template("sample.html")

if __name__=="__main__":
    app.run(debug=True)