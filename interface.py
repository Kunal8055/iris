
import pickle
import numpy as np
from flask import Flask,jsonify,render_template,request
from analitical import iris




app=Flask(__name__)
@app.route("/")
def Home():
    return render_template("dthome.html")


@app.route("/predicted",methods=["POST"])
def predicted():
    Q1=[float(x) for x in request.form.values()]
    print("this is Q1",Q1)
    if Q1[0]==0.0:
        return render_template("Classification.html")

    elif Q1[0]==1.0:
        return render_template("PetalLength.html")
    
    elif Q1[0]==2.0:
        return render_template("PetalWidth.html")
    elif Q1[0]==3.0:
        return render_template("SepalLength.html")
    else:
        return render_template("SepalWidth.html")


@app.route("/Classification",methods=["POST"])   
def Classification():
    Q2=[float(x) for x in request.form.values()]
    Qa2=[np.array(Q2)]
    Model=iris(Qa2)
    Model1=Model.prediction_Classification()
    return render_template('Classification.html',result1="The Flower Species is  {}".format(Model1))

@app.route("/Petal_length",methods=["POST"])   
def Petal_length():
    Q3=[float(x) for x in request.form.values()]
    Qa3=[np.array(Q3)]
    Model=iris(Qa3)
    Model1=Model.prediction_Petallength()
    return render_template('PetalLength.html',result1="Petal length of flower is  {}".format(Model1))

@app.route("/Petal_Width",methods=["POST"])   
def Petal_Width():
    Q4=[float(x) for x in request.form.values()]
    Qa4=[np.array(Q4)]
    Model=iris(Qa4)
    Model1=Model.prediction_Petalwidth()
    return render_template('PetalWidth.html',result1="Petal Width of flower is  {}".format(Model1))

@app.route("/Sepal_length",methods=["POST"])   
def Sepal_length():
    Q4=[float(x) for x in request.form.values()]
    Qa4=[np.array(Q4)]
    Model=iris(Qa4)
    Model1=Model.prediction_SepalLength()
    return render_template('SepalLength.html',result1="Sepal Length of flower is  {}".format(Model1))

@app.route("/Sepal_Width",methods=["POST"])   
def Sepal_Width():
    Q4=[float(x) for x in request.form.values()]
    Qa4=[np.array(Q4)]
    Model=iris(Qa4)
    Model1=Model.prediction_SepalWidth()
    return render_template('SepalWidth.html',result1="Sepal Width of flower is  {}".format(Model1))




if __name__=="__main__":
    app.run(debug=True)

