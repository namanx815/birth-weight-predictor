from flask import Flask,request,jsonify,render_template
import pandas as pd
import pickle

app=Flask(__name__)

@app.route("/",methods=["GET"])
def home():
    return render_template("index.html")



def get_cleaned_data(form_data):
    gestation=float(form_data["gestation"])
    parity=form_data["parity"]
    age=float(form_data["age"])
    height=float(form_data["height"])
    weight=float(form_data["weight"])
    smoke=float(form_data["smoke"])

    cleaned_data={"gestation":[gestation],
                  "parity":[parity],
                  "age":[age],
                  "height":[height],
                  "weight":[weight],
                  "smoke":[smoke]}
    return cleaned_data


##define your endpoint
@app.route("/predict",methods=["POST"])
def get_prediction():
    baby_data=request.form

    baby_data_cleaned=get_cleaned_data(baby_data)
    baby_df=pd.DataFrame(baby_data_cleaned )
    with open("model/model.pkl","rb") as obj:
        model=pickle.load(obj)
    prediction=model.predict(baby_df)
    prediction=round(float(prediction),2)
    return render_template("index.html",prediction=prediction)



    # #get data from user
    # baby_data=request.get_json()
    # #convert the dtaa into dataframe
    # baby_df=pd.DataFrame(baby_data)

    # #load machine learning trained model
    # with open("model/model.pkl","rb") as obj:
    #     model=pickle.load(obj)
    # #make prediction on user data
    # prediction=model.predict(baby_df)

    # prediction=round(float(prediction),2)
    # #return respone in json format

    # response={"Prediction:":prediction}
    # return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True)
