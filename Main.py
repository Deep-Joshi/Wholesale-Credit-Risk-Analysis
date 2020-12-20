from flask import Flask, render_template, request
import numpy as np
import pandas as pd

import pickle
app = Flask(__name__,static_url_path='/static')

@app.route('/')
def predict():
   return render_template('Home.html')

@app.route('/prediction',methods = ['POST', 'GET'])
def result():
	if request.method == 'POST':
	
		a=0
		b=0
		c=0
		yes=0
		no=0
		ci=0
		cre_permanent=0
		cre_construction=0
		t982=0
		x456=0
		r567=0
		y237=0
		z009 = 0
		
		x1 = request.form.get('X1')
		x3 = request.form.get('X3')
		x4 = request.form.get('X4')
		x5 = request.form.get('X5')
		x6 = request.form.get('X6')
		x7 = request.form.get('X7')
		x8 = request.form.get('X8')
		x9 = request.form.get('X9')
		x10 = request.form.get('X10')
		x11 = request.form.get('X11')
		x12 = request.form.get('X12')
		x13 = request.form.get('X13')
		x14 = request.form.get('X14')
		x15 = request.form.get('X15')

		if x1 == "C&I":
			ci = 1
		elif x1 == "CRE Permanent":
			cre_permanent = 1
		else:
			cre_construction = 1
		
		
		if x13 == 'A':
			a = 1
		elif x13 == 'B':
			b = 1
		else:
			c = 1
		
		if x14 == 'T982':
			t982 = 1
		elif x14 == 'X456':
			x456 = 1
		elif x14 == 'R567':
			r567 = 1
		elif x14 == 'Y237':
			y237 = 1
		else:
			z009 = 1
		
		if x15 == 'Yes':
			yes = 1
		else:
			no = 1
		
		
		
		testing = pd.DataFrame([x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,cre_construction,cre_permanent,b,c,t982,x456,y237,z009,yes])
		
		with open('stdScaler','rb') as s:
			scalar = pickle.load(s)
		
		testing = testing.values.reshape(1,19)
		testing = scalar.transform(testing)
		
		with open('model_pickle_optimized_XGBoost','rb') as f:
			mp = pickle.load(f)
			res = mp.predict(testing)
		
		if res == 0:
			outcome = "No Default"
		else:
			outcome = "Default!!!"
		
		return render_template("Home.html",	prediction_text = outcome)

if __name__ == '__main__':
   app.run(debug = True)
