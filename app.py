
from __future__ import division, print_function

from flask import Flask,render_template,url_for,request

from flask_bootstrap import Bootstrap

import os

import glob

import re
import subprocess

import pandas as pd

import pickle

from sklearn.preprocessing import LabelEncoder

import numpy as np

from werkzeug.utils import secure_filename

from subprocess import Popen, PIPE

from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter

from pdfminer.converter import TextConverter

from pdfminer.layout import LAParams

from pdfminer.pdfpage import PDFPage

import io

from io import StringIO

from sklearn.feature_extraction.text import TfidfVectorizer

import os

import glob
from flask_mysqldb import MySQL
import yaml

import nltk 
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

import string

from nltk.corpus import stopwords

import matplotlib.pyplot as plt

from nltk.stem import WordNetLemmatizer

import warnings


app = Flask(__name__)

Bootstrap(app)

db = yaml.load(open('db.yaml'))
app.config['MYSQL_HOST'] = db['mysql_host']
app.config['MYSQL_USER'] = db['mysql_user']
app.config['MYSQL_PASSWORD'] = db['mysql_password']
app.config['MYSQL_DB'] = db['mysql_db']
mysql = MySQL(app)




@app.route('/')

def index():



    return render_template('predict.html')





@app.route('/predict', methods=['POST'])

def predict():





    df = pd.read_csv('legal_case_documents1.csv')

    df = df[pd.notnull(df['Text_Data'])]

    col = ['Category', 'Text_Data']

    df = df[col]

    df.columns

    df.columns = ['Category', 'Text_Data']



    df['category_id'] = df['Category'].factorize()[0]

    from io import StringIO

    category_id_df = df[['Category', 'category_id']].drop_duplicates().sort_values('category_id')

    category_to_id = dict(category_id_df.values)

    id_to_category = dict(category_id_df[['category_id', 'Category']].values)

    print(df.head())

    from sklearn.feature_extraction.text import TfidfVectorizer



    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')



    features = tfidf.fit_transform(df.Text_Data).toarray()

    labels = df.category_id

    print(features.shape)



    from sklearn.model_selection import train_test_split

    from sklearn.feature_extraction.text import CountVectorizer

    from sklearn.feature_extraction.text import TfidfTransformer



    X_train, X_test, y_train, y_test = train_test_split(df['Text_Data'], df['Category'], random_state = 0)

    count_vect = CountVectorizer()

    X_train_counts = count_vect.fit_transform(X_train)

    tfidf_transformer = TfidfTransformer()







    if request.method == 'POST':

        file = request.files['image']

        file1 = request.files['image1']

        file2 = request.files['image2']

        file3 = request.files['image3']

        file4 = request.files['image4']

        



        basepath = os.path.dirname(__file__)


        filename = secure_filename(file.filename)
        file_path = os.path.join(


            basepath, 'static/uploads', filename)

        file.save(file_path)

        filename1 = secure_filename(file1.filename)

        file_path1 = os.path.join(

           basepath, 'static/uploads', filename1)

        file1.save(file_path1)

        filename2 = secure_filename(file2.filename)

        file_path2 = os.path.join(



            basepath, 'static/uploads', filename2)

        file2.save(file_path2)


        filename3 = secure_filename(file3.filename)
        file_path3 = os.path.join(

           basepath, 'static/uploads', filename3)

        file3.save(file_path3)


        filename4 = secure_filename(file4.filename)
        file_path4 = os.path.join(



            basepath, 'static/uploads', filename4)

        file4.save(file_path4)



        def convert2txt():

	        alltexts = []

	        with open(file_path, 'rb') as fh:

		        rsrcmgr = PDFResourceManager()

		        retstr = StringIO()

		        codec = 'utf-8'

		        laparams = LAParams()

		        device = TextConverter(rsrcmgr, retstr, laparams=laparams)

		        fp = open(file_path, 'rb')

		        interpreter = PDFPageInterpreter(rsrcmgr, device)

		        password = ""

		        maxpages = 0

		        caching = True

		        pagenos=set()



		        for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password,caching=caching, check_extractable=True):

		            interpreter.process_page(page)



		        text = retstr.getvalue()

		        alltexts.append(text)

		        fp.close()

		        device.close()

		        retstr.close()



	        return alltexts



        def convert2txt1():



   	        alltexts = []

   	        with open(file_path1, 'rb') as fh:

   		        rsrcmgr = PDFResourceManager()

   		        retstr = StringIO()

   		        codec = 'utf-8'

   		        laparams = LAParams()

   		        device = TextConverter(rsrcmgr, retstr, laparams=laparams)

   		        fp = open(file_path1, 'rb')

   		        interpreter = PDFPageInterpreter(rsrcmgr, device)

   		        password = ""

   		        maxpages = 0

   		        caching = True

   		        pagenos=set()



   		        for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password,caching=caching, check_extractable=True):

   		            interpreter.process_page(page)



   		        text = retstr.getvalue()

   		        alltexts.append(text)

   		        fp.close()

   		        device.close()

   		        retstr.close()



   	        return alltexts



        def convert2txt2():

	        alltexts = []

	        with open(file_path2, 'rb') as fh:

		        rsrcmgr = PDFResourceManager()

		        retstr = StringIO()

		        codec = 'utf-8'

		        laparams = LAParams()

		        device = TextConverter(rsrcmgr, retstr, laparams=laparams)

		        fp = open(file_path2, 'rb')

		        interpreter = PDFPageInterpreter(rsrcmgr, device)

		        password = ""

		        maxpages = 0

		        caching = True

		        pagenos=set()



		        for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password,caching=caching, check_extractable=True):

		            interpreter.process_page(page)



		        text = retstr.getvalue()

		        alltexts.append(text)

		        fp.close()

		        device.close()

		        retstr.close()



	        return alltexts



        def convert2txt3():

	        alltexts = []

	        with open(file_path3, 'rb') as fh:

		        rsrcmgr = PDFResourceManager()

		        retstr = StringIO()

		        codec = 'utf-8'

		        laparams = LAParams()

		        device = TextConverter(rsrcmgr, retstr, laparams=laparams)

		        fp = open(file_path3, 'rb')

		        interpreter = PDFPageInterpreter(rsrcmgr, device)

		        password = ""

		        maxpages = 0

		        caching = True

		        pagenos=set()



		        for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password,caching=caching, check_extractable=True):

		            interpreter.process_page(page)



		        text = retstr.getvalue()

		        alltexts.append(text)

		        fp.close()

		        device.close()

		        retstr.close()



	        return alltexts



        def convert2txt4():

	        alltexts = []

	        with open(file_path4, 'rb') as fh:

		        rsrcmgr = PDFResourceManager()

		        retstr = StringIO()

		        codec = 'utf-8'

		        laparams = LAParams()

		        device = TextConverter(rsrcmgr, retstr, laparams=laparams)

		        fp = open(file_path4, 'rb')

		        interpreter = PDFPageInterpreter(rsrcmgr, device)

		        password = ""

		        maxpages = 0

		        caching = True

		        pagenos=set()



		        for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password,caching=caching, check_extractable=True):

		            interpreter.process_page(page)



		        text = retstr.getvalue()

		        alltexts.append(text)

		        fp.close()

		        device.close()

		        retstr.close()



	        return alltexts






        textdata = convert2txt()

        textdata1 = convert2txt1()

        textdata2 = convert2txt2()

        textdata3 = convert2txt3()

        textdata4 = convert2txt4()

      

        # Feature engineering to get the data in right format

        dfdemo = pd.DataFrame(textdata, columns = ['Data'])

        dfdemo['Data'] = dfdemo['Data'].apply(lambda x: " ".join(x.lower() for x in x.split())) # lower case conversiondfdemo['Data'] = dfdemo['Data'].str.replace('[^\w\s]','') # getting rid of special characters

        dfdemo['Data'] = dfdemo['Data'].str.replace('\d+', '') # removing numeric values from between the words

        dfdemo['Data'] = dfdemo['Data'].apply(lambda x: x.translate(string.digits)) # removing numerical numbers

        stop = stopwords.words('english')

        dfdemo['Data'] = dfdemo['Data'].apply(lambda x: " ".join(x for x in x.split() if x not in stop)) #removing stop words

        stemmer = WordNetLemmatizer()

        dfdemo['Data'] = [stemmer.lemmatize(word) for word in dfdemo['Data']]



        dfdemo1 = pd.DataFrame(textdata1, columns = ['Data'])

        dfdemo1['Data'] = dfdemo1['Data'].apply(lambda x: " ".join(x.lower() for x in x.split())) # lower case conversiondfdemo['Data'] = dfdemo['Data'].str.replace('[^\w\s]','') # getting rid of special characters

        dfdemo1['Data'] = dfdemo1['Data'].str.replace('\d+', '') # removing numeric values from between the words

        dfdemo1['Data'] = dfdemo1['Data'].apply(lambda x: x.translate(string.digits)) # removing numerical numbers

        stop = stopwords.words('english')

        dfdemo1['Data'] = dfdemo1['Data'].apply(lambda x: " ".join(x for x in x.split() if x not in stop)) #removing stop words

        stemmer = WordNetLemmatizer()

        dfdemo1['Data'] = [stemmer.lemmatize(word) for word in dfdemo1['Data']]



        dfdemo2 = pd.DataFrame(textdata2, columns = ['Data'])

        dfdemo2['Data'] = dfdemo2['Data'].apply(lambda x: " ".join(x.lower() for x in x.split())) # lower case conversiondfdemo['Data'] = dfdemo['Data'].str.replace('[^\w\s]','') # getting rid of special characters

        dfdemo2['Data'] = dfdemo2['Data'].str.replace('\d+', '') # removing numeric values from between the words

        dfdemo2['Data'] = dfdemo2['Data'].apply(lambda x: x.translate(string.digits)) # removing numerical numbers

        stop = stopwords.words('english')

        dfdemo2['Data'] = dfdemo2['Data'].apply(lambda x: " ".join(x for x in x.split() if x not in stop)) #removing stop words

        stemmer = WordNetLemmatizer()

        dfdemo2['Data'] = [stemmer.lemmatize(word) for word in dfdemo2['Data']]



        dfdemo3 = pd.DataFrame(textdata3, columns = ['Data'])

        dfdemo3['Data'] = dfdemo3['Data'].apply(lambda x: " ".join(x.lower() for x in x.split())) # lower case conversiondfdemo['Data'] = dfdemo['Data'].str.replace('[^\w\s]','') # getting rid of special characters

        dfdemo3['Data'] = dfdemo3['Data'].str.replace('\d+', '') # removing numeric values from between the words

        dfdemo3['Data'] = dfdemo3['Data'].apply(lambda x: x.translate(string.digits)) # removing numerical numbers

        stop = stopwords.words('english')

        dfdemo3['Data'] = dfdemo3['Data'].apply(lambda x: " ".join(x for x in x.split() if x not in stop)) #removing stop words

        stemmer = WordNetLemmatizer()

        dfdemo3['Data'] = [stemmer.lemmatize(word) for word in dfdemo3['Data']]



        dfdemo4 = pd.DataFrame(textdata4, columns = ['Data'])

        dfdemo4['Data'] = dfdemo4['Data'].apply(lambda x: " ".join(x.lower() for x in x.split())) # lower case conversiondfdemo['Data'] = dfdemo['Data'].str.replace('[^\w\s]','') # getting rid of special characters

        dfdemo4['Data'] = dfdemo4['Data'].str.replace('\d+', '') # removing numeric values from between the words

        dfdemo4['Data'] = dfdemo4['Data'].apply(lambda x: x.translate(string.digits)) # removing numerical numbers

        stop = stopwords.words('english')

        dfdemo4['Data'] = dfdemo4['Data'].apply(lambda x: " ".join(x for x in x.split() if x not in stop)) #removing stop words

        stemmer = WordNetLemmatizer()

        dfdemo4['Data'] = [stemmer.lemmatize(word) for word in dfdemo4['Data']]



        print(dfdemo4['Data'])


        inputs =  count_vect.transform(dfdemo['Data'])

        inputs1 = count_vect.transform(dfdemo1['Data'])

        inputs2 = count_vect.transform(dfdemo2['Data'])

        inputs3 = count_vect.transform(dfdemo3['Data'])

        inputs4 =  count_vect.transform(dfdemo4['Data'])

        


        savedmodel = pickle.load(open('legal_case.pkl','rb'))

        pred = savedmodel.predict(inputs)

        pred1 = savedmodel.predict(inputs1)



        pred2 = savedmodel.predict(inputs2)



        pred3 = savedmodel.predict(inputs3)



        pred4 = savedmodel.predict(inputs4)
        
        #sub = subprocess.Popen([path], shell=True)



    return render_template('predict.html', pred = pred, pred1=pred1, filename=filename, filename1=filename1,

                           filename2=filename2, filename3=filename3, filename4=filename4,

                           pred2=pred2, pred3=pred3, pred4=pred4, file_path=file_path)

@app.route('/views', methods=['POST'])

def views():
    
    if request.method == 'POST':

        file = request.form["name"]

        
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(


            basepath, 'static/uploads', file)

        
        subprocess.Popen([file_path], shell=True)

    return  render_template('predict.html')

@app.route('/view', methods=['POST'])

def view():

    if request.method == 'POST':

        file1 = request.form["name1"]
        
      
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(


            basepath, 'static/uploads', file1)

        
        subprocess.Popen([file_path], shell=True)

    return  render_template('predict.html')

@app.route('/view1', methods=['POST'])

def view1():
    if request.method == 'POST':

        file2 = request.form["name2"]
        
   
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(


            basepath, 'static/uploads', file2)

        
        subprocess.Popen([file_path], shell=True)

    return  render_template('predict.html')

@app.route('/view2', methods=['POST'])

def view2():
    if request.method == 'POST':

        file3 = request.form["name3"]

    
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(


            basepath, 'static/uploads', file3)

        
        subprocess.Popen([file_path], shell=True)

    return  render_template('predict.html')

@app.route('/view3', methods=['POST'])

def view3():
    if request.method == 'POST':

        file4 = request.form["name4"]
        filename = secure_filename(file4.filename)
        

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(


            basepath, 'static/uploads', file4)
        
        
        subprocess.Popen([file_path], shell=True)

    return  render_template('predict.html', file_path=file_path)



@app.route('/add_document')
def add_document():
    
    
    cur1 = mysql.connection.cursor()
        
    result1 = cur1.execute("SELECT * from cases")
    if(result1>0):
        
    
        view = cur1.fetchall()
    
    if request.method == 'POST':
        
        file = request.files['file']
            
        basepath = os.path.dirname(__file__)
        filename = secure_filename(file.filename)
        
        file_path = os.path.join(
            
            basepath, '', filename)

        file.save(file_path)
        

        
        cur = mysql.connection.cursor()
        query = "INSERT INTO cases (court_cases) VALUES (%s)"
        cur.execute(query, (filename, ))
        mysql.connection.commit()
        cur.close()
        marked = 'sucessful'
        
        
       
            
            
            

        return render_template('add.html', marked=marked, view=view)
    
    return render_template('add.html')




@app.route('/document', methods=["POST"])

def document():
    
    
    if request.method == 'POST':
        
        file = request.files['tochi']
            
        basepath = os.path.dirname(__file__)
        filename = secure_filename(file.filename)
        
        file_path = os.path.join(
            
            basepath, '', filename)

        file.save(file_path)
        

        
        cur = mysql.connection.cursor()
        query = "INSERT INTO cases (court_cases) VALUES (%s)"
        cur.execute(query, (filename, ))
        mysql.connection.commit()
        cur.close()
        marked = 'sucessful'
        
        
        cur1 = mysql.connection.cursor()
        
        result1 = cur1.execute("SELECT * from cases")
        if(result1>0):
        
            view = cur1.fetchall()
            
            
            

        return render_template('add.html', marked=marked, view=view)
        
    return render_template('add.html')

@app.route('/vi', methods=['POST'])
def vi():

    if request.method == 'POST':
        cur1 = mysql.connection.cursor()
        
        result1 = cur1.execute("SELECT * from cases")
        if(result1>0):
        
            view = cur1.fetchall()
            

        doc = request.form['tochi']
        print(doc)
        
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(


            basepath, 'static/uploads', doc)
        
        
        subprocess.Popen([file_path], shell=True)

    return render_template('add.html',view=view)







































if __name__=='__main__':



	app.run(debug=True)

