from flask import Flask,request
from flask_mysqldb import MySQL
from flask_restful import Api,Resource
from flask_cors import CORS
from house import Model
import json
app = Flask(__name__)

app.config['ENV'] = 'development'
app.config['DEBUG'] = True
app.config['TESTING'] = True
# app.config['MYSQL_USER'] = 'root'
# app.config['MYSQL_PASSWORD'] = 'League123!'
# app.config['MYSQL_HOST'] = 'localhost'
# app.config['MYSQL_DB'] ='anime'
# app.config['MYSQL_CURSORCLASS'] ='DictCursor'

mysql =  MySQL(app)
api = Api(app)
CORS(app)
class House(Resource):
    def post(self):
        content = request.json
        new_house = []
        for key in content.keys():
            new_house.append(content[key])
        model = Model()
        price = model.predict([new_house])
        obj= {"price":price[0]}
        return json.dumps(str(obj))
        
api.add_resource(House,'/api/house')
if __name__ =="__main__":
    app.run(debug = True)
