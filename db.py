from flask import current_app, g
from flask_pymongo import PyMongo

def get_db():
    db = getattr(g, "_database", None)
    if db is None:
        db = g._database = PyMongo(current_app).db
       
    return db

def add_user(db, open_id):
    id_doc = {'open_id': open_id}
    return db.users.insert_one(id_doc)

def user_exist(db, open_id):
    if db.users.find_one({'open_id': open_id}):
        return True
    return False

def delete_user(db, open_id):
    response = db.users.delete_one( { "open_id": open_id } )
    return response