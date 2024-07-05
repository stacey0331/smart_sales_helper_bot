#!/usr/bin/env python3.8

import os
import logging
import requests
import threading
from api import MessageApiClient
from event import MessageReceiveEvent, MeetingStartedEvent, MeetingEndedEvent, UrlVerificationEvent, EventManager 
from flask import Flask, jsonify
from dotenv import load_dotenv, find_dotenv
import json

from script.speech_to_text import *
from google.cloud import speech_v1p1beta1 as speech
import joblib

# db
from db import *
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

# load env parameters form file named .env
load_dotenv(find_dotenv())

app = Flask(__name__)

processed_message_ids = set() # handles repeated pushes

# load from env
APP_ID = os.getenv("APP_ID")
APP_SECRET = os.getenv("APP_SECRET")
VERIFICATION_TOKEN = os.getenv("VERIFICATION_TOKEN")
ENCRYPT_KEY = os.getenv("ENCRYPT_KEY")
LARK_HOST = os.getenv("LARK_HOST")

# init service
message_api_client = MessageApiClient(APP_ID, APP_SECRET, LARK_HOST)
event_manager = EventManager()

# enrolled users
active_threads = {}
stop_events = {}

# mongodb init
MONGO_URI = os.getenv("MONGO_URI")
db_client = MongoClient(MONGO_URI, server_api=ServerApi('1'))
try:
    db_client.admin.command('ping')
    print("You successfully connected to MongoDB!")
except Exception as e:
    print(e)
db = db_client.get_database("sales-helper")

@event_manager.register("vc.meeting.all_meeting_started_v1")
def meeting_started_event_handler(req_data: MeetingStartedEvent):
    open_id = req_data.event.operator.id.open_id
    if not user_exist(db, open_id):
        return jsonify()
    
    if open_id not in active_threads:
        stop_event = threading.Event()
        stop_events[open_id] = stop_event
        thread = threading.Thread(target=start_transcription, args=(open_id,stop_event))
        active_threads[open_id] = thread
        thread.start()

    return jsonify()

def start_transcription(open_id, stop_event):
    language_code = "en-US"
    client = speech.SpeechClient()
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code=language_code,
    )

    streaming_config = speech.StreamingRecognitionConfig(
        config=config, interim_results=True
    )

    log_reg = joblib.load('./model/logistic_reg_model.pkl')
    glove_model = joblib.load('./model/glove_model.pkl')

    with MicrophoneStream() as stream:
        audio_generator = stream.generator()
        requests = (
            speech.StreamingRecognizeRequest(audio_content=content)
            for content in audio_generator
        )

        responses = client.streaming_recognize(streaming_config, requests)
        listen_print_loop(responses, open_id, log_reg, glove_model, message_api_client, stop_event)

@event_manager.register("vc.meeting.all_meeting_ended_v1")
def meeting_ended_event_handler(req_data: MeetingEndedEvent):
    open_id = req_data.event.operator.id.open_id
    if open_id in active_threads:
        stop_events[open_id].set()
        active_threads[open_id].join()
        del active_threads[open_id]
        del stop_events[open_id]
    return jsonify()

# @event_manager.register("vc.meeting.join_meeting_v1")
# def join_meeting_event_handler(req_data: JoinMeetingEvent):
#     print('join meeting \n\n')
#     return jsonify()

@event_manager.register("url_verification")
def request_url_verify_handler(req_data: UrlVerificationEvent):
    # url verification, just need return challenge
    if req_data.event.token != VERIFICATION_TOKEN:
        raise Exception("VERIFICATION_TOKEN is invalid")
    return jsonify({"challenge": req_data.event.challenge})


@event_manager.register("im.message.receive_v1")
def message_receive_event_handler(req_data: MessageReceiveEvent):
    sender_id = req_data.event.sender.sender_id
    message = req_data.event.message
    if message.message_type != "text":
        logging.warn("Other types of messages have not been processed yet")
        return jsonify()
    if message.message_id in processed_message_ids: # handles repeated pushes
        return jsonify()
    processed_message_ids.add(message.message_id)
    open_id = sender_id.open_id
    text_content = message.content
    
    valid_user = user_exist(db, open_id)
    if json.loads(text_content)['text'].lower() == 'enroll':
        if not valid_user:
            add_user(db, open_id)
        text_content = {
            "text": "Welcome! \nYou've successfully enrolled. You will receive reminders for your meetings with Lark! \n To reverse this action, type \"STOP\"."
        }
        message_api_client.send_text_with_open_id(open_id, json.dumps(text_content))
    elif json.loads(text_content)['text'].lower() == 'stop':
        delete_user(db, open_id)
        text_content = {
            "text": "You will not receive future messages again. To use this bot again, type \"ENROLL\"."
        }
        message_api_client.send_text_with_open_id(open_id, json.dumps(text_content))
    elif valid_user:
        text_content = {
            "text": "You're already enrolled in receiving reminders. To unenroll, type \"STOP\".\n I'm currently not a chatbot. "
        }
        message_api_client.send_text_with_open_id(open_id, json.dumps(text_content))
    else: 
        text_content = {
            "text": "You're currently not enrolled to receive reminders. Type \"ENROLL\" to enroll. \nI'm currently not a chatbot. "
        }
        message_api_client.send_text_with_open_id(open_id, json.dumps(text_content))

    return jsonify()


@app.errorhandler
def msg_error_handler(ex):
    logging.error(ex)
    response = jsonify(message=str(ex))
    response.status_code = (
        ex.response.status_code if isinstance(ex, requests.HTTPError) else 500
    )
    return response


@app.route("/", methods=["POST"])
def callback_event_handler():
    # init callback instance and handle
    event_handler, event = event_manager.get_handler_with_event(VERIFICATION_TOKEN, ENCRYPT_KEY)

    return event_handler(event)


if __name__ == "__main__":
    # init()
    app.run(host="0.0.0.0", port=3000, debug=True)
