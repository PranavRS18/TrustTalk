from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
import speech_recognition as sr
import asyncio
import queue
import threading
import json
import logging
import pickle
import numpy as np
from typing import Set
import modeluse

import pyaudio
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Mount the static folder
app.mount("/static", StaticFiles(directory="static"), name="static")


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


active_websockets: Set[WebSocket] = set()

class TranscriberWithAI:
    def __init__(self, model_path='model.pkl'):
        self.recognizer = sr.Recognizer()
        self.is_running = False
        self.audio_queue = queue.Queue()
        import modeluse
        # Load AI model
        self.model = modeluse.load_model_with_custom_layer("mymodel.h5")
        '''try:
            #with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            logger.info("AI model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model = None'''
        
        
        # Configure recognizer
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8
    
    def predict_text(self, text: str):
        """Make prediction using the loaded model"""
        
        try:
            if self.model is None:
                return {"error": "Model not loaded"}
            
            # Make prediction (modify this according to your model's requirements)
            text_array = np.array([text])  # Convert to correct format
            print(self.model.predict([text_array]))
            print(text_array)
            prediction = self.model.predict([text_array])[0]
             
            
            #probability = self.model.predict_proba([text_array])[0].max()
            
            return {
                "class": str(prediction),
                "scam": float(prediction)
            }
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {"error": str(e)}

    async def broadcast_transcription(self, text: str):
        """Send transcription and prediction to all clients"""
        disconnected = set()
        
        # Get prediction
        prediction = self.predict_text(text)
        
        for websocket in active_websockets:
            try:
                await websocket.send_json({
                    "text": text,
                    "prediction": prediction
                })
            except:
                disconnected.add(websocket)
        
        active_websockets.difference_update(disconnected)

    def process_audio(self):
        """Process audio chunks from queue"""
        while self.is_running:
            try:
                audio = self.audio_queue.get(timeout=1)
                try:
                    text = self.recognizer.recognize_google(audio)
                    asyncio.run(self.broadcast_transcription(text.lower()))
                except sr.UnknownValueError:
                    logger.info("Could not understand audio")
                except sr.RequestError as e:
                    logger.error(f"Speech recognition error: {e}")
            except queue.Empty:
                continue

    def record_audio(self):
        """Record audio continuously"""
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            logger.info("Recording started")
            
            while self.is_running:
                try:
                    audio = self.recognizer.listen(source, phrase_time_limit=5)
                    self.audio_queue.put(audio)
                except Exception as e:
                    logger.error(f"Recording error: {e}")
                    continue

    def start(self):
        """Start transcription and processing"""
        if not self.is_running:
            self.is_running = True
            threading.Thread(target=self.record_audio, daemon=True).start()
            threading.Thread(target=self.process_audio, daemon=True).start()

    def stop(self):
        """Stop transcription"""
        self.is_running = False

# Create single instance
transcriber = TranscriberWithAI()

@app.get("/")
async def home():
    return FileResponse("index_2.html")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_websockets.add(websocket)
    
    try:
        if not transcriber.is_running:
            transcriber.start()
        
        while True:
            data = await websocket.receive_text()
            if data == "stop":
                break
    except:
        logger.info("Client disconnected")
    finally:
        active_websockets.remove(websocket)
        if not active_websockets:
            transcriber.stop()

@app.on_event("shutdown")
async def shutdown_event():
    transcriber.stop()
