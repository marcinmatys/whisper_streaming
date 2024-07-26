import asyncio
import websockets
import numpy as np
import soundfile
import logging
import os
from whisper_online import *
import argparse


def warm_up():
    logger.info("warming up...")
    warmup_file= "sample.wav"
    # warm up the ASR because the very first transcribe takes more time than the others.
    # Test results in https://github.com/ufal/whisper_streaming/pull/81
    msg = "Whisper is not warmed up. The first chunk processing may take longer."
    if warmup_file:
        if os.path.isfile(warmup_file):
            a = load_audio_chunk(warmup_file,0,1)
            asr.transcribe(a)
            logger.info("Whisper is warmed up.")
        else:
            logger.critical("The warm up file is not available. "+msg)
            sys.exit(1)
    else:
        logger.warning(msg)


def format_output_transcript(o):
    now = time.time()-start

    if o[0] is not None:
        return "%1.4f %1.0f %1.0f %s" % (now*1000, o[0]*1000,o[1]*1000,o[2])
    else:
        logger.debug("No text in this segment")
        return None


async def audio_stream(websocket, path):

    out = []
    silence_candidate = []
    silence_started = False
    last_silence_log_time = 0
    online.init()
    logger.info("OnlineASRProcessor init")

    try:

        async for message in websocket:
            audio_data = message

            #convert audio to NumPy array
            sf = soundfile.SoundFile(io.BytesIO(audio_data), channels=1,endian="LITTLE",samplerate=SAMPLING_RATE, subtype="PCM_16",format="RAW")
            audio, _ = librosa.load(sf,sr=SAMPLING_RATE,dtype=np.float32)

            out.append(audio)

            o = (None, None, None)

            rms = np.sqrt(np.mean(audio**2))
            
            if rms < SILENCE_THRESHOLD:

                silence_candidate.append(audio)
                silence_candidate_len = sum(len(x) for x in silence_candidate)

                if silence_candidate_len >= SILENCE_SIZE:
                    out = []
                    current_time = time.time()  # Get the current time
                    if current_time - last_silence_log_time >= 2:  # Check if 2 seconds have passed
                        logger.info("Silence detected")
                        last_silence_log_time = current_time  # Update the timestamp
                    
                    if not silence_started:
                        o = online.finish()
                    online.init()

                    silence_started = True
            else:
                if silence_started:
                    last_silence_log_time = 0
                    silence_started = False

                silence_candidate = []

            out_len = sum(len(x) for x in out)
            if out_len >= MIN_CHUNK_SIZE:
                out_chunk = np.concatenate(out)
                online.insert_audio_chunk(out_chunk)
                o = online.process_iter()
                out = []

            result = format_output_transcript(o)
            if result is not None:
                await websocket.send(o[2])

    except websockets.exceptions.ConnectionClosed:
        online.finish()
        logger.info("Connection closed")

parser = argparse.ArgumentParser()
parser.add_argument('--silence-size', type=float, default=2.0, help='Silence segment size in seconds')
parser.add_argument('--silence-threshold', type=float, default=0.01, help='Silence threshold')
add_shared_args(parser)
args = parser.parse_args()

asr, online = asr_factory(args)

SAMPLING_RATE = 16000
MIN_CHUNK_SIZE = args.min_chunk_size*SAMPLING_RATE
SILENCE_SIZE = args.silence_size*SAMPLING_RATE
SILENCE_THRESHOLD = args.silence_threshold


start = time.time()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
package1_log = logging.getLogger('whisper_online')
package1_log.setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)

warm_up()
logger.info("Server started")
start_server = websockets.serve(audio_stream, 'localhost', 43007)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
