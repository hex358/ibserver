import sys
import json
import os
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from random import randint
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import base64
from itertools import cycle
import hashlib
import secrets
from slowapi.errors import RateLimitExceeded
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

DEPLOY_VERSION: int = 3

config = json.load(open("config.json", "r"))

_PBKDF2_ITERS = 100_000

limiter = Limiter(key_func=get_remote_address)

def hash_password(password: str) -> str:
    try:
        salt = os.urandom(16)
        dk = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            _PBKDF2_ITERS
        )
        return (
            base64.urlsafe_b64encode(salt).decode("ascii")
            + "$"
            + base64.urlsafe_b64encode(dk).decode("ascii")
        )
    except:
        raise BaseException("Invalid Input Data")

def verify_password(password: str, stored: str) -> bool:
    try:
        salt_b64, hash_b64 = stored.split("$", 1)
        salt = base64.urlsafe_b64decode(salt_b64.encode("ascii"))
        expected_dk = base64.urlsafe_b64decode(hash_b64.encode("ascii"))
    except Exception:
        return False

    new_dk = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        _PBKDF2_ITERS
    )
    return secrets.compare_digest(new_dk, expected_dk)

def _key_stream(key: str, length: int) -> bytes:
    digest = hashlib.sha256(key.encode()).digest()
    return bytes(next(b for b in cycle(digest)) for _ in range(length))

def encrypt(text: str, key: str) -> str:
    data = text.encode()
    ks = _key_stream(key, len(data))
    cipher_bytes = bytes(d ^ k for d, k in zip(data, ks))
    return base64.urlsafe_b64encode(cipher_bytes).decode()

def decrypt(cipher_b64: str, key: str) -> str:
    cipher = base64.urlsafe_b64decode(cipher_b64.encode())
    ks = _key_stream(key, len(cipher))
    plain_bytes = bytes(c ^ k for c, k in zip(cipher, ks))
    return plain_bytes.decode()

def lifespan(app: FastAPI):
    ready()
    yield
    exit()

app = FastAPI(lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

coll = None
skins_coll = None
def ready():
    global coll
    app.client = MongoClient(config["URL"], tlscertificatekeyfile=None, ssl=True, tls=True, server_api=ServerApi('1'))
    app.db = app.client["infblock"]
    app.users_collection = app.db["users"]
    app.skins_collection = app.db["skins"]
    app.users_collection.create_index("username", unique=True)
    app.skins_collection.create_index("skin", unique=True)
    coll = app.users_collection
    skins_coll = app.skins_collection

def exit():
    app.client.close()

class Player(BaseModel):
    username: str
    password: str

class SentData(BaseModel):
    username: str
    password: str
    data: str

variant = lambda x: x

default_data = {"password_hash": "",
"player_unique_id": "1",
"skin": "default",
"avatar": "default",
"DEPLOY_VERSION": DEPLOY_VERSION}

field_types: dict = {"player_unique_id": str, "DEPLOY_VERSION": int}
convert_fields: dict = {
    "player_unique_id":
        {
            1: {3: decrypt, -1: decrypt}
        }
}
args_req = {decrypt: ["decrypt_cipher_b64", "decrypt_key"]}

def convert(data: dict, username: str, password_key: str):
    for to_convert in convert_fields:

        if not to_convert in data: continue
        #print(to_convert)
        conversion_result = data[to_convert]
        try:
            convert_into = DEPLOY_VERSION if DEPLOY_VERSION in convert_fields[to_convert][data["DEPLOY_VERSION"]] else -1
            conversion_call = convert_fields[to_convert][data["DEPLOY_VERSION"]][convert_into]
            args = []
            for arg_name in args_req[conversion_call]:
                match arg_name:
                    case "decrypt_cipher_b64": args.append(data["player_unique_id"])
                    case "decrypt_key": args.append(password_key)
            conversion_result = conversion_call(*args)
        except:
            pass
        data[to_convert] = conversion_result
    data["DEPLOY_VERSION"] = DEPLOY_VERSION
    _set_element(username, data)

def _get_element(name: str, default = {}, password_key: str = "", in_coll = None, key: str = "username", auto_convert: bool = True):
    try:
        if in_coll == None: in_coll = coll
        data = in_coll.find_one({key: name})["data"]
        if auto_convert and in_coll == app.users_collection and data:

            for key in default_data:
                if not key in data:
                    data[key] = default_data[key]

            if DEPLOY_VERSION > data["DEPLOY_VERSION"]:
                convert(data, username, password_key)

        return data
    except:
        return default

def _has_element(name: str, in_coll = None, key: str = "username"):
    if in_coll == None: in_coll = coll
    return in_coll.find_one({key: name}, {"_id": 1}) != None

def _set_element(name: str, data: dict, in_coll = None, key: str = "username"):
    if in_coll == None: in_coll = coll
    in_coll.update_one(
        {key: name},
        [{
            "$set": {
                "data": {
                    "$mergeObjects": [ { "$ifNull": ["$data", {}] }, data]
                }
            }
        }],
        upsert=True
    )

def get_unique_id() -> int:
    return 1000000 + randint(0, 99999)

def hashed(string: str) -> str:
    # deprecated!
    return hashlib.sha256(string.encode()).hexdigest()

def is_agent_acceptable(request: Request, data) -> bool:
    user_agent = request.headers.get("user-agent", "").lower()
    return "godot" in user_agent

class GetSkinData(BaseModel):
    username: str


default_skins = set(["default", "zombieskin"])
@app.post("/getskin")
@limiter.limit("5/minute")
async def get_skin(data: GetSkinData, request: Request):
    #print(data.username)
    if not is_agent_acceptable(request, data):
        raise HTTPException(status_code=400, detail="CLIENT_INVALID")
    if not _has_element(data.username):
        raise HTTPException(status_code=400, detail="USERNAME_WRONG")

    skin_uid = _get_element(data.username, {}, "", auto_convert=False)["skin"]
    if skin_uid in default_skins: return {"detail": "OK", "skin_name": skin_uid, "skin_data": ""}

    element = _get_element(skin_uid, {}, "", app.skins_collection, "skin")
    decompressed = conv.perform_decompress(bson.BSON.decode(element["skin_data"])["data"], (element["width"], element["height"]))["data"]

    godot_bytes = bytearray()
    godot_bytes.append(element["width"]); godot_bytes.append(element["height"])
    count_stack = 1
    prev_color: tuple[int] = (0, 0, 0, 0)
    for i in range(len(decompressed)):
        current_color = decompressed[i]
        if prev_color == current_color and count_stack < 254:
            count_stack += 1
        else:
            godot_bytes += bytearray([prev_color[0], prev_color[1], prev_color[2], prev_color[3], count_stack])
            count_stack = 1
        prev_color = current_color
    length = len(godot_bytes)
    godot_bytes = bytearray(gzip.compress(godot_bytes))
    godot_bytes.extend(struct.pack("H", length))

    return {"detail": "OK", "skin_name": skin_uid, "skin_data": base64.b64encode(godot_bytes)}

read_only_properties = set(["player_unique_id", "DEPLOY_VERSION", "password_hash"])
from PIL import Image


class SetSkinData(BaseModel):
    username: str
    password: str
    data: str
    skin_name: str

import struct
import gzip
from PIL import Image
import util_converter as conv
import bson
import traceback

@app.post("/setskin")
@limiter.limit("5/minute")
async def set_skin(data: SetSkinData, request: Request):
    if not is_agent_acceptable(request, {"username": data.username}):
        raise HTTPException(400, "CLIENT_INVALID")

    if not try_login(data.username, data.password):
        raise HTTPException(401, "CANT_LOGIN")

    if data.skin_name in default_skins:
        _set_element(data.username, {"skin": data.skin_name})
        return {"detail": "OK"}

    try:
        compressed = bytearray(base64.b64decode(data.data))
        bytes = bytearray(gzip.decompress(compressed[:-2]))
        dims = (bytes.pop(0), bytes.pop(0))
        counter = -2

        image_data = [(0,0,0,0) for _ in range(dims[0] * dims[1])]

        for i in range(int(len(bytes)/5)):
            i = i*5
            color = (bytes[i], bytes[i+1], bytes[i+2], bytes[i+3])
            for j in range(bytes[i+4]):
                counter += 1
                image_data[counter] = color
        rectangled = conv.perform_compress(image_data, dims)
        bson_encoded = bson.BSON.encode({"data": rectangled})
        uid = str(get_unique_id())
        _set_element(uid, {"skin_data": bson_encoded, "width": dims[0], "height": dims[1]}, app.skins_collection,"skin")
        _set_element(data.username, {"skin": uid})
    except Exception as e:
        traceback.print_exc()
        print(e)
        raise HTTPException(400, "INVALID_DATA")


    return {"detail": "OK"}



@app.post("/update")
@limiter.limit("5/minute")
async def update(data: SentData, request: Request):
    if not is_agent_acceptable(request, data):
        raise HTTPException(status_code=400, detail="CLIENT_INVALID")
    if not try_login(data.username, data.password):
        raise HTTPException(status_code=401, detail="CANT_LOGIN")

    data.data = json.loads(data.data)

    for key in data.data:
        if not key in default_data or key in read_only_properties:
            raise HTTPException(status_code=401, detail="PROPERTY_CHANGE_PROHIBITED")

    for i in data.data:
        data.data[i] = field_types.get(i, variant)(data.data[i])
    _set_element(data.username, data.data)
    return {"detail": "OK"}

@app.post("/register")
@limiter.limit("5/minute")
async def register(player: Player, request: Request):
    if not is_agent_acceptable(request, player):
        raise HTTPException(status_code=400, detail="CLIENT_INVALID")

    if _has_element(player.username):
        raise HTTPException(status_code=401, detail="ERR_USERNAME_OCCUPIED")

    player_unique_id = str(get_unique_id())
    reg_data = default_data.copy()
    hashed = None
    try:
        hashed = hash_password(player.password)
    except:
        return {"INVALID_DATA"}
    reg_data.update({"password_hash": hashed,
                  "player_unique_id": player_unique_id,
                  "DEPLOY_VERSION":DEPLOY_VERSION})

    _set_element( player.username, reg_data)

    return {"detail": "OK", "player_unique_id": player_unique_id}

def try_login(username: str, password: str) -> bool:
    element = _get_element(username, None, password)
    if not element: return False
    return verify_password(password, element["password_hash"])

@app.post("/login")
@limiter.limit("5/minute")
async def login(player: Player, request: Request):
    if not is_agent_acceptable(request, player):
        raise HTTPException(status_code=400, detail="CLIENT_INVALID")

    element = _get_element(player.username, None, player.password)

    if not element:
        raise HTTPException(status_code=401, detail="USERNAME_WRONG")

    if not verify_password(player.password, element["password_hash"]):
        raise HTTPException(status_code=401, detail="PASS_WRONG")

    return {"detail": "OK", "player_unique_id": str(element["player_unique_id"])}

#uvicorn ibserver:app --port 8100 --host '::' --proxy-headers --forwarded-allow-ips "::1"