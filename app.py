# ---------------------------------------
# Raqib (رقيب) - CCTV analytics: who came in, when, and whether they are allowed
# ---------------------------------------
# Routes:
#   GET  /                  the console (recording or live RTSP camera)
#   GET  /health            what is installed
#   POST /cctv/start        start analysing (file upload or 'url')
#   GET  /cctv/stream/<id>  live annotated picture (MJPEG)
#   GET  /cctv/status/<id>  progress, people, vehicles, money, working time
#   POST /cctv/stop/<id>    stop
#   GET  /cctv/video/<id>   download the analysed video
#   GET  /cctv/log/<id>.csv, /cctv/report/<id>.csv   Excel logs
#   POST /cctv/ack/<id>/<track>   operator acknowledged an alarm
#   POST /cctv/add/<id>/<track>   name an unknown person from the video
#   GET/POST /cctv/lists    people lists (green / red / black)
#   GET/POST /cctv/plates   car lists by plate
#   POST /people/add        add a person from photos (name + list)
#   POST /people/forget     remove a person
#
# Started from the face recognition of the Ru'ya smart-glasses project.
# Listens on this computer only (127.0.0.1) until a login is added: anyone on the
# same network could otherwise open the cameras and the black list.
# ---------------------------------------
import os

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

import cctv
import face_function as faces
import lpr
from core.models import path

app = Flask(__name__, template_folder=path("templates"), static_folder=path("static"))
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024   # recordings can be large (2 GB max)
app.config["TEMPLATES_AUTO_RELOAD"] = True
MAX_IMAGE_BYTES = 16 * 1024 * 1024
UPLOAD_FOLDER = path("uploads")
HOST = os.environ.get("RAQIB_HOST", "127.0.0.1")
PORT = int(os.environ.get("RAQIB_PORT", "5002"))


def read_image(file_storage):
    data = np.frombuffer(file_storage.read(MAX_IMAGE_BYTES + 1), dtype=np.uint8)
    if data.size == 0 or data.size > MAX_IMAGE_BYTES:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


@app.route("/health")
def health():
    return jsonify({"ok": True, "plates": lpr.available()})


# ---------------------------------------
# People from photos
# ---------------------------------------
@app.route("/people/add", methods=["POST"])
def people_add():
    """Add someone from 1-5 clear photos of their face, and put them on a list."""
    images = [img for img in (read_image(f) for f in request.files.getlist("photos")) if img is not None]
    name = (request.form.get("name") or "").strip()
    group = request.form.get("list", "green")
    if not name:
        return jsonify({"error": "Type the person's name."}), 400
    if not images:
        return jsonify({"error": "Choose at least one photo of their face."}), 400
    if group not in cctv.LIST_NAMES:
        return jsonify({"error": "Choose green, red or black."}), 400
    text, info = faces.enroll(images[:5], name, "en")
    if not info.get("saved"):
        return jsonify({"error": text}), 400
    cctv.set_list(faces.clean_name(name), group)
    return jsonify({"ok": True, "message": text})


@app.route("/people/forget", methods=["POST"])
def people_forget():
    name = (request.form.get("name") or "").strip()
    message = faces.forget(name, "en")
    lists = cctv.load_lists()
    if lists.pop(faces.clean_name(name), None) is not None:
        import json
        with open(cctv.lists_path(), "w", encoding="utf-8") as f:
            json.dump(lists, f, ensure_ascii=False, indent=1)
    return jsonify({"ok": True, "message": message})


# ---------------------------------------
# CCTV console
# ---------------------------------------
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mkv", ".mov", ".m4v", ".webm", ".mpg", ".mpeg", ".ts", ".dav", ".asf")


@app.route("/")
@app.route("/cctv")
def cctv_page():
    return render_template("cctv.html")


@app.route("/cctv/start", methods=["POST"])
def cctv_start():
    upload = request.files.get("video")
    if upload and upload.filename:
        name = os.path.basename(upload.filename)
        if not name.lower().endswith(VIDEO_EXTENSIONS):
            return jsonify({"error": "That file is not a video. Use MP4, AVI, MKV, MOV or a DVR export."}), 400
        folder = os.path.join(UPLOAD_FOLDER, "cctv")
        os.makedirs(folder, exist_ok=True)
        for old in os.listdir(folder):          # keep only the newest recording
            try:
                os.remove(os.path.join(folder, old))
            except OSError:
                pass
        saved = os.path.join(folder, "input_" + "".join(c for c in name if c.isalnum() or c in "._-"))
        upload.save(saved)
        job = cctv.start(saved, name, is_file=True, count_money=request.form.get("money") == "1")
    else:
        url = (request.form.get("url") or "").strip()
        if not url:
            return jsonify({"error": "Choose a video file or type a camera address."}), 400
        if not (url.isdigit() or url.lower().startswith(("rtsp://", "rtsps://", "http://", "https://"))):
            return jsonify({"error": "Camera address must start with rtsp:// or http://, or be a camera number like 0."}), 400
        title = url.split("@")[-1] if "@" in url else url      # never show the password
        job = cctv.start(url, title, is_file=False, count_money=request.form.get("money") == "1")
    return jsonify({"id": job.id})


@app.route("/cctv/stream/<job_id>")
def cctv_stream(job_id):
    from flask import Response
    job = cctv.get(job_id)
    if not job:
        return "Not found", 404
    return Response(job.frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/cctv/status/<job_id>")
def cctv_status(job_id):
    job = cctv.get(job_id)
    if not job:
        return jsonify({"error": "Not found"}), 404
    since = request.args.get("since", type=int, default=0)
    data = job.snapshot()
    data["events"] = data["events"][since:]
    data["since"] = since
    return jsonify(data)


@app.route("/cctv/stop/<job_id>", methods=["POST"])
def cctv_stop(job_id):
    job = cctv.get(job_id)
    if job:
        job.stop()
    return jsonify({"ok": True})


@app.route("/cctv/video/<job_id>")
def cctv_video(job_id):
    from flask import send_file
    job = cctv.get(job_id)
    if not job or not job.output_path or not os.path.exists(job.output_path):
        return "Not found", 404
    return send_file(job.output_path, as_attachment=True, download_name=f"cctv_{job_id}.mp4")


@app.route("/cctv/add/<job_id>/<int:track_id>", methods=["POST"])
def cctv_add(job_id, track_id):
    """An unknown person from the video: give them a name and a list."""
    job = cctv.get(job_id)
    if not job:
        return jsonify({"error": "This video is no longer open. Analyse it again."}), 404
    try:
        return jsonify(job.add_person(track_id, request.form.get("name", ""), request.form.get("list", "green")))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/cctv/ack/<job_id>/<int:track_id>", methods=["POST"])
def cctv_ack(job_id, track_id):
    job = cctv.get(job_id)
    if not job:
        return jsonify({"error": "Not found"}), 404
    return jsonify({"ack": job.acknowledge(track_id, (request.form.get("by") or "")[:40])})


@app.route("/cctv/log/<job_id>.csv")
def cctv_log(job_id):
    """Visitors and vehicles log (opens in Excel)."""
    from flask import Response
    job = cctv.get(job_id)
    if not job:
        return "Not found", 404
    return Response("\ufeff" + job.log_csv(), mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment; filename=cctv_log_{job_id}.csv"})


@app.route("/cctv/report/<job_id>.csv")
def cctv_report(job_id):
    """Working-time report (opens in Excel)."""
    from flask import Response
    job = cctv.get(job_id)
    if not job:
        return "Not found", 404
    return Response("\ufeff" + job.report_csv(), mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment; filename=working_time_{job_id}.csv"})


@app.route("/cctv/plates", methods=["GET", "POST"])
def cctv_plates():
    """Car lists by plate: GET all, POST plate + list (+ owner) to add or change one."""
    if request.method == "POST":
        try:
            if request.form.get("remove") == "1":
                lpr.remove(request.form.get("plate", ""))
            else:
                lpr.set_list(request.form.get("plate", ""), request.form.get("list", ""), request.form.get("owner", ""))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
    data = lpr.load_lists()
    return jsonify({"enabled": lpr.available(),
                    "plates": [dict(v, key=k) for k, v in sorted(data.items(), key=lambda kv: kv[1]["plate"])]})


@app.route("/cctv/lists", methods=["GET", "POST"])
def cctv_lists():
    if request.method == "POST":
        try:
            cctv.set_list(request.form.get("name", ""), request.form.get("list", ""))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
    names, _ = faces.load_face_db()
    lists = cctv.load_lists()
    people = sorted(set(names))
    return jsonify({"people": [{"name": n, "list": cctv.list_of(n, lists)} for n in people]})


# ---------------------------------------
# Run
# ---------------------------------------
if __name__ == "__main__":
    import socket
    _original_getfqdn = socket.getfqdn

    def _safe_getfqdn(name=""):      # Windows + Python 3.14 idna crash on some PC names
        try:
            return _original_getfqdn(name)
        except Exception:
            return name or "localhost"
    socket.getfqdn = _safe_getfqdn

    print("Loading models ...")
    faces.load_face_db()
    print(f"Raqib is ready: open http://localhost:{PORT} in Chrome")
    import threading
    import webbrowser
    threading.Timer(2, lambda: webbrowser.open(f"http://localhost:{PORT}")).start()
    app.run(host=HOST, port=PORT, debug=False, threaded=True, use_reloader=False)
