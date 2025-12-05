import glob
import os
import shutil
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory

import generate_frames
import interpolate
from config_loader import get_api_key, get_depth_model_id, load_config


BASE_OUTPUT = "street_view_project_output"
os.makedirs(BASE_OUTPUT, exist_ok=True)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret")
DEFAULT_STREETVIEW_COUNT = int(os.environ.get("STREETVIEW_PHOTOS", "4"))
CONFIG = load_config()
DEFAULT_API_KEY = get_api_key(CONFIG)
DEPTH_MODEL_ID = get_depth_model_id(CONFIG)

# Memoire simple pour l'etat des executions en cours
RUNS = {}


def build_run_id():
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"run-{stamp}"


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def list_available_runs():
    """Retourne les dossiers disponibles dans la sortie, triés par date."""
    if not os.path.isdir(BASE_OUTPUT):
        return []
    entries = []
    for name in os.listdir(BASE_OUTPUT):
        full = os.path.join(BASE_OUTPUT, name)
        if os.path.isdir(full):
            entries.append((name, os.path.getmtime(full)))
    entries.sort(key=lambda item: item[1], reverse=True)
    return [name for name, _ in entries]


def clean_folder(path):
    if not os.path.isdir(path):
        return
    for fname in os.listdir(path):
        full = os.path.join(path, fname)
        if os.path.isfile(full):
            os.remove(full)


def list_sources_from_disk(run_id, direction):
    folder = os.path.join(BASE_OUTPUT, run_id, direction, "sources")
    if not os.path.isdir(folder):
        return []
    patterns = [os.path.join(folder, "*.jpg"), os.path.join(folder, "*.jpeg"), os.path.join(folder, "*.png")]
    found = []
    for pattern in patterns:
        found.extend(glob.glob(pattern))
    return sorted(found)


def list_frames(folder):
    if not os.path.isdir(folder):
        return []
    frames = sorted(glob.glob(os.path.join(folder, "frame_*.jpg")))
    return [url_for("serve_output", filename=os.path.relpath(frame, BASE_OUTPUT).replace("\\", "/")) for frame in frames]


def path_to_url(path):
    return url_for("serve_output", filename=os.path.relpath(path, BASE_OUTPUT).replace("\\", "/"))


@app.route("/", methods=["GET", "POST"])
def index():
    default_key = DEFAULT_API_KEY
    available_runs = list_available_runs()

    if request.method == "POST":
        address = request.form.get("address", "").strip()
        api_key = request.form.get("api_key", "").strip() or default_key
        num_sources = max(2, int(request.form.get("num_sources", DEFAULT_STREETVIEW_COUNT)))

        if not address or not api_key:
            flash("Adresse et cle API sont obligatoires.")
            return redirect(url_for("index"))

        run_id = build_run_id()
        run_root = ensure_dir(os.path.join(BASE_OUTPUT, run_id))

        dirs = {
            "forward": {"sources": ensure_dir(os.path.join(run_root, "forward", "sources"))},
            "backward": {"sources": ensure_dir(os.path.join(run_root, "backward", "sources"))},
        }

        try:
            forward_sources, forward_meta = generate_frames.fetch_source_images(
                address=address,
                api_key=api_key,
                output_folder=dirs["forward"]["sources"],
                inverser_sens=False,
                num_sources=num_sources,
                return_meta=True,
            )
            backward_sources, backward_meta = generate_frames.fetch_source_images(
                address=address,
                api_key=api_key,
                output_folder=dirs["backward"]["sources"],
                inverser_sens=True,
                num_sources=num_sources,
                return_meta=True,
            )
        except Exception as exc:
            flash(f"Erreur pendant la generation des images : {exc}")
            return redirect(url_for("index"))

        if len(forward_sources) < 2 or len(backward_sources) < 2:
            flash("Impossible de recuperer suffisamment d'images officielles.")
            return redirect(url_for("index"))

        RUNS[run_id] = {
            "address": address,
            "api_key": api_key,
            "num_sources": num_sources,
            "paths": {
                "forward": {"sources": forward_sources, "meta": forward_meta},
                "backward": {"sources": backward_sources, "meta": backward_meta},
            },
        }

        return redirect(url_for("index", run_id=run_id))

    run_id = request.args.get("run_id")
    run = RUNS.get(run_id) if run_id else None

    forward_urls = backward_urls = None
    frames_forward = frames_backward = []
    address = ""
    num_sources = DEFAULT_STREETVIEW_COUNT

    if run_id:
        frames_forward = list_frames(os.path.join(BASE_OUTPUT, run_id, "frames", "forward"))
        frames_backward = list_frames(os.path.join(BASE_OUTPUT, run_id, "frames", "backward"))

        if not run:
            forward_disk = list_sources_from_disk(run_id, "forward")
            backward_disk = list_sources_from_disk(run_id, "backward")
            if forward_disk or backward_disk:
                # Reconstruit un run minimal pour permettre l'affichage dans la galerie.
                run = RUNS[run_id] = {
                    "address": "",
                    "api_key": "",
                    "num_sources": max(len(forward_disk), len(backward_disk)),
                    "paths": {
                        "forward": {"sources": forward_disk, "meta": []},
                        "backward": {"sources": backward_disk, "meta": []},
                    },
                }

    if run:
        address = run.get("address", "")
        num_sources = run.get("num_sources", DEFAULT_STREETVIEW_COUNT)
        forward = run.get("paths", {}).get("forward", {}).get("sources", []) or []
        backward = run.get("paths", {}).get("backward", {}).get("sources", []) or []
        if forward:
            forward_urls = [path_to_url(v) for v in forward]
        if backward:
            backward_urls = [path_to_url(v) for v in backward]

    return render_template(
        "run.html",
        run_id=run_id,
        address=address,
        forward=forward_urls,
        backward=backward_urls,
        frames_forward=frames_forward,
        frames_backward=frames_backward,
        default_key=default_key,
        num_sources=num_sources,
        available_runs=available_runs,
    )


@app.route("/interpolate", methods=["POST"])
def run_interpolation():
    run_id = request.form.get("run_id")
    run = RUNS.get(run_id)
    if not run:
        flash("Session introuvable, veuillez relancer une generation.")
        return redirect(url_for("index"))

    direction = request.form.get("direction", "forward")
    num_frames = max(2, int(request.form.get("frames_per_official", 30)))

    if direction not in ("forward", "backward"):
        flash("Sens inconnu.")
        return redirect(url_for("index", run_id=run_id))

    sources = run["paths"][direction]["sources"]
    meta = run["paths"][direction].get("meta") or []
    distances = [item.get("distance_from_prev_m", None) for item in meta]
    if len(sources) < 2:
        flash("Pas assez d'images officielles pour interpoler.")
        return redirect(url_for("index", run_id=run_id))

    segments = len(sources) - 1
    frames_dir = ensure_dir(os.path.join(BASE_OUTPUT, run_id, "frames", direction))
    clean_folder(frames_dir)

    try:
        frame_index = 0

        # On commence toujours par la photo officielle source_00 pour eviter un premier frame dechiré.
        first_official_dest = os.path.join(frames_dir, f"frame_{frame_index:03d}.jpg")
        shutil.copyfile(sources[0], first_official_dest)
        frame_index += 1

        for idx in range(segments):
            distance_hint = distances[idx + 1] if distances and idx + 1 < len(distances) else None

            written = interpolate.process_interpolation(
                img_a_path=sources[idx],
                img_b_path=sources[idx + 1],
                output_frames_folder=frames_dir,
                num_frames=num_frames,
                start_index=frame_index,
                skip_first_frame=True,
                distance_hint_m=distance_hint,
                depth_model_id=DEPTH_MODEL_ID,
            )
            frame_index += written

            # Ajoute la photo officielle suivante pour reancrer la sequence
            # et eviter un frame final trop degrade par le warp.
            if idx + 1 < len(sources):
                official_path = os.path.join(frames_dir, f"frame_{frame_index:03d}.jpg")
                shutil.copyfile(sources[idx + 1], official_path)
                frame_index += 1
    except Exception as exc:
        flash(f"Erreur pendant l'interpolation : {exc}")
        return redirect(url_for("index", run_id=run_id))

    flash(f"{frame_index} frames generees pour le sens {direction} ({segments} segment(s)).")
    return redirect(url_for("index", run_id=run_id))


@app.route("/output/<path:filename>")
def serve_output(filename):
    # Permet d'exposer les images generees sans copier dans static/
    return send_from_directory(BASE_OUTPUT, filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
