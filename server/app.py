import os
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.utils import secure_filename
import datetime
from redis import Redis
from rq import Queue
from task import evaluate
import pymysql
from pathlib import Path
import subprocess

# ---------------------- 基本配置 ----------------------
# 指定文件后缀名
ALLOWED_EXTENSIONS = {".pt" , ".yaml"}
# 指定文件上传路径
UPLOAD_DIR = Path(__file__).parent / "uploaded_files"
UPLOAD_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.secret_key = "zheshisealabdemiyao8008208820"
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)

# 连接 Redis 并初始化队列
redis_conn = Redis(host="localhost", port=6379, db=0)
task_q     = Queue("file_tasks", connection=redis_conn)

def allowed(fname): return Path(fname).suffix.lower() in ALLOWED_EXTENSIONS

# ---------------------- 工具函数 ----------------------
def allowed_file(filename: str) -> bool:
    return (Path(filename).suffix.lower() in ALLOWED_EXTENSIONS)

def mysql_conn():
    return pymysql.connect(
        host=os.getenv("MYSQL_HOST", "10.15.88.88"),
        port=int(os.getenv("MYSQL_PORT", "8001")),
        user=os.getenv("MYSQL_USER", "heyy"),
        password=os.getenv("MYSQL_PASS", "vrlab123"),
        database=os.getenv("MYSQL_DB", "leaderboard"),
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True,
    )

def get_team_by_token(token: str):
    if not token:
        return None
    sql = "SELECT team_id, team_name FROM teams WHERE token=%s LIMIT 1"
    conn = mysql_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (token.strip(),))
            row = cur.fetchone()
            return (row[0], row[1]) if row else None
    finally:
        conn.close()


# ---------------------- 路由 ----------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("upload.html")

def allowed_file(fname):
    return Path(fname).suffix.lower() in ALLOWED_EXTENSIONS

@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        # 1) 先验 token
        token = request.form.get("token", "").strip()

        ckpt = request.files.get("ckpt")
        yaml = request.files.get("yaml")
        robot_type = request.form.get("robot_type")
        task = request.form.get("task")
        # 根据需要修改这里的robot类别
        if not robot_type or robot_type not in {"g1", "h1", "smpl"}:
            flash("请选择 g1 / h1 / smpl")
            return redirect(request.url)
        
        if not task or task not in {"Lie", "Push", "Touch","Lift","Sit","Claw"}:
            flash("请选择 Lie / Push / Touch / Lift / Sit / Claw")
            return redirect(request.url)

        if not ckpt or ckpt.filename == "":
            flash("未选择文件")
            return redirect(request.url)

        if not allowed_file(ckpt.filename):
            flash(f"仅支持{ALLOWED_EXTENSIONS}后缀")
            return redirect(request.url)
        
        if not yaml or yaml.filename == "":
            flash("未选择文件")
            return redirect(request.url)
        
        if not allowed_file(yaml.filename):
            flash(f"仅支持{ALLOWED_EXTENSIONS}后缀")
            return redirect(request.url)
        

        # 此处是队伍id或者token，因此不需要检验是否安全
        date_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        user_dir = UPLOAD_DIR / token / date_str # ./uploaded_files/<用户名>
        user_dir.mkdir(parents=True, exist_ok=True)

        safe_name = secure_filename(ckpt.filename)
        ckpt_path = user_dir / safe_name
        ckpt.save(ckpt_path)

        safe_name = secure_filename(yaml.filename)
        yaml_path = user_dir / safe_name
        yaml.save(yaml_path)

        task_q.enqueue(evaluate, str(ckpt_path), str(yaml_path),token, robot_type, task)

        return render_template(
            "result.html",
            api_token=token,
            filepath=ckpt_path,
            notice="文件已入队处理，稍后可查看结果"
        )
    return render_template("upload.html")

# ---------------------- 入口 ----------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5555, debug=True)