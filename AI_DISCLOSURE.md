# CHAT WITH AI

**`NOTE 1:`** `THIS PROJECT WOULD BE IMPOSSIBLE FOR ME TO COMPLETE WITHOUT AI TOOLS WITHIN DEADLINE. I HAD USED AI IN RESOLVING BUGS, SCRIPTS GENERATION AND CODE GENERATION.`

**`NOTE 2:`** `ALL PROJECT DECISION WERE MINE, HOW TO PUSH IT TO THE GOALS AND KEEP IT ALIGN WITH THE COURSEWORK AND LECTURES WHICH ARE TAUGHT TO US. COMPLETE FLOW OF PROJECT IS UNDER MY SUPERVISION.`

## PROJECT REQUIREMENTS AND STRATEGY SEARCH

**Prompts**
- Explain me requirements of this end to end project what each term means, It is given to us as an assignment ? I have to select one project on my own, and need to implement from the scratch and demonstrate complete orchestration objective in MLOps paradigm. first introduce me about the tools mentioned in guidelines, I am not aware of some of those.

Q: What domain interests you most?
A: Computer Vision

Q: What hardware do you have?
A: NVIDIA GPU (>4GB)

Q: How much implementation time do you have?
A: 3-4 weeks

- I didn't like project ideas, give me some project ideas which can be publicly used not as per business only, like which can help me in doing assignment, job search, pdf managing, text writer, text extractor, doctor handwriting detection and real use case of traffic detection.

| Criteria           | Handwriting | Traffic | Document Intel | Notes Digitizer | Receipt Parser | Answer Evaluator |
| ------------------ | ----------- | ------- | -------------- | --------------- | -------------- | ---------------- |
| Dataset ease       | ⚠️ Medium   | ✅ Easy  | ✅ Easy         | ⚠️ Medium       | ✅ Easy         | ⚠️ Hard          |
| Personal usability | ⭐⭐          | ⭐⭐      | ⭐⭐⭐            | ⭐⭐⭐             | ⭐⭐⭐            | ⭐⭐⭐              |
| Pipeline richness  | High        | High    | High           | Medium          | Medium         | High             |
| UI wow factor      | High        | High    | Medium         | High            | Medium         | High             |
| Drift story        | Strong      | Strong  | Medium         | Medium          | Medium         | Medium           |

- How they are trainable ?
- IDEA on 3D reconstruction project ?
- MOBILE deposit
- can I make spam mail detector and highlight most spamming words ?
### My final search was on project decision
- skin disease detection I want to choose

- Proposal writing for form submission

- create a text document only keep simple and 2 page for form filling text field, don't decorate it much. Just write on behalf of given prompt

## Dataset queries

- from where I  can download ham10000 original dataset, what is its structure ? 
to see the reputed non-errorneous dataset which are publicly available

- (venv) lab@ganapathy:~/Music/DA25S003_SPACE/E2E_Project_DA5402$ python src/utils/check_files.py
Part 1 images : 5000
Part 2 images : 5015
Metadata rows : 10015
Missing images: 0
Test images : 1512
Test metadata rows : 1512
Missing test images 1 : {'ISIC_0035068'}

for tackling missing images, I prompt to debug my `verify_file.py`.
- change to csv also, why test is showing 1512 instead 1511

- Also add size of max min size of images in part 1 part2 test images, segmentations, and save it to raw_data summary, also add if something other is important to trace.

- calculate KNOWN_MISSING_TEST = set(metadata_images) - set(test_images) this way, don't hardcode image name, also keep all code in short and humanly written, make single line comment where ever is necessary

- what is the difference between airflow pipeline and dvc pipeline, how MLFlow would be implemented and at what stage ? also help me in understanding frontend implementation,

- I want to first focus on model training and good macro f1 score 60, 70% for this dvc pipeline and data preprocessing is required, while training if mlflow or airflow is required use that also, Will work on frontend at the last, also want to monitor the user request, like dislike ratios, database up if I would use any nosql database for user image queries and response on answer, per class mistake percentage, status code in pie chart on grafana, lets move first to data processing step then model training, don't miss any of project requirement while creating any code., guide me at every step to consider and not to consider

- do we really need augmentation ?

- make a stratified train val split, test split and csv is already there, also let me know the right time of dvc add and track, this time I want to use drive instead of dagshub,

- will use dagshub for model and data versioning plus experiment tracking. 

- data/reports/ should I dvc track these or not, class_distribution.csv  file_check_summary.csv  image_stats.csv  raw_data_summary.csv

- why lesion_id is repeated in metadata.csv ?

- **`ERROR`**: MODULE ERROR, COLUMN ERROR, KEY ERROR in `prepare.py`
```bash
(venv) lab@ganapathy:~/Music/DA25S003_SPACE/E2E_Project_DA5402$ python src/data_proc/prepare.py
Traceback (most recent call last):                                                                                                                                                     
  File "/home/lab/Music/DA25S003_SPACE/E2E_Project_DA5402/src/data_proc/prepare.py", line 9, in <module>                                                                               
    from src.utils.logger import get_logger                                                                                                                                            
ModuleNotFoundError: No module named 'src'                                                                                                                                             
(venv) lab@ganapathy:~/Music/DA25S003_SPACE/E2E_Project_DA5402$ python src/data_proc/prepare.py                                                                                        
Traceback (most recent call last):                                                                                                                                                     
  File "/home/lab/Music/DA25S003_SPACE/E2E_Project_DA5402/src/data_proc/prepare.py", line 9, in <module>                                                                               
    from utils.logger import get_logger                                                                                                                                                
ModuleNotFoundError: No module named 'utils'                                                                                                                                           
(venv) lab@ganapathy:~/Music/DA25S003_SPACE/E2E_Project_DA5402$ python src/data_proc/prepare.py                                                                                        
Traceback (most recent call last):                                                                                                                                                     
  File "/home/lab/Music/DA25S003_SPACE/E2E_Project_DA5402/src/data_proc/prepare.py", line 9, in <module>                                                                               
    from logger import get_logger                                                                                                                                                      
ModuleNotFoundError: No module named 'logger'
```

## DVC AND GIT VERSIONING QUERIES
- should I add processed/ to dvc or git ?

```yaml
stages:
  prepare:
    cmd: python src/data_proc/prepare.py
    deps:
      - src/data_proc/prepare.py
      - data/raw/HAM10000_metadata.csv
      - data/raw/images/part_1
      - data/raw/images/part_2
      - data/raw/test_images
      - data/raw/ISIC2018_Task3_Test_GroundTruth.csv
    params:
      - prepare
    outs:
      - data/processed/train.csv
      - data/processed/val.csv
      - data/processed/test.csv
    metrics:    
      - data/reports/prepare_summary.json:
          cache: false
```

is something missing here, from prepare.py, some outputs or metrics ?


- what to dvc track and push to dvc list me command git st
```bash
On branch master
Your branch is ahead of 'origin/master' by 1 commit.
  (use "git push" to publish your local commits)
Untracked files:
  (use "git add <file>..." to include in what will be committed)
        data/processed/
        data/reports/
        dvc.lock
        dvc.yaml
        params.yaml
        setup.py
        src/
```

## MLFLOW AND MODEL TRAINING
- Let's enter into phase 4 + 5, and also take care of every necessary logs in MLflow, I am bit unaware of MLflow, guide me also through it. I also don't want to miss any of the model training metrics to log in MLflow. Please properly guide me through to complete this project

`RESPONDED in train.py | inference.py | params.yaml | model.py | metrics.py | mlflow_utils.py`
```bash
On branch master
Your branch is up to date with 'origin/master'.
Changes to be committed:
  (use "git restore --staged <file>..." to unstage)
        modified:   .dvc/config
        modified:   dvc.yaml
        modified:   params.yaml
        modified:   requirements.txt
        new file:   src/models/__init__.py
        new file:   src/models/inference.py
        new file:   src/models/model.py
        new file:   src/models/train.py
        new file:   src/utils/metrics.py
        new file:   src/utils/mlflow_utils.py
Untracked files:
  (use "git add <file>..." to include in what will be committed)
        help.py
        mlartifacts/
        mlflow.db
        outputs/
```
should I gitignore mlartifacts, mlflow.db ? Also I am facing problem in watching data on dvc dagshub, dropdown 

```bash
 git log --oneline
f0757f6 (HEAD -> master, origin/master, origin/HEAD, dagshub/master) cleaned prepare dvc pipeline
e672bd0 pipeline: prepare stage — splits, baseline stats, dvc pipeline
d3b282b data: track raw data with dvc
8870940 dvc remote dagshub cred added"
a32a5c1 init: dvc setup
13a1eaa initial commit
ab99764 Initial commit

 git remote -v
dagshub https://dagshub.com/da25s003/E2E_Project_DA5402.git (fetch)
dagshub https://dagshub.com/da25s003/E2E_Project_DA5402.git (push)
origin  git@github.com:harshit-8118/E2E_Project_DA5402.git (fetch)
origin  git@github.com:harshit-8118/E2E_Project_DA5402.git (push)
```

```yaml
[core]
    remote = origin
['remote "origin"']
    url = https://dagshub.com/da25s003/E2E_Project_DA5402.dvc
```

['remote "origin"']     auth = basic     user = da25s003     password = ******

how to make mlflow remote ?

- are your .dvc pointer files pushed to dagshub?
git push dagshub master then verify dvc data is pushed
dvc push
I tried this thing also, 
but it is not still visible.

- Dagshub mlflow is same as mlflow on local, UI looks too simple, I don't think dagshub would be having that much options

## PORT RESOLUTION QUERIES
```bash
(venv) gowthamaan@mirl-a6000:~/Music/MLOPs/E2E_Project_DA5402$ lsof -i :5000
COMMAND     PID       USER   FD   TYPE    DEVICE SIZE/OFF NODE NAME
python3 2733738 gowthamaan    3u  IPv4 159524008      0t0  TCP localhost:5000 (LISTEN)
python3 2951811 gowthamaan    3u  IPv4 159524008      0t0  TCP localhost:5000 (LISTEN)
python3 2952078 gowthamaan    3u  IPv4 159524008      0t0  TCP localhost:5000 (LISTEN)
python3 2952247 gowthamaan    3u  IPv4 159524008      0t0  TCP localhost:5000 (LISTEN)
python3 2952415 gowthamaan    3u  IPv4 159524008      0t0  TCP localhost:5000 (LISTEN)
```
they are repeatedly coming

* not able to open mlflow ui, When I reset back to http://127.0.0.1:5000 I am connected to linux machine via ssh. help me out, it is loading but not started

```bash
(venv) gowthamaan@mirl-a6000:~/Music/MLOPs/E2E_Project_DA5402$ mlflow ui
Backend store URI not provided. Using sqlite:///mlflow.db
Registry store URI not provided. Using backend store URI.
[MLflow] Security middleware enabled with default settings (localhost-only). To allow connections from other hosts, use --host 0.0.0.0 and configure --allowed-hosts and --cors-allowed-origins.
INFO:     Uvicorn running on http://127.0.0.1:5000 (Press CTRL+C to quit)
INFO:     Started parent process [3138033]
```
when I set up dagshub.mlflow. why it is still trying to run on localhost:5000

- have you created all the code with seed values, to avoid reproducibility across machine, 
torch.seed, deterministic ? src/models/train.py src/models/inference.py src/models/model.py ?

## CODE DEBUGGING

- `params.yaml` + prompt 
please correct if I am missing anything here, I don't want to miss any of metrics, params, models, at any stage. You have already seen inference.py, this one is train.py, correct where it is need change

-  `train.py` and `inference.py` + prompt to debug it.

```bash
> python3 -m src.models.train
Traceback (most recent call last):
  File "/usr/lib/python3.10/runpy.py", line 196, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/usr/lib/python3.10/runpy.py", line 86, in _run_code
    exec(code, run_globals)
  File "/home/gowthamaan/Music/MLOPs/E2E_Project_DA5402/src/models/train.py", line 19, in <module>
    from model import build_model, load_class_weights
ModuleNotFoundError: No module named 'model'
ERROR: failed to reproduce 'train': failed to run: python3 -m src.models.train, exited with 1   
```

- why this dummy input required for this registering ? will it change original weights or original image size like 336 in my cases

```bash
File "/usr/lib/python3.10/runpy.py", line 196, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/usr/lib/python3.10/runpy.py", line 86, in _run_code
    exec(code, run_globals)
  File "/home/gowthamaan/Music/MLOPs/E2E_Project_DA5402/src/models/inference.py", line 225, in <module>
    main()
  File "/home/gowthamaan/Music/MLOPs/E2E_Project_DA5402/src/models/inference.py", line 202, in main
    mlflow.pytorch.log_model(
  File "/home/gowthamaan/.local/lib/python3.10/site-packages/mlflow/pytorch/__init__.py", line 291, in log_model
    return Model.log(
  File "/home/gowthamaan/.local/lib/python3.10/site-packages/mlflow/models/model.py", line 1209, in log
    flavor.save_model(path=local_path, mlflow_model=mlflow_model, **kwargs)
  File "/home/gowthamaan/.local/lib/python3.10/site-packages/mlflow/pytorch/__init__.py", line 473, in save_model
    raise MlflowException(
mlflow.exceptions.MlflowException: If `export_model` is True, then the model input signature must contain only one tensor spec.
```

```python
 mlflow.pytorch.log_model(
                pytorch_model=model,
                name="model",
                registered_model_name="skin-disease-classifier",
                export_model=True,
                input_example=np.random.randn(1, 3, tp["image_size"], tp["image_size"]).astype(np.float32),
                pip_requirements=[
                    f"torch=={torch.__version__}",
                    "torchvision",
                    "Pillow",
                    "numpy",
                ]
            )
```
ERROR IN `model logging`

```bash
Traceback (most recent call last):
  File "/usr/lib/python3.10/runpy.py", line 196, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/usr/lib/python3.10/runpy.py", line 86, in _run_code
    exec(code, run_globals)
  File "/home/gowthamaan/Music/MLOPs/E2E_Project_DA5402/src/models/inference.py", line 223, in <module>
    main()
  File "/home/gowthamaan/Music/MLOPs/E2E_Project_DA5402/src/models/inference.py", line 202, in main
    mlflow.pytorch.log_model(
  File "/home/gowthamaan/.local/lib/python3.10/site-packages/mlflow/pytorch/__init__.py", line 291, in log_model
    return Model.log(
  File "/home/gowthamaan/.local/lib/python3.10/site-packages/mlflow/models/model.py", line 1191, in log
    log_model_metrics_for_step(
  File "/home/gowthamaan/.local/lib/python3.10/site-packages/mlflow/models/model.py", line 1135, in log_model_metrics_for_step
    client.log_batch(run_id=run_id, metrics=metrics_for_step)
  File "/home/gowthamaan/.local/lib/python3.10/site-packages/mlflow/tracking/client.py", line 2697, in log_batch
    return self._tracking_client.log_batch(
  File "/home/gowthamaan/.local/lib/python3.10/site-packages/mlflow/telemetry/track.py", line 30, in wrapper
    result = func(*args, **kwargs)
  File "/home/gowthamaan/.local/lib/python3.10/site-packages/mlflow/tracking/_tracking_service/client.py", line 584, in log_batch
    self.store.log_batch(run_id=run_id, metrics=metrics_batch, params=[], tags=[])
  File "/home/gowthamaan/.local/lib/python3.10/site-packages/mlflow/store/tracking/rest_store.py", line 982, in log_batch
    self._call_endpoint(LogBatch, req_body)
  File "/home/gowthamaan/.local/lib/python3.10/site-packages/mlflow/store/tracking/rest_store.py", line 233, in _call_endpoint
    return call_endpoint(
  File "/home/gowthamaan/.local/lib/python3.10/site-packages/mlflow/utils/rest_utils.py", line 627, in call_endpoint
    response = verify_rest_response(
  File "/home/gowthamaan/.local/lib/python3.10/site-packages/mlflow/utils/rest_utils.py", line 341, in verify_rest_response
    raise RestException(json.loads(response.text))
mlflow.exceptions.RestException: BAD_REQUEST: Response: {'error_code': 'BAD_REQUEST'}
ERROR: failed to reproduce 'evaluate': failed to run: python3 -m src.models.inference, exited with 1
```

- How to delete model registry, saved on dagshub with new mlflow it is logged with run model-registration, will new experiment replace previous saved model registry ?

## MODEL TRAINING AND INFERENCE QUERIES
- model is overfitting, add mixup cutmix logic in train loop

- add label_smoothening

- As per my training strategy, params configuration batch_size 64, image_size 336 
can you calculate the GPU requirements, and describe me the calculations, I am using efficientnet_b3 model.

- it is showing 27gb

```bash
gowthamaan@mirl-a6000:~/Music/MLOPs/E2E_Project_DA5402$ nvidia-smi
Sun Apr  5 09:53:35 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.126.09             Driver Version: 580.126.09     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA RTX A6000               Off |   00000000:D8:00.0  On |                  Off |
| 71%   85C    P2            295W /  300W |   27525MiB /  49140MiB |     99%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            2464      G   /usr/lib/xorg/Xorg                      159MiB |
|    0   N/A  N/A            2677      G   /usr/bin/gnome-shell                    172MiB |
|    0   N/A  N/A            6147      G   /usr/share/code/code                    156MiB |
|    0   N/A  N/A          986061      G   .../7766/usr/lib/firefox/firefox        400MiB |
|    0   N/A  N/A         3288294      C   python3                               26296MiB |
+-----------------------------------------------------------------------------------------+

File "/home/gowthamaan/Music/MLOPs/E2E_Project_DA5402/src/models/train.py", line 219, in main
    train_loss, train_preds, train_labels = run_epoch(
  File "/home/gowthamaan/Music/MLOPs/E2E_Project_DA5402/src/models/train.py", line 121, in run_epoch
    with autocast(enabled=(scaler is not None)):
TypeError: autocast.__init__() missing 1 required positional argument: 'device_type'
ERROR: failed to reproduce 'train': failed to run: python3 -m src.models.train, exited with 1 
```
```python
from torch.amp import autocast, GradScaler

why test metric is not logged on dagshub, 
Is anything is missing, 
artifacts got saved but not sklearn metrics

File "/home/gowthamaan/Music/MLOPs/E2E_Project_DA5402/src/models/train.py", line 320, in <module>
    main()
  File "/home/gowthamaan/Music/MLOPs/E2E_Project_DA5402/src/models/train.py", line 206, in main
    setup_mlflow("skin-disease-detection")
  File "/home/gowthamaan/Music/MLOPs/E2E_Project_DA5402/src/utils/mlflow_utils.py", line 19, in setup_mlflow
    mlflow.set_experiment(experiment_name)
  File "/home/gowthamaan/.local/lib/python3.10/site-packages/mlflow/tracking/fluent.py", line 220, in set_experiment
    raise MlflowException(
mlflow.exceptions.MlflowException: Cannot set a deleted experiment 'skin-disease-detection' as the active experiment. You can restore the experiment, or permanently delete the experiment to create a new one.
ERROR: failed to reproduce 'train': failed to run: python3 -m src.models.train, exited with 1
```

- Model training and testing is done. 

- Now we can move to Then we start `src/api/main.py` — FastAPI backend with `/predict`, `/explain`, `/feedback`, `/health`, `/ready` endpoints loading the registered MLflow model. with this task, write me a full fledged files for this task, and documentation to run this servers and visualize and test the scenes, remember all the necessaties of the assignment and configure correct code with right choice, I want to upload image and in result predict disease of it with gradcam visualization for more beautification results we can discuss later. 

- write complete code for this task. Then will move to ui, airflow, dashboard, deploy etc

- ERROR:api:Model load failed: Could not find a registered artifact repository for: c:\Users\hhars\Videos\E2E_Project_DA5402\outputs\models\best_model.pth. Currently registered schemes are: ['', 'file', 's3', 'r2', 'b2', 'gs', 'wasbs', 'ftp', 'sftp', 'dbfs', 'hdfs', 'viewfs', 'runs', 'models', 'http', 'https', 'mlflow-artifacts', 'abfss']

https://dagshub.com/da25s003/E2E_Project_DA5402.mlflow/#/experiments/0/runs/22cc051b050e431ab64158fe95a02d21/artifacts

artifacts are present at this location

- I wanted to change my code at cpu level also, list me the required changes that I need, also I wanted to implement that at separate branch feature/cpu-deployment

- tell me the changes required for model logging that could work on my local cpu machine,  currently I am working on lab machine linux
artifact_path: mlflow-artifacts:/54775d1754be427d9fdd80026aaf3e9b/models/m-c5c2be67de264a9491461901e38e173a/artifacts flavors:   python_function:     code: code     config:       device: null     data: data     env:       conda: conda.yaml       virtualenv: python_env.yaml     loader_module: mlflow.pytorch     pickle_module_name: mlflow.pytorch.pickle_module     python_version: 3.10.12   pytorch:     code: code     model_data: data     pytorch_version: 2.11.0+cu130 mlflow_version: 3.11.1 model_id: m-c5c2be67de264a9491461901e38e173a model_size_bytes: 43462835 model_uuid: m-c5c2be67de264a9491461901e38e173a prompts: null run_id: f8a0be4f4bb34682baaef1f809f5de4b utc_time_created: '2026-04-15 14:58:21.404944'

- I don't want any issue at later in my code, multiple user uses that machine, their could be issue regarding ports, because multple services would be running, grafana, prometheus, alertmanager, frontend, backend, metrics, model serving, etc guide me through minimal changes that I need to do I am sharing my env, train, inference, data_aug, main:app show me the minimal changes for cpu. `train`, `inference` etc files are prompted.

- I would not train my model on cpu, also want to run a single epoch from gpu machine for cpu level code,


## MONGO DB SETUP

- Now create me MongoDB — replace in-memory feedback_store
and prometheus, alert manager, rules.yaml for grafana dashboard. 

- I want alerts on critical moderate alerts for memory, inference, cpu, gpu. set in a way, I will use smtp gmail for alerts,  complete fully the Prometheus + Grafana — wire /metrics, build dashboards and MongoDB — replace in-memory feedback_store.  Then will move on Docker compose, I am logging cpu model and will do rest of the work on my cpu, If it works fine otherwise will use lab linux machine. set complete files.

- I got this error when tried loading production model, with updated code  Actually i logged this model training on linux cuda machine, but used cpu instead gpu Artifact_path: mlflow-artifacts:/54775d1754be427d9fdd80026aaf3e9b/models/m-3083c33d881f474591d8391dc4639024/artifacts flavors:   python_function:     code: code     config:       device: null     data: data     env:       conda: conda.yaml       virtualenv: python_env.yaml     loader_module: mlflow.pytorch     pickle_module_name: mlflow.pytorch.pickle_module     python_version: 3.10.12   pytorch:     code: code     model_data: data     pytorch_version: 2.11.0+cu130 mlflow_version: 3.11.1 model_id: m-3083c33d881f474591d8391dc4639024 model_size_bytes: 43454943 model_uuid: m-3083c33d881f474591d8391dc4639024 prompts: null run_id: 431cd8226fbe4250af9793e57b059ca5 utc_time_created: '2026-04-16 20:18:43.723446'

took me around 8 hours for 30 epochs, I don't want to training it again on cpu, I can again train it on gpu instead, Please give me a proper resolution to avoid this issue, 

Now I don't want to train my model on linux, because lab machine have restriction at later stage I could face this issue for machine access on grafana, etc I wanted to switch on my cpu with trained model, logged on mlflow, give me better way to switch now

- 
```bash
This is the workflow now, 

completed till ui/ux
please read all the files, 
monitoring, 
src/api/
src/db/
src/utils/
src/models/
src/data_proc/

and yml files, 

I am also confused to push secret credentials ? to github please give me alternatives to use this, 

next task to docker and dashboard, 
help me in doing that too, 

will I be dockerizing all application along with prometheus and alertmanager ? 
grafana, and airflow, 

also read port numbers from the .env

guide me in detail for monitoring and dockerization and airflow 
in which sequence I should do this ?
```

## DOCKERIZATION QUERIES
- 

These are my files and workflow of project, 

If it need any improvements in backend, frontend, mongo please correct that, 

also additionally attach airflow in it, 

Make docker compose cpu only, thats why I have attached docker-requirements.txt, 

use cache or no cache as per requirements for faster building in deployment, 
use stable versions of mongo, prometheus, node-xporter, alertmanager, 

also give me .env file which needs all the required variables, 

also if it needs any correction in rules, premetheus, alertmanager, please correct those as well, 

I have mentioned ports in .env please prefer to use those in dockerization, 

I had skipped airflow till now, but that too for now, 
take me to the finish of this project


- 
These are my files and current workflow 

I am trying to Dockerize, but getting some errors in prometheus and mongodb, 

Also I am unable to understand, how flow of data is being done in this project, 

for the sake of simplicity, please correct 

grafana: 
logger=context userId=0 orgId=0 uname= t=2026-04-18T15:54:12.296344502Z level=info msg="Request Completed" method=GET path=/api/live/ws status=401 remote_addr=172.19.0.1 time_ms=0 duration=844.811µs size=105 referer= handler=/api/live/ws status_source=server errorReason=Unauthorized errorMessageID=session.token.rotate error="token needs to be rotated"

mongodb:
{"t":{"$date":"2026-04-18T15:54:21.971+00:00"},"s":"I",  "c":"WTCHKPT",  "id":22430,   "ctx":"Checkpointer","msg":"WiredTiger message","attr":{"message":{"ts_sec":1776527661,"ts_usec":971375,"thread":"1:0x7f742faf1640","session_name":"WT_SESSION.checkpoint","category":"WT_VERB_CHECKPOINT_PROGRESS","category_id":6,"verbose_level":"DEBUG_1","verbose_level_id":1,"msg":"saving checkpoint snapshot min: 6, snapshot max: 6 snapshot count: 0, oldest timestamp: (0, 0) , meta checkpoint timestamp: (0, 0) base write gen: 316"}}}

prometheus:
time=2026-04-18T15:55:09.705Z level=ERROR source=main.go:658 msg="Error loading rule file patterns from config (--config.file=\"/etc/prometheus/prometheus.yml\")" file=/etc/prometheus/prometheus.yml err="parse rules from file \"/etc/prometheus/rules.yml\" (pattern: \"/etc/prometheus/rules.yml\"): /etc/prometheus/rules.yml: yaml: unmarshal errors:\n  line 224: field inhibit_rules not found in type rulefmt.RuleGroups"

Also may I know, Will users data sent to mongodb compass, installed locally 
or into the docker, 

also please focus on .env, and docker-compose.yml

also check the mounted volumes if missing or not, 
also verify the port numbers, 
or env setup to run the container, 

please correct all the codes, 
next my target is to make grafana dashboard, with easy, moderate, rigrous testing, with testing files, 

also if it is possible to mount the monitoring file, please do so, 
because I don't want to rebuild image again and again, 
I think you are already following this technique, 

on swagger it is showing failed to fetch, 
Please verify if frontend is able to connect to backend or not, 

also please help me configuring mongod settings, 

make my code dockerized and running condition with all functionalities upto date,

- 
docker exec -it dermai-backend python -c "import pymongo; print(pymongo.MongoClient('mongodb://mongodb:27017').admin.command('ping'))"
{'ok': 1.0}

I tried pinging like this it responded with {'ok': 1.0}

it is not working appropriately container is running but showing db not avaliable, on swagger and frontend auth page, 


also container is opened in localhost:27017 instead of mongodb:27017
please find the issue and refine it, 

Sometimes it is working sometimes not, 
I started afresh docker compose down -v then up container, 
it started working, 
please refine if there is any logical error, or missing runs in composing image, 

why host is localhost instead of mongodb: ?

will complete dashboard tomorrow, 
please make robust image and containers, 

I don't want to run docker compose down -v 
and flush all the reading and data, 

also make a back of this if it is possible, 

will move next to airflow dag, data drift alerts, grafana, have target to finish everything by tomorrow, 

first I need ensurity if my containers are robust or not,

- 
some moderate test and rigrous tests are failed, 

check if everything is logged correctly or not, 

Also create me a documentation to create grafana dashboard,  style, design, tools, stats selection, in very detailed format grafana dashboard, 

I am thinking of making single dashboard, application metrics on the top cpu, gpu memory, application db up down, request calls, inference, latency, etc

also make sure nothing vanishes, if container stops,


- 
check if every logging is correct or not ? 
because system metrics is not showing any movement when doing rigrous.testing ? 

system_cpu_percent{job="fastapi_backend"}

trying this prom command 

similarly memory line plots are not showing any spikes on request made  ? 

is there any command issues or application logging issue in backend ? 
please refine that one 

Also please verify if you are missing these metrics any where, 

also if it is not considering system stats please consider that one also,


## GRAFANA AND ALERT TRIGGERING QUERIES
- 
Also total predictions and feedback positive rate is refreshed on every backend off and on, 
it should be updated from the db, not from the app store please ensure these metrics are coming correctly, I guess that is the issue, 

create me a data_creation.py which sample 100 images from data/raw/test/ directory and will be used by easy, moderate, and rigrous test python files, Also update test python files in a single file which would have four option, easy, moderate, rigrous, and overall which also create errors and also create cases where 5 users are submitting /predict requests concurrently, and each one randomly giving feedback of positive/negative 0.8 ratio.

Also increase some of the rules, 
p95 memory per request warning on 500mb and critical alert on 1000mb, 

Also give me very strict rules such that they could hit on moderate and rigrous testing, I have reduced percentage of critical alerts on cpu and memory testing, I have to demonstrate everything for this assignment project and also reduced time 1h to 5m to avoid long time email alerts, 

also if possible add more visualization in dashboard, rates of request, image recived, rate of errors etc, 

MY SYSTEM: I am using 16gb ram hp pavillion, intel i5, 11th gen laptop and using Docker desktop, 

create me moderate airflow pipeline which integrates my data versioning, and data drift scenarios,

- 
This is my final code don't want to increase services etc, 

also attached project instruction for this assignment, 
please make sure if it is safe for this project or not, 
if not, let me know what are the minimal changes I can make for this, if it is avoidable then no problem, 

Also you have written ingestion dag pipeline for me, 
also write instructions how to do setup for this, and finalize my model, 

I guess this would be the last task for this assignment, 

then will move onto commenting and documentation for this assignment along with other necessitites.

- 
DocumentEst. TimeContentUser Manual1 hrScreenshots + step-by-step for non-technical userHLD diagram30 minBlock diagram: Browser → Frontend → Backend → MongoDB → MLflowLLD document1 hrTable of every API endpoint with request/response JSONTest plan + report30 minPaste test output, define acceptance criteria

complete these, 
Also I have updated DAG pipeline to give alert when rerun needed instead of dvc repro ? I can't trigger repro in my cpu machine, 

**COMPLETE REST OF THE TASKS**

## PRODUCTION QUERIES
-
Why I am not getting parallelism ? I couldn't seen more than one active requests on dashboard,

is it because of I am using single cpu system, and single container, 

How to bring parallelism which can handle multiple requests ? 
What is the idea ?

- 
Is it enough changing these two things predict file and docker-compose.yml file only will bring parallelism ? 
My predict file looks like this, Please do necessary change in this file, I had updated my code previously.

## REPORT, HLD, LLD, TEST REPORT, USER_MANUAL generated by AI PROMPT.