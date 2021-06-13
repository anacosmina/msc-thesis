### Activity recognition, object detection, plane segmentation

#### How to run:
~~~
python full_system.py tracking --load_model object_detector/coco_tracking.pth
~~~

The input video is set in `full_system.py` (variable `video_src`). `verbose = 1` only prints results as text, `verbose = 2` also generates the corresponding images (motion maps, bounding boxes etc). `results_to_speech` triggers audio messages.

Other parameters that can be varied according to desired performance and available hardware are in `constants.py`: `FRAME_STEP`, `MHI_THRESHOLD`, `CONF_THRESHOLD`, `SIMILARITY_ERROR`.

_Note_: in order for the activity detection model to work, it needs to be deployed in the cloud. This is not happening all the time, as deployment is being paid per hour. Plus, you need a security key (`har-rgb-3cadab83ede7.json`) to call the model. When needed, please ask me to send you the key and to deploy the model. 

#### Installation:
Download the object detection model and save directory `object_detector` in the same place where the sources are saved:
https://drive.google.com/file/d/1UQR9MS73JYWMjsoWu_m2o2Nb-MkPwj9x/view?usp=sharing.

Then:
~~~
conda create --name cosmina python=3.6
conda activate cosmina
conda install pytorch==1.4.0 torchvision==0.5.0 -c pytorch
pip install cython; pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
navigate to object_detector/ and install the requirements: pip install -r requirements.txt
pip install google-cloud-automl
pip install opencv-contrib-python
pip install matplotlib
pip install gTTS
~~~

As a reference, here are all the packages that I have in my environment; note that not all of them are needed for running the demo, but the list may be useful if some dependencies need manual installation after steps 1-9:

~~~
absl-py==0.12.0
actionlib==1.13.2
angles==1.9.13
argon2-cffi==20.1.0
astunparse==1.6.3
async-generator==1.10
attrs==20.3.0
backcall==0.2.0
bleach==3.3.0
bondpy==1.8.6
cachetools==4.2.1
camera-calibration==1.15.3
camera-calibration-parsers==1.12.0
catkin==0.8.10
certifi==2020.12.5
cffi==1.14.5
chardet==4.0.0
click==8.0.1
controller-manager==0.19.4
controller-manager-msgs==0.19.4
cv-bridge==1.15.0
cycler==0.10.0
Cython==0.29.23
dataclasses==0.8
-e git+https://github.com/CharlesShang/DCNv2/@c7f778f28b84c66d3af2bf16f19148a07051dac1#egg=DCNv2
decorator==5.0.7
defusedxml==0.7.1
descartes==1.1.0
diagnostic-analysis==1.10.4
diagnostic-common-diagnostics==1.10.4
diagnostic-updater==1.10.4
dynamic-reconfigure==1.7.1
easydict==1.9
entrypoints==0.3
fire==0.4.0
flake8==3.9.1
flake8-import-order==0.18.1
flatbuffers==1.12
gast==0.3.3
gazebo-plugins==2.9.2
gazebo-ros==2.9.2
gencpp==0.6.5
geneus==3.0.0
genlisp==0.4.18
genmsg==0.5.16
gennodejs==2.0.2
genpy==0.6.15
google-api-core==1.30.0
google-auth==1.30.2
google-auth-oauthlib==0.4.4
google-cloud-automl==2.3.0
google-pasta==0.2.0
googleapis-common-protos==1.53.0
grpcio==1.32.0
gTTS==2.2.2
h5py==2.10.0
idna==2.10
image-geometry==1.15.0
importlib-metadata==4.0.1
iniconfig==1.1.1
interactive-markers==1.12.0
ipykernel==5.5.3
ipython==7.16.1
ipython-genutils==0.2.0
ipywidgets==7.6.3
jedi==0.18.0
Jinja2==2.11.3
joblib==1.0.1
joint-state-publisher==1.15.0
joint-state-publisher-gui==1.15.0
jsonschema==3.2.0
jupyter==1.0.0
jupyter-client==6.1.12
jupyter-console==6.4.0
jupyter-core==4.7.1
jupyterlab-pygments==0.1.2
jupyterlab-widgets==1.0.0
Keras-Preprocessing==1.1.2
kiwisolver==1.3.1
laser-geometry==1.6.7
llvmlite==0.36.0
Markdown==3.3.4
MarkupSafe==1.1.1
matplotlib==3.3.4
mccabe==0.6.1
message-filters==1.15.11
mistune==0.8.4
mkl-fft==1.3.0
mkl-random==1.1.1
mkl-service==2.3.0
motmetrics==1.2.0
nbclient==0.5.3
nbconvert==6.0.7
nbformat==5.1.3
nest-asyncio==1.5.1
notebook==6.3.0
numba==0.53.1
numpy @ file:///tmp/build/80754af9/numpy_and_numpy_base_1603487797006/work
nuscenes-devkit==1.1.5
oauthlib==3.1.1
olefile==0.46
opencv-contrib-python==4.1.2.30
opencv-python==4.5.1.48
opt-einsum==3.3.0
packaging==20.9
pandas==1.1.5
pandocfilters==1.4.3
parso==0.8.2
pexpect==4.8.0
pickleshare==0.7.5
Pillow @ file:///tmp/build/80754af9/pillow_1617386154532/work
pluggy==0.13.1
progress==1.5
prometheus-client==0.10.1
prompt-toolkit==3.0.18
proto-plus==1.18.1
protobuf==3.17.3
ptyprocess==0.7.0
py==1.10.0
py-cpuinfo==8.0.0
pyasn1==0.4.8
pyasn1-modules==0.2.8
pycocotools==2.0.2
pycodestyle==2.7.0
pycparser==2.20
pyflakes==2.3.1
Pygments==2.8.1
pyparsing==2.4.7
pyquaternion==0.9.9
pyrsistent==0.17.3
pytest==6.2.3
pytest-benchmark==3.4.1
python-dateutil==2.8.1
python-qt-binding==0.4.3
pytz==2021.1
PyYAML==5.4.1
pyzmq==22.0.3
qt-dotgraph==0.4.2
qt-gui==0.4.2
qt-gui-cpp==0.4.2
qt-gui-py-common==0.4.2
qtconsole==5.0.3
QtPy==1.9.0
requests==2.25.1
requests-oauthlib==1.3.0
resource-retriever==1.12.6
rosbag==1.15.11
rosboost-cfg==1.15.7
rosclean==1.15.7
roscreate==1.15.7
rosgraph==1.15.11
roslaunch==1.15.11
roslib==1.15.7
roslint==0.12.0
roslz4==1.15.11
rosmake==1.15.7
rosmaster==1.15.11
rosmsg==1.15.11
rosnode==1.15.11
rosparam==1.15.11
rospy==1.15.11
rosservice==1.15.11
rostest==1.15.11
rostopic==1.15.11
rosunit==1.15.7
roswtf==1.15.11
rqt-action==0.4.9
rqt-bag==0.5.1
rqt-bag-plugins==0.5.1
rqt-console==0.4.11
rqt-dep==0.4.12
rqt-graph==0.4.14
rqt-gui==0.5.2
rqt-gui-py==0.5.2
rqt-image-view==0.4.16
rqt-launch==0.4.9
rqt-logger-level==0.4.11
rqt-moveit==0.5.10
rqt-msg==0.4.10
rqt-nav-view==0.5.7
rqt-plot==0.4.13
rqt-pose-view==0.5.11
rqt-publisher==0.4.10
rqt-py-common==0.5.2
rqt-py-console==0.4.10
rqt-reconfigure==0.5.4
rqt-robot-dashboard==0.5.8
rqt-robot-monitor==0.5.13
rqt-robot-steering==0.5.12
rqt-runtime-monitor==0.5.9
rqt-rviz==0.6.1
rqt-service-caller==0.4.10
rqt-shell==0.4.11
rqt-srv==0.4.9
rqt-tf-tree==0.6.2
rqt-top==0.4.10
rqt-topic==0.4.12
rqt-web==0.4.10
rsa==4.7.2
rviz==1.14.7
scikit-learn==0.22.2
scipy==1.5.4
Send2Trash==1.5.0
sensor-msgs==1.13.1
Shapely==1.7.1
six @ file:///tmp/build/80754af9/six_1605205335545/work
smach==2.5.0
smach-ros==2.5.0
smclib==1.8.6
tensorboard==2.5.0
tensorboard-data-server==0.6.1
tensorboard-plugin-wit==1.8.0
tensorflow-estimator==2.4.0
termcolor==1.1.0
terminado==0.9.4
testpath==0.4.4
tf==1.13.2
tf-conversions==1.13.2
tf2-geometry-msgs==0.7.5
tf2-kdl==0.7.5
tf2-py==0.7.5
tf2-ros==0.7.5
toml==0.10.2
topic-tools==1.15.11
torch==1.4.0
torchvision==0.5.0
tornado==6.1
tqdm==4.60.0
traitlets==4.3.3
typing-extensions==3.7.4.3
urllib3==1.26.5
wcwidth==0.2.5
webencodings==0.5.1
Werkzeug==2.0.1
widgetsnbextension==3.5.1
wrapt==1.12.1
xacro==1.14.6
xmltodict==0.12.0
zipp==3.4.1
absl-py==0.12.0
actionlib==1.13.2
angles==1.9.13
argon2-cffi==20.1.0
astunparse==1.6.3
async-generator==1.10
attrs==20.3.0
backcall==0.2.0
bleach==3.3.0
bondpy==1.8.6
cachetools==4.2.1
camera-calibration==1.15.3
camera-calibration-parsers==1.12.0
catkin==0.8.10
certifi==2020.12.5
cffi==1.14.5
chardet==4.0.0
click==8.0.1
controller-manager==0.19.4
controller-manager-msgs==0.19.4
cv-bridge==1.15.0
cycler==0.10.0
Cython==0.29.23
dataclasses==0.8
-e git+https://github.com/CharlesShang/DCNv2/@c7f778f28b84c66d3af2bf16f19148a07051dac1#egg=DCNv2
decorator==5.0.7
defusedxml==0.7.1
descartes==1.1.0
diagnostic-analysis==1.10.4
diagnostic-common-diagnostics==1.10.4
diagnostic-updater==1.10.4
dynamic-reconfigure==1.7.1
easydict==1.9
entrypoints==0.3
fire==0.4.0
flake8==3.9.1
flake8-import-order==0.18.1
flatbuffers==1.12
gast==0.3.3
gazebo-plugins==2.9.2
gazebo-ros==2.9.2
gencpp==0.6.5
geneus==3.0.0
genlisp==0.4.18
genmsg==0.5.16
gennodejs==2.0.2
genpy==0.6.15
google-api-core==1.30.0
google-auth==1.30.2
google-auth-oauthlib==0.4.4
google-cloud-automl==2.3.0
google-pasta==0.2.0
googleapis-common-protos==1.53.0
grpcio==1.32.0
gTTS==2.2.2
h5py==2.10.0
idna==2.10
image-geometry==1.15.0
importlib-metadata==4.0.1
iniconfig==1.1.1
interactive-markers==1.12.0
ipykernel==5.5.3
ipython==7.16.1
ipython-genutils==0.2.0
ipywidgets==7.6.3
jedi==0.18.0
Jinja2==2.11.3
joblib==1.0.1
joint-state-publisher==1.15.0
joint-state-publisher-gui==1.15.0
jsonschema==3.2.0
jupyter==1.0.0
jupyter-client==6.1.12
jupyter-console==6.4.0
jupyter-core==4.7.1
jupyterlab-pygments==0.1.2
jupyterlab-widgets==1.0.0
Keras-Preprocessing==1.1.2
kiwisolver==1.3.1
laser-geometry==1.6.7
llvmlite==0.36.0
Markdown==3.3.4
MarkupSafe==1.1.1
matplotlib==3.3.4
mccabe==0.6.1
message-filters==1.15.11
mistune==0.8.4
mkl-fft==1.3.0
mkl-random==1.1.1
mkl-service==2.3.0
motmetrics==1.2.0
nbclient==0.5.3
nbconvert==6.0.7
nbformat==5.1.3
nest-asyncio==1.5.1
notebook==6.3.0
numba==0.53.1
numpy @ file:///tmp/build/80754af9/numpy_and_numpy_base_1603487797006/work
nuscenes-devkit==1.1.5
oauthlib==3.1.1
olefile==0.46
opencv-contrib-python==4.1.2.30
opencv-python==4.5.1.48
opt-einsum==3.3.0
packaging==20.9
pandas==1.1.5
pandocfilters==1.4.3
parso==0.8.2
pexpect==4.8.0
pickleshare==0.7.5
Pillow @ file:///tmp/build/80754af9/pillow_1617386154532/work
pluggy==0.13.1
progress==1.5
prometheus-client==0.10.1
prompt-toolkit==3.0.18
proto-plus==1.18.1
protobuf==3.17.3
ptyprocess==0.7.0
py==1.10.0
py-cpuinfo==8.0.0
pyasn1==0.4.8
pyasn1-modules==0.2.8
pycocotools==2.0.2
pycodestyle==2.7.0
pycparser==2.20
pyflakes==2.3.1
Pygments==2.8.1
pyparsing==2.4.7
pyquaternion==0.9.9
pyrsistent==0.17.3
pytest==6.2.3
pytest-benchmark==3.4.1
python-dateutil==2.8.1
python-qt-binding==0.4.3
pytz==2021.1
PyYAML==5.4.1
pyzmq==22.0.3
qt-dotgraph==0.4.2
qt-gui==0.4.2
qt-gui-cpp==0.4.2
qt-gui-py-common==0.4.2
qtconsole==5.0.3
QtPy==1.9.0
requests==2.25.1
requests-oauthlib==1.3.0
resource-retriever==1.12.6
rosbag==1.15.11
rosboost-cfg==1.15.7
rosclean==1.15.7
roscreate==1.15.7
rosgraph==1.15.11
roslaunch==1.15.11
roslib==1.15.7
roslint==0.12.0
roslz4==1.15.11
rosmake==1.15.7
rosmaster==1.15.11
rosmsg==1.15.11
rosnode==1.15.11
rosparam==1.15.11
rospy==1.15.11
rosservice==1.15.11
rostest==1.15.11
rostopic==1.15.11
rosunit==1.15.7
roswtf==1.15.11
rqt-action==0.4.9
rqt-bag==0.5.1
rqt-bag-plugins==0.5.1
rqt-console==0.4.11
rqt-dep==0.4.12
rqt-graph==0.4.14
rqt-gui==0.5.2
rqt-gui-py==0.5.2
rqt-image-view==0.4.16
rqt-launch==0.4.9
rqt-logger-level==0.4.11
rqt-moveit==0.5.10
rqt-msg==0.4.10
rqt-nav-view==0.5.7
rqt-plot==0.4.13
rqt-pose-view==0.5.11
rqt-publisher==0.4.10
rqt-py-common==0.5.2
rqt-py-console==0.4.10
rqt-reconfigure==0.5.4
rqt-robot-dashboard==0.5.8
rqt-robot-monitor==0.5.13
rqt-robot-steering==0.5.12
rqt-runtime-monitor==0.5.9
rqt-rviz==0.6.1
rqt-service-caller==0.4.10
rqt-shell==0.4.11
rqt-srv==0.4.9
rqt-tf-tree==0.6.2
rqt-top==0.4.10
rqt-topic==0.4.12
rqt-web==0.4.10
rsa==4.7.2
rviz==1.14.7
scikit-learn==0.22.2
scipy==1.5.4
Send2Trash==1.5.0
sensor-msgs==1.13.1
Shapely==1.7.1
six @ file:///tmp/build/80754af9/six_1605205335545/work
smach==2.5.0
smach-ros==2.5.0
smclib==1.8.6
tensorboard==2.5.0
tensorboard-data-server==0.6.1
tensorboard-plugin-wit==1.8.0
tensorflow-estimator==2.4.0
termcolor==1.1.0
terminado==0.9.4
testpath==0.4.4
tf==1.13.2
tf-conversions==1.13.2
tf2-geometry-msgs==0.7.5
tf2-kdl==0.7.5
tf2-py==0.7.5
tf2-ros==0.7.5
toml==0.10.2
topic-tools==1.15.11
torch==1.4.0
torchvision==0.5.0
tornado==6.1
tqdm==4.60.0
traitlets==4.3.3
typing-extensions==3.7.4.3
urllib3==1.26.5
wcwidth==0.2.5
webencodings==0.5.1
Werkzeug==2.0.1
widgetsnbextension==3.5.1
wrapt==1.12.1
xacro==1.14.6
xmltodict==0.12.0
zipp==3.4.1
~~~
