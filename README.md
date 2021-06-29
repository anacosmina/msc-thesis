## Activity recognition, object detection, plane segmentation

### How to install:
Download the object detection model and save directory `object_detector` in the same place where the sources are saved:
https://drive.google.com/file/d/1UQR9MS73JYWMjsoWu_m2o2Nb-MkPwj9x/view?usp=sharing.

Then:
~~~
conda create --name robo-framework python=3.6 tornado=6.1
conda activate robo-framework
unset PYTHONPATH
conda install pytorch==1.4.0 torchvision==0.5.0 -c pytorch
pip install cython; pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
cd object_detector/
pip install -r requirements.txt
cd ..
pip install google-cloud-automl
pip install opencv-contrib-python==4.1.2.30
pip install matplotlib
pip install gTTS
pip install tensorflow==1.13.1
pip install tflearn
pip install imutils
~~~

### How to run:
On a machine that has GPUs:
~~~
python full_system.py tracking --load_model object_detector/coco_tracking.pth
~~~

To run only on CPU, add the `--gpus -1` parameter and comment the `torch.cuda.synchronize()` lines in `object_detector/lib/detector.py`.

The input video is set in `full_system.py` (variable `video_src`). `verbose = 1` only prints results as text, `verbose = 2` also generates the corresponding images (motion maps, bounding boxes etc). `results_to_speech` triggers audio messages.

Other parameters that can be varied according to desired performance and available hardware are in `constants.py`: `FRAME_STEP`, `MHI_THRESHOLD`, `CONF_THRESHOLD`, `SIMILARITY_ERROR`.

_Note_: in order for the activity detection model to work, it needs to be deployed in the cloud. This is not happening all the time, as deployment is being paid per hour. Plus, you need a security key (`har-rgb-3cadab83ede7.json`) to call the model. When needed, please ask me to send you the key and to deploy the model.

### Troubleshooting:
Other command for playing audio files, in case mpg321 does not work:
~~~
ffplay out.mp3 -nodisp -autoexit > /dev/null 2>&1
~~~

If while running any errors related to scikit-learn occur:
~~~
pip uninstall scikit-learn
conda install -c conda-forge scikit-learn=0.22.2
~~~

Upon installing/using `gTTS`, it might be needed to also install `mpg321`:
~~~
sudo apt-get install mpg321
~~~

If this error occurs: `ALSA lib pcm_dmix.c:1022:(snd_pcm_dmix_open) unable to open slave`, follow the steps here: https://dev.to/setevoy/linux-alsa-lib-pcmdmixc1108sndpcmdmixopen-unable-to-open-slave-38on.

If this error occurs: `AttributeError: module 'cv2.cv2' has no attribute 'motempl'`, try:
~~~
pip uninstall opencv-python
pip uninstall opencv-contrib-python
pip install opencv-contrib-python==4.1.2.30
~~~

If any error related to DCNv2 (that resides in `object_detector/lib/model/networks/DCNv2/`) occurs, download it and compile it again:
~~~
git clone https://github.com/CharlesShang/DCNv2/
cd DCNv2
./make.sh
~~~

As a reference, here are all the packages that I have in my environment; note that not all of them are needed for running the demo, but the list may be useful if some dependencies need manual installation after the previous steps:

<details>
  <summary>
    See the output of `pip freeze`
  </summary>
  argon2-cffi==20.1.0<br>
  async-generator==1.10<br>
  attrs==21.2.0<br>
  backcall==0.2.0<br>
  bleach==3.3.0<br>
  cachetools==4.2.2<br>
  certifi==2021.5.30<br>
  cffi==1.14.5<br>
  chardet==4.0.0<br>
  click==8.0.1<br>
  cycler==0.10.0<br>
  Cython==0.29.23<br>
  decorator==5.0.9<br>
  defusedxml==0.7.1<br>
  descartes==1.1.0<br>
  easydict==1.9<br>
  entrypoints==0.3<br>
  fire==0.4.0<br>
  flake8==3.9.2<br>
  flake8-import-order==0.18.1<br>
  google-api-core==1.30.0<br>
  google-auth==1.31.0<br>
  google-cloud-automl==2.3.0<br>
  googleapis-common-protos==1.53.0<br>
  grpcio==1.38.0<br>
  gTTS==2.2.2<br>
  idna==2.10<br>
  importlib-metadata==4.5.0<br>
  iniconfig==1.1.1<br>
  ipykernel==5.5.5<br>
  ipython==7.16.1<br>
  ipython-genutils==0.2.0<br>
  ipywidgets==7.6.3<br>
  jedi==0.18.0<br>
  Jinja2==3.0.1<br>
  joblib==1.0.1<br>
  jsonschema==3.2.0<br>
  jupyter==1.0.0<br>
  jupyter-client==6.1.12<br>
  jupyter-console==6.4.0<br>
  jupyter-core==4.7.1<br>
  jupyterlab-pygments==0.1.2<br>
  jupyterlab-widgets==1.0.0<br>
  kiwisolver==1.3.1<br>
  llvmlite==0.36.0<br>
  MarkupSafe==2.0.1<br>
  matplotlib==3.3.4<br>
  mccabe==0.6.1<br>
  mistune==0.8.4<br>
  mkl-fft==1.2.0<br>
  mkl-random==1.0.4<br>
  mkl-service==2.3.0<br>
  motmetrics==1.2.0<br>
  nbclient==0.5.3<br>
  nbconvert==6.0.7<br>
  nbformat==5.1.3<br>
  nest-asyncio==1.5.1<br>
  notebook==6.4.0<br>
  numba==0.53.1<br>
  numpy==1.19.5<br>
  nuscenes-devkit==1.1.5<br>
  opencv-contrib-python=4.1.2.30<br>
  packaging==20.9<br>
  pandas==1.1.5<br>
  pandocfilters==1.4.3<br>
  parso==0.8.2<br>
  pexpect==4.8.0<br>
  pickleshare==0.7.5<br>
  Pillow==8.2.0<br>
  pluggy==0.13.1<br>
  progress==1.5<br>
  prometheus-client==0.11.0<br>
  prompt-toolkit==3.0.18<br>
  proto-plus==1.18.1<br>
  protobuf==3.17.3<br>
  ptyprocess==0.7.0<br>
  py==1.10.0<br>
  py-cpuinfo==8.0.0<br>
  pyasn1==0.4.8<br>
  pyasn1-modules==0.2.8<br>
  pycocotools==2.0.2<br>
  pycodestyle==2.7.0<br>
  pycparser==2.20<br>
  pyflakes==2.3.1<br>
  Pygments==2.9.0<br>
  pyparsing==2.4.7<br>
  pyquaternion==0.9.9<br>
  pyrsistent==0.17.3<br>
  pytest==6.2.4<br>
  pytest-benchmark==3.4.1<br>
  python-dateutil==2.8.1<br>
  pytz==2021.1<br>
  PyYAML==5.4.1<br>
  pyzmq==22.1.0<br>
  qtconsole==5.1.0<br>
  QtPy==1.9.0<br>
  requests==2.25.1<br>
  rsa==4.7.2<br>
  scikit-learn==0.22.2.post1<br>
  scipy==1.5.4<br>
  Send2Trash==1.5.0<br>
  Shapely==1.7.1<br>
  six==1.16.0<br>
  sklearn==0.0<br>
  termcolor==1.1.0<br>
  terminado==0.10.1<br>
  testpath==0.5.0<br>
  threadpoolctl @ file:///tmp/tmp9twdgx9k/threadpoolctl-2.1.0-py3-none-any.whl<br>
  toml==0.10.2<br>
  torch==1.4.0<br>
  torchvision==0.5.0<br>
  tornado==6.1<br>
  tqdm==4.61.1<br>
  traitlets==4.3.3<br>
  typing-extensions==3.10.0.0<br>
  urllib3==1.26.5<br>
  wcwidth==0.2.5<br>
  webencodings==0.5.1<br>
  widgetsnbextension==3.5.1<br>
  xmltodict==0.12.0<br>
  zipp==3.4.1
</details>
