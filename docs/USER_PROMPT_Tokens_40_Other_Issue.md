# What is VNLP Colab?

VNLP Colab is a refactored and optimized version of the VNLP (Turkish NLP) library, specifically designed for high performance and compatibility with Google Colab, Keras 3, and TensorFlow 2.x+.

## Key Features

### 1. High-Performance Architecture
- **Singleton Pattern**: Models loaded once per session, instantaneous re-initialization
- **Tokenize Once Principle**: Single tokenization per sentence, reused across all models
- **Batch Processing**: tf.data API for GPU-optimized parallel inference
- **@tf.function Compilation**: Graph optimization for maximum speed

### 2. Comprehensive NLP Tasks
- **Part-of-Speech (PoS) Tagging**: SPUContextPoS, TreeStackPoS
- **Named Entity Recognition (NER)**: SPUContextNER, CharNER
- **Dependency Parsing (DP)**: SPUContextDP, TreeStackDP
- **Morphological Analysis**: StemmerAnalyzer (Yildiz + Shen)
- **Sentiment Analysis**: SPUCBiGRU Sentiment Analyzer
- **Text Normalization**: Accent removal, typo correction

### 3. Colab-Optimized
- Automatic resource caching in `/content/vnlp_cache`
- Progress bars for downloads and processing (tqdm)
- Structured logging with configurable verbosity
- T4 GPU optimized

## Project Goals

1. **Maintain Functional Parity**: All original VNLP functionality preserved
2. **Optimize Inference Performance**: Dramatic speedup on T4 GPU through batching
3. **Enhance Developer Experience**: Modern Python practices, type hints, clear documentation
4. **Deliver Data-Centric Pipeline**: Single `VNLPipeline` class processes CSV → enriched DataFrame

## Quick Start

### Installation
```python
!pip install -q git+https://github.com/KadirYigitUS/vnlp_colab.git
```

### Basic Usage
```python
from vnlp_colab.pipeline_colab import VNLPipeline
import pandas as pd

# Initialize pipeline with desired models
pipeline = VNLPipeline(models_to_load=['pos', 'ner', 'dep', 'stemmer', 'sentiment'])

# Process CSV file
final_df = pipeline.run(
    csv_path="/content/input.csv",
    output_pickle_path="/content/output.pkl",
    batch_size=64
)

# View results
print(final_df.head())
```

### Input & Process & OUTPUT

- Reads CSV -> DataFrame 
- Preprocesses DataFrame (normalize, tokenize)
- For each row:
	- Calls Stemmer.predict(tokens) -> 'morph', 'lemma'
	- Calls PoSTagger.predict(tokens) -> 'pos'
	- Calls NER.predict(tokens) -> 'ner'
	- Calls DP.predict(batched_tokens) -> 'dep'
	- Saves final DataFrame        


### Output Columns
- Original: `t_code`, `ch_no`, `p_no`, `s_no`, `sentence`
- Generated: `no_accents`, `tokens`, `morph`, `lemma`, `pos`, `ner`, `dep`, `sentiment`

## Architecture Highlights

### Three-Layer Design
1. **Utility & Resource Layer** (`utils_colab.py`)
   - Hardware abstraction (CPU/GPU/TPU detection)
   - Resource downloading and caching
   - Keras 3 helper functions

2. **Model Layer** (e.g., `pos/pos_colab.py`)
   - Specialized modules for each NLP task
   - Singleton factories for efficient loading
   - Batch prediction methods

3. **Orchestration Layer** (`pipeline_colab.py`)
   - User-facing `VNLPipeline` API
   - Coordinates preprocessing and model execution
   - Manages data flow and batching

## Performance Metrics

### Before Optimization (row-by-row pandas.apply)
- ~0.3-0.5 rows/second on large datasets

### After Optimization (tf.data batching)
- Target: **10-50x speedup** depending on batch size and model complexity
- Batch size 32-64 recommended for T4 GPU
- Automatic padding and prefetching for maximum GPU utilization

## Key Design Patterns

### Singleton Factory
```python
# First call: downloads weights, compiles model
pos_tagger = PoSTagger(model='SPUContextPoS')

# Subsequent calls: instant return of cached instance
pos_tagger_2 = PoSTagger(model='SPUContextPoS')
```

### Dependency Injection
```python
# TreeStackDP requires TreeStackPoS which requires Stemmer
# Automatically resolved:
pipeline = VNLPipeline(models_to_load=['dep:TreeStackDP'])
# Loads: stemmer → pos:TreeStackPoS → dep:TreeStackDP
```

### Facade Pattern
`VNLPipeline` provides simple interface hiding complex model interactions

## Modern Python Practices

- **Type Hints**: Full PEP 484 compliance
- **Docstrings**: PEP 257 Google-style documentation
- **Logging**: Structured logging instead of print statements
- **Keras 3 Functional API**: Modern, explicit model definitions
- **pathlib**: Cross-platform path handling
- **Context Managers**: Proper resource management

## License

GNU Affero General Public License v3.0 (AGPL-3.0)

## Credits

Based on VNLP by vngrs-ai: https://github.com/vngrs-ai/vnlp
Refactored for Colab by KadirYigitUS: https://github.com/KadirYigitUS/vnlp_colab

---

## VNLP Colab - Versions `vnlp_colab_tokens_40` and `vnlp_colab_latest`

### `vnlp-colab-project_pre_tf_data.zip.xml` or `vnlp_colab_tokens_40`
* is the previous version of `vnlp_colab_latest`
* contains tokens_40 calculation which contains List[List] form of max_40_tokens_list[[tokens_list_1<=len(40)], [tokens__list2<=len(40)], ...] as df dataframe
* **lacks @tf.function Compilation** that is available in `vnlp_colab_latest`

### `vnlp-colab-project_tf_data_2025-11-15_07.23.zip.xml` or `vnlp_colab_latest`

* `vnlp-colab-project_tf_data_2025-11-15_07.23.zip.xml` contains the final structure of VNLP-Colab: A High-Performance Turkish NLP Pipeline
* The repository is being hosted at: `https://github.com/KadirYigitUS/vnlp_colab`
* installation is via: !pip install -q git+https://github.com/your-username/vnlp-colab.git
* **has** @tf.function Compilation available 
* **lacks tokens_40 calculation** which contains List[List] form of max_40_tokens_list[[tokens_list_1<=len(40)], [tokens__list2<=len(40)], ...] as df dataframe column `tokens_40`


---

<USER PROMPT>: Perfom the following tasks

# TASKS A: Study `vnlp_colab_tokens_40` and `vnlp_colab_latest` Codebase and Documentation

1. **Grok** `vnlp-colab-project_tf_data_2025-11-15_07.23.zip.xml` or `vnlp_colab_latest`
2. **Grok** `vnlp-colab-project_pre_tf_data.zip.xml` or `vnlp_colab_tokens_40`
3. **Grok** all other context md files
4. **Grok** latest Colab Automation Pipeline Script: `COLAB_VNLP_automation_pipeline_2.py`

## TASK B: RESOLVE VNLP Colab Library ISSUES

## 1 - **PACKAGE UPDATES CAUSED DEPENDENCY CONFLICTS**:

### ISSUE LEVEL: **MODERATE**

### ISSUE DEFINITON

Perhaps due to recent updates on dependencies, we are getting an error message while importing vnlp_colab:

```python
# Install VNLP Colab package from GitHub repository
print("📦 Installing VNLP Colab package...")

# Use pip to install directly from GitHub
# -q flag suppresses verbose output for cleaner display
# git+ protocol tells pip to clone from a git repository
!pip install -q git+https://github.com/KadirYigitUS/vnlp_colab.git

print("✅ VNLP Colab package installed successfully")
```

* *Output:*

📦 Installing VNLP Colab package...
  Installing build dependencies ... done
  Getting requirements to build wheel ... done
  Preparing metadata (pyproject.toml) ... done
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 61.0/61.0 kB 2.7 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 16.0/16.0 MB 49.4 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 300.5/300.5 MB 3.6 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 455.8/455.8 kB 36.3 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 162.1/162.1 kB 16.3 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 46.0/46.0 kB 4.4 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 86.8/86.8 kB 8.7 MB/s eta 0:00:00
  Building wheel for vnlp_colab (pyproject.toml) ... done
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
ydf 0.13.0 requires protobuf<7.0.0,>=5.29.1, but you have protobuf 3.20.3 which is incompatible.
tensorflow-metadata 1.17.2 requires protobuf>=4.25.2; python_version >= "3.11", but you have protobuf 3.20.3 which is incompatible.
opentelemetry-proto 1.37.0 requires protobuf<7.0,>=5.0, but you have protobuf 3.20.3 which is incompatible.
grpcio-status 1.71.2 requires protobuf<6.0dev,>=5.26.1, but you have protobuf 3.20.3 which is incompatible.
✅ VNLP Colab package installed successfully

**TASK**: Update VNLP Colab Package Dependencies to the latest versions so on Colab so no issues arise, and the code runs without the need for downgrading or upgrading available packages that will definitely cause conflicts. **Do deep web research on the available libraries to avoid all conflicts.**

**INFO**: Latest packages available on Colab

```bash
# list all available libraries on Colab Platform
!pip list
```

* *Output:*

```
Package                                  Version
---------------------------------------- -------------------
absl-py                                  1.4.0
accelerate                               1.12.0
access                                   1.1.9
affine                                   2.4.0
aiofiles                                 24.1.0
aiohappyeyeballs                         2.6.1
aiohttp                                  3.13.2
aiosignal                                1.4.0
aiosqlite                                0.21.0
alabaster                                1.0.0
albucore                                 0.0.24
albumentations                           2.0.8
ale-py                                   0.11.2
alembic                                  1.17.2
altair                                   5.5.0
annotated-types                          0.7.0
antlr4-python3-runtime                   4.9.3
anyio                                    4.12.0
anywidget                                0.9.21
argon2-cffi                              25.1.0
argon2-cffi-bindings                     25.1.0
array_record                             0.8.3
arrow                                    1.4.0
arviz                                    0.22.0
astropy                                  7.2.0
astropy-iers-data                        0.2025.12.8.0.38.44
astunparse                               1.6.3
atpublic                                 5.1
attrs                                    25.4.0
audioread                                3.1.0
Authlib                                  1.6.5
autograd                                 1.8.0
babel                                    2.17.0
backcall                                 0.2.0
beartype                                 0.22.8
beautifulsoup4                           4.13.5
betterproto                              2.0.0b6
bigframes                                2.30.0
bigquery-magics                          0.10.3
bleach                                   6.3.0
blinker                                  1.9.0
blis                                     1.3.3
blobfile                                 3.1.0
blosc2                                   3.12.2
bokeh                                    3.7.3
Bottleneck                               1.4.2
bqplot                                   0.12.45
branca                                   0.8.2
brotli                                   1.2.0
CacheControl                             0.14.4
cachetools                               6.2.2
catalogue                                2.0.10
certifi                                  2025.11.12
cffi                                     2.0.0
chardet                                  5.2.0
charset-normalizer                       3.4.4
chex                                     0.1.90
clarabel                                 0.11.1
click                                    8.3.1
click-plugins                            1.1.1.2
cligj                                    0.7.2
cloudpathlib                             0.23.0
cloudpickle                              3.1.2
cmake                                    3.31.10
cmdstanpy                                1.3.0
colorcet                                 3.1.0
colorlover                               0.3.0
colour                                   0.1.5
community                                1.0.0b1
confection                               0.1.5
cons                                     0.4.7
contourpy                                1.3.3
cramjam                                  2.11.0
cryptography                             43.0.3
cuda-bindings                            12.9.4
cuda-core                                0.3.2
cuda-pathfinder                          1.3.3
cuda-python                              12.9.4
cuda-toolkit                             12.9.1
cudf-cu12                                25.10.0
cudf-polars-cu12                         25.10.0
cufflinks                                0.17.3
cuml-cu12                                25.10.0
cupy-cuda12x                             13.6.0
curl_cffi                                0.13.0
cvxopt                                   1.3.2
cvxpy                                    1.6.7
cycler                                   0.12.1
cyipopt                                  1.5.0
cymem                                    2.0.13
Cython                                   3.0.12
dask                                     2025.9.1
dask-cuda                                25.10.0
dask-cudf-cu12                           25.10.0
dataproc-spark-connect                   1.0.1
datasets                                 4.0.0
db-dtypes                                1.4.4
dbus-python                              1.2.18
debugpy                                  1.8.15
decorator                                4.4.2
defusedxml                               0.7.1
deprecation                              2.1.0
diffusers                                0.36.0
dill                                     0.3.8
distributed                              2025.9.1
distributed-ucxx-cu12                    0.46.0
distro                                   1.9.0
dlib                                     19.24.6
dm-tree                                  0.1.9
docstring_parser                         0.17.0
docutils                                 0.21.2
dopamine_rl                              4.1.2
duckdb                                   1.3.2
earthengine-api                          1.5.24
easydict                                 1.13
editdistance                             0.8.1
eerepr                                   0.1.2
einops                                   0.8.1
en_core_web_sm                           3.8.0
entrypoints                              0.4
esda                                     2.8.0
et_xmlfile                               2.0.0
etils                                    1.13.0
etuples                                  0.3.10
Farama-Notifications                     0.0.4
fastai                                   2.8.5
fastapi                                  0.118.3
fastcore                                 1.8.17
fastdownload                             0.0.7
fastjsonschema                           2.21.2
fastprogress                             1.0.3
fastrlock                                0.8.3
fasttransform                            0.0.2
ffmpy                                    1.0.0
filelock                                 3.20.0
fiona                                    1.10.1
firebase-admin                           6.9.0
Flask                                    3.1.2
flatbuffers                              25.9.23
flax                                     0.10.7
folium                                   0.20.0
fonttools                                4.61.0
fqdn                                     1.5.1
frozendict                               2.4.7
frozenlist                               1.8.0
fsspec                                   2025.3.0
future                                   1.0.0
gast                                     0.7.0
gcsfs                                    2025.3.0
GDAL                                     3.8.4
gdown                                    5.2.0
geemap                                   0.35.3
geocoder                                 1.38.1
geographiclib                            2.1
geopandas                                1.1.1
geopy                                    2.4.1
giddy                                    2.3.8
gin-config                               0.5.0
gitdb                                    4.0.12
GitPython                                3.1.45
glob2                                    0.7
google                                   3.0.0
google-adk                               1.20.0
google-ai-generativelanguage             0.6.15
google-api-core                          2.28.1
google-api-python-client                 2.187.0
google-auth                              2.43.0
google-auth-httplib2                     0.2.1
google-auth-oauthlib                     1.2.2
google-cloud-aiplatform                  1.129.0
google-cloud-appengine-logging           1.7.0
google-cloud-audit-log                   0.4.0
google-cloud-bigquery                    3.38.0
google-cloud-bigquery-connection         1.19.0
google-cloud-bigquery-storage            2.35.0
google-cloud-bigtable                    2.34.0
google-cloud-core                        2.5.0
google-cloud-dataproc                    5.23.0
google-cloud-datastore                   2.21.0
google-cloud-discoveryengine             0.13.12
google-cloud-firestore                   2.21.0
google-cloud-functions                   1.21.0
google-cloud-language                    2.18.0
google-cloud-logging                     3.12.1
google-cloud-monitoring                  2.28.0
google-cloud-resource-manager            1.15.0
google-cloud-secret-manager              2.25.0
google-cloud-spanner                     3.59.0
google-cloud-speech                      2.34.0
google-cloud-storage                     3.7.0
google-cloud-trace                       1.17.0
google-cloud-translate                   3.23.0
google-colab                             1.0.0
google-crc32c                            1.7.1
google-genai                             1.54.0
google-generativeai                      0.8.5
google-pasta                             0.2.0
google-resumable-media                   2.8.0
googleapis-common-protos                 1.72.0
googledrivedownloader                    1.1.0
gradio                                   5.50.0
gradio_client                            1.14.0
graphviz                                 0.21
greenlet                                 3.3.0
groovy                                   0.1.2
grpc-google-iam-v1                       0.14.3
grpc-interceptor                         0.15.4
grpcio                                   1.76.0
grpcio-status                            1.71.2
grpclib                                  0.4.8
gspread                                  6.2.1
gspread-dataframe                        4.0.0
gym                                      0.25.2
gym-notices                              0.1.0
gymnasium                                1.2.2
h11                                      0.16.0
h2                                       4.3.0
h5netcdf                                 1.7.3
h5py                                     3.15.1
hdbscan                                  0.8.40
hf_transfer                              0.1.9
hf-xet                                   1.2.0
highspy                                  1.12.0
holidays                                 0.86
holoviews                                1.22.1
hpack                                    4.1.0
html5lib                                 1.1
httpcore                                 1.0.9
httpimport                               1.4.1
httplib2                                 0.31.0
httpx                                    0.28.1
httpx-sse                                0.4.3
huggingface-hub                          0.36.0
humanize                                 4.14.0
hyperframe                               6.1.0
hyperopt                                 0.2.7
ibis-framework                           9.5.0
idna                                     3.11
ImageIO                                  2.37.2
imageio-ffmpeg                           0.6.0
imagesize                                1.4.1
imbalanced-learn                         0.14.0
immutabledict                            4.2.2
importlib_metadata                       8.7.0
importlib_resources                      6.5.2
imutils                                  0.5.4
inequality                               1.1.2
inflect                                  7.5.0
iniconfig                                2.3.0
intel-cmplr-lib-ur                       2025.3.1
intel-openmp                             2025.3.1
ipyevents                                2.0.4
ipyfilechooser                           0.6.0
ipykernel                                6.17.1
ipyleaflet                               0.20.0
ipyparallel                              8.8.0
ipython                                  7.34.0
ipython-genutils                         0.2.0
ipython-sql                              0.5.0
ipytree                                  0.2.2
ipywidgets                               7.7.1
isoduration                              20.11.0
itsdangerous                             2.2.0
jaraco.classes                           3.4.0
jaraco.context                           6.0.1
jaraco.functools                         4.3.0
jax                                      0.7.2
jax-cuda12-pjrt                          0.7.2
jax-cuda12-plugin                        0.7.2
jaxlib                                   0.7.2
jeepney                                  0.9.0
jieba                                    0.42.1
Jinja2                                   3.1.6
jiter                                    0.12.0
joblib                                   1.5.2
jsonpatch                                1.33
jsonpickle                               4.1.1
jsonpointer                              3.0.0
jsonschema                               4.25.1
jsonschema-specifications                2025.9.1
jupyter_client                           7.4.9
jupyter-console                          6.6.3
jupyter_core                             5.9.1
jupyter-events                           0.12.0
jupyter_kernel_gateway                   2.5.2
jupyter-leaflet                          0.20.0
jupyter_server                           2.14.0
jupyter_server_terminals                 0.5.3
jupyterlab_pygments                      0.3.0
jupyterlab_widgets                       3.0.16
jupytext                                 1.18.1
kaggle                                   1.7.4.5
kagglehub                                0.3.13
keras                                    3.10.0
keras-hub                                0.21.1
keras-nlp                                0.21.1
keyring                                  25.7.0
keyrings.google-artifactregistry-auth    1.1.2
kiwisolver                               1.4.9
langchain                                1.1.3
langchain-core                           1.1.3
langgraph                                1.0.4
langgraph-checkpoint                     3.0.1
langgraph-prebuilt                       1.0.5
langgraph-sdk                            0.2.15
langsmith                                0.4.58
lark                                     1.3.1
launchpadlib                             1.10.16
lazr.restfulclient                       0.14.4
lazr.uri                                 1.0.6
lazy_loader                              0.4
libclang                                 18.1.1
libcudf-cu12                             25.10.0
libcugraph-cu12                          25.10.1
libcuml-cu12                             25.10.0
libkvikio-cu12                           25.10.0
libpysal                                 4.13.0
libraft-cu12                             25.10.0
librmm-cu12                              25.10.0
librosa                                  0.11.0
libucx-cu12                              1.19.0
libucxx-cu12                             0.46.0
lightgbm                                 4.6.0
linkify-it-py                            2.0.3
llvmlite                                 0.43.0
locket                                   1.0.0
logical-unification                      0.4.7
lxml                                     6.0.2
Mako                                     1.3.10
mapclassify                              2.10.0
Markdown                                 3.10
markdown-it-py                           4.0.0
MarkupSafe                               3.0.3
matplotlib                               3.10.0
matplotlib-inline                        0.2.1
matplotlib-venn                          1.1.2
mcp                                      1.23.3
mdit-py-plugins                          0.5.0
mdurl                                    0.1.2
mgwr                                     2.2.1
miniKanren                               1.0.5
missingno                                0.5.2
mistune                                  3.1.4
mizani                                   0.13.5
mkl                                      2025.3.0
ml_dtypes                                0.5.4
mlxtend                                  0.23.4
momepy                                   0.10.0
more-itertools                           10.8.0
moviepy                                  1.0.3
mpmath                                   1.3.0
msgpack                                  1.1.2
multidict                                6.7.0
multipledispatch                         1.0.0
multiprocess                             0.70.16
multitasking                             0.0.12
murmurhash                               1.0.15
music21                                  9.9.1
namex                                    0.1.0
narwhals                                 2.13.0
natsort                                  8.4.0
nbclassic                                1.3.3
nbclient                                 0.10.2
nbconvert                                7.16.6
nbformat                                 5.10.4
ndindex                                  1.10.1
nest-asyncio                             1.6.0
networkx                                 3.6.1
nibabel                                  5.3.3
nltk                                     3.9.1
notebook                                 6.5.7
notebook_shim                            0.2.4
numba                                    0.60.0
numba-cuda                               0.19.1
numexpr                                  2.14.1
numpy                                    2.0.2
nvidia-cublas-cu12                       12.6.4.1
nvidia-cuda-cccl-cu12                    12.9.27
nvidia-cuda-cupti-cu12                   12.6.80
nvidia-cuda-nvcc-cu12                    12.5.82
nvidia-cuda-nvrtc-cu12                   12.6.77
nvidia-cuda-runtime-cu12                 12.6.77
nvidia-cudnn-cu12                        9.10.2.21
nvidia-cufft-cu12                        11.3.0.4
nvidia-cufile-cu12                       1.11.1.6
nvidia-curand-cu12                       10.3.7.77
nvidia-cusolver-cu12                     11.7.1.2
nvidia-cusparse-cu12                     12.5.4.2
nvidia-cusparselt-cu12                   0.7.1
nvidia-ml-py                             13.590.44
nvidia-nccl-cu12                         2.27.5
nvidia-nvjitlink-cu12                    12.6.85
nvidia-nvshmem-cu12                      3.3.20
nvidia-nvtx-cu12                         12.6.77
nvtx                                     0.2.14
nx-cugraph-cu12                          25.10.0
oauth2client                             4.1.3
oauthlib                                 3.3.1
omegaconf                                2.3.0
onemkl-license                           2025.3.0
openai                                   2.9.0
opencv-contrib-python                    4.12.0.88
opencv-python                            4.12.0.88
opencv-python-headless                   4.12.0.88
openpyxl                                 3.1.5
opentelemetry-api                        1.37.0
opentelemetry-exporter-gcp-logging       1.11.0a0
opentelemetry-exporter-gcp-monitoring    1.11.0a0
opentelemetry-exporter-gcp-trace         1.11.0
opentelemetry-exporter-otlp-proto-common 1.37.0
opentelemetry-exporter-otlp-proto-http   1.37.0
opentelemetry-proto                      1.37.0
opentelemetry-resourcedetector-gcp       1.11.0a0
opentelemetry-sdk                        1.37.0
opentelemetry-semantic-conventions       0.58b0
opt_einsum                               3.4.0
optax                                    0.2.6
optree                                   0.18.0
orbax-checkpoint                         0.11.30
orjson                                   3.11.5
ormsgpack                                1.12.0
osqp                                     1.0.5
overrides                                7.7.0
packaging                                25.0
pandas                                   2.2.2
pandas-datareader                        0.10.0
pandas-gbq                               0.30.0
pandas-stubs                             2.2.2.240909
pandocfilters                            1.5.1
panel                                    1.8.4
param                                    2.3.1
parso                                    0.8.5
parsy                                    2.2
partd                                    1.4.2
patsy                                    1.0.2
peewee                                   3.18.3
peft                                     0.18.0
pexpect                                  4.9.0
pickleshare                              0.7.5
pillow                                   11.3.0
pip                                      24.1.2
platformdirs                             4.5.1
plotly                                   5.24.1
plotnine                                 0.14.5
pluggy                                   1.6.0
plum-dispatch                            2.6.0
ply                                      3.11
pointpats                                2.5.2
polars                                   1.31.0
pooch                                    1.8.2
portpicker                               1.5.2
preshed                                  3.0.12
prettytable                              3.17.0
proglog                                  0.1.12
progressbar2                             4.5.0
prometheus_client                        0.23.1
promise                                  2.3
prompt_toolkit                           3.0.52
propcache                                0.4.1
prophet                                  1.2.1
proto-plus                               1.26.1
protobuf                                 5.29.5
psutil                                   5.9.5
psycopg2                                 2.9.11
psygnal                                  0.15.0
ptyprocess                               0.7.0
PuLP                                     3.3.0
py-cpuinfo                               9.0.0
py4j                                     0.10.9.9
pyarrow                                  18.1.0
pyasn1                                   0.6.1
pyasn1_modules                           0.4.2
pycairo                                  1.29.0
pycocotools                              2.0.10
pycparser                                2.23
pycryptodomex                            3.23.0
pydantic                                 2.12.3
pydantic_core                            2.41.4
pydantic-settings                        2.12.0
pydata-google-auth                       1.9.1
pydot                                    4.0.1
pydotplus                                2.0.2
PyDrive2                                 1.21.3
pydub                                    0.25.1
pyerfa                                   2.0.1.5
pygame                                   2.6.1
pygit2                                   1.19.0
Pygments                                 2.19.2
PyGObject                                3.48.2
PyJWT                                    2.10.1
pylibcudf-cu12                           25.10.0
pylibcugraph-cu12                        25.10.1
pylibraft-cu12                           25.10.0
pymc                                     5.26.1
pynndescent                              0.5.13
pyogrio                                  0.12.1
pyomo                                    6.9.5
PyOpenGL                                 3.1.10
pyOpenSSL                                24.2.1
pyparsing                                3.2.5
pyperclip                                1.11.0
pyproj                                   3.7.2
pysal                                    25.7
pyshp                                    3.0.3
PySocks                                  1.7.1
pyspark                                  4.0.1
pytensor                                 2.35.1
pytest                                   8.4.2
python-apt                               0.0.0
python-box                               7.3.2
python-dateutil                          2.9.0.post0
python-dotenv                            1.2.1
python-json-logger                       4.0.0
python-louvain                           0.16
python-multipart                         0.0.20
python-slugify                           8.0.4
python-snappy                            0.7.3
python-utils                             3.9.1
pytz                                     2025.2
pyviz_comms                              3.0.6
PyWavelets                               1.9.0
PyYAML                                   6.0.3
pyzmq                                    26.2.1
quantecon                                0.10.1
raft-dask-cu12                           25.10.0
rapids-dask-dependency                   25.10.0
rapids-logger                            0.1.19
rasterio                                 1.4.3
rasterstats                              0.20.0
ratelim                                  0.1.6
referencing                              0.37.0
regex                                    2025.11.3
requests                                 2.32.4
requests-oauthlib                        2.0.0
requests-toolbelt                        1.0.0
requirements-parser                      0.9.0
rfc3339-validator                        0.1.4
rfc3986-validator                        0.1.1
rfc3987-syntax                           1.1.0
rich                                     13.9.4
rmm-cu12                                 25.10.0
roman-numerals-py                        3.1.0
rpds-py                                  0.30.0
rpy2                                     3.5.17
rsa                                      4.9.1
rtree                                    1.4.1
ruff                                     0.14.8
safehttpx                                0.1.7
safetensors                              0.7.0
scikit-image                             0.25.2
scikit-learn                             1.6.1
scipy                                    1.16.3
scooby                                   0.11.0
scs                                      3.2.9
seaborn                                  0.13.2
SecretStorage                            3.5.0
segregation                              2.5.3
semantic-version                         2.10.0
Send2Trash                               1.8.3
sentence-transformers                    5.1.2
sentencepiece                            0.2.1
sentry-sdk                               2.47.0
setuptools                               75.2.0
shap                                     0.50.0
shapely                                  2.1.2
shellingham                              1.5.4
simple-parsing                           0.1.7
simplejson                               3.20.2
simsimd                                  6.5.3
six                                      1.17.0
sklearn-pandas                           2.2.0
slicer                                   0.0.8
smart_open                               7.5.0
smmap                                    5.0.2
sniffio                                  1.3.1
snowballstemmer                          3.0.1
sortedcontainers                         2.4.0
soundfile                                0.13.1
soupsieve                                2.8
soxr                                     1.0.0
spacy                                    3.8.11
spacy-legacy                             3.0.12
spacy-loggers                            1.0.5
spaghetti                                1.7.6
spanner-graph-notebook                   1.1.8
spglm                                    1.1.0
Sphinx                                   8.2.3
sphinxcontrib-applehelp                  2.0.0
sphinxcontrib-devhelp                    2.0.0
sphinxcontrib-htmlhelp                   2.1.0
sphinxcontrib-jsmath                     1.0.1
sphinxcontrib-qthelp                     2.0.0
sphinxcontrib-serializinghtml            2.0.0
spint                                    1.0.7
splot                                    1.1.7
spopt                                    0.7.0
spreg                                    1.8.4
SQLAlchemy                               2.0.45
sqlalchemy-spanner                       1.17.1
sqlglot                                  25.20.2
sqlparse                                 0.5.4
srsly                                    2.5.2
sse-starlette                            3.0.3
stanio                                   0.5.1
starlette                                0.48.0
statsmodels                              0.14.6
stringzilla                              4.4.0
stumpy                                   1.13.0
sympy                                    1.14.0
tables                                   3.10.2
tabulate                                 0.9.0
tbb                                      2022.3.0
tblib                                    3.2.2
tcmlib                                   1.4.1
tenacity                                 9.1.2
tensorboard                              2.19.0
tensorboard-data-server                  0.7.2
tensorflow                               2.19.0
tensorflow-datasets                      4.9.9
tensorflow_decision_forests              1.12.0
tensorflow-hub                           0.16.1
tensorflow-metadata                      1.17.2
tensorflow-probability                   0.25.0
tensorflow-text                          2.19.0
tensorstore                              0.1.79
termcolor                                3.2.0
terminado                                0.18.1
text-unidecode                           1.3
textblob                                 0.19.0
tf_keras                                 2.19.0
tf-slim                                  1.1.0
thinc                                    8.3.10
threadpoolctl                            3.6.0
tifffile                                 2025.10.16
tiktoken                                 0.12.0
timm                                     1.0.22
tinycss2                                 1.4.0
tobler                                   0.12.1
tokenizers                               0.22.1
toml                                     0.10.2
tomlkit                                  0.13.3
toolz                                    0.12.1
torch                                    2.9.0+cu126
torchao                                  0.10.0
torchaudio                               2.9.0+cu126
torchdata                                0.11.0
torchsummary                             1.5.1
torchtune                                0.6.1
torchvision                              0.24.0+cu126
tornado                                  6.5.1
tqdm                                     4.67.1
traitlets                                5.7.1
traittypes                               0.2.3
transformers                             4.57.3
treelite                                 4.4.1
treescope                                0.1.10
triton                                   3.5.0
tsfresh                                  0.21.1
tweepy                                   4.16.0
typeguard                                4.4.4
typer                                    0.20.0
typer-slim                               0.20.0
types-pytz                               2025.2.0.20251108
types-setuptools                         80.9.0.20250822
typing_extensions                        4.15.0
typing-inspection                        0.4.2
tzdata                                   2025.2
tzlocal                                  5.3.1
uc-micro-py                              1.0.3
ucxx-cu12                                0.46.0
umap-learn                               0.5.9.post2
umf                                      1.0.2
uri-template                             1.3.0
uritemplate                              4.2.0
urllib3                                  2.5.0
uuid_utils                               0.12.0
uvicorn                                  0.38.0
vega-datasets                            0.9.0
wadllib                                  1.3.6
wandb                                    0.23.1
wasabi                                   1.1.3
watchdog                                 6.0.0
wcwidth                                  0.2.14
weasel                                   0.4.3
webcolors                                25.10.0
webencodings                             0.5.1
websocket-client                         1.9.0
websockets                               15.0.1
Werkzeug                                 3.1.4
wheel                                    0.45.1
widgetsnbextension                       3.6.10
wordcloud                                1.9.4
wrapt                                    2.0.1
wurlitzer                                3.1.1
xarray                                   2025.12.0
xarray-einstats                          0.9.1
xgboost                                  3.1.2
xlrd                                     2.0.2
xxhash                                   3.6.0
xyzservices                              2025.11.0
yarl                                     1.22.0
ydf                                      0.13.0
yellowbrick                              1.5
yfinance                                 0.2.66
zict                                     3.0.0
zipp                                     3.23.0
zstandard                                0.25.0
```
---

## 2 - **TOKEN LENGTH BEYOND 40 CAUSES PARSING AND TF BATCHING ISSUES**:

### ISSUE LEVEL: **SEVERE**

### ISSUE DEFINITON 

`COLAB_VNLP_automation_pipeline_2.py` when run on Colab Platform produces the following errors:

```python
# ============================================================================
# BATCH PROCESSING EXECUTION
# ============================================================================
# This cell processes all CSV files in INPUT_DIR with checkpointing.
# Skips already-processed files to save time.
# Continues processing even if individual files fail.
# ============================================================================

print("="*80)
print("VNLP TURKISH NLP BATCH PROCESSING")
print("="*80)

# ============================================================================
# PHASE 1: Environment Validation
# ============================================================================

print("\n[1] Verifying Google Drive access...")

# Check if Google Drive was mounted in Section 4
# MyDrive is the standard mount point for user's Drive root
if not os.path.exists('/content/drive/MyDrive'):
    # Drive not mounted - cannot proceed
    print("❌ ERROR: Google Drive not mounted!")
    print("💡 Please run the 'Mount Google Drive' cell first")
    raise RuntimeError("Google Drive not mounted")

# Check if INPUT_DIR exists (set in Section 4)
if not os.path.exists(INPUT_DIR):
    # Input directory not found - cannot discover CSV files
    print(f"❌ ERROR: Input directory not found: {INPUT_DIR}")
    print("💡 Please update INPUT_DIR path in the configuration cell")
    raise RuntimeError(f"Input directory not found: {INPUT_DIR}")

# Environment validation passed
print(f"✅ Input directory: {INPUT_DIR}")
print(f"✅ Output directory: {OUTPUT_DIR}")

# ============================================================================
# PHASE 2: File Discovery
# ============================================================================

print(f"\n[2] Discovering CSV files...")

# Create glob pattern for CSV files
# Pattern: <INPUT_DIR>/*.csv (matches any .csv file in directory)
csv_pattern = os.path.join(INPUT_DIR, "*.csv")

# Find all matching CSV files
# glob.glob returns list of full paths to matching files
csv_files = glob.glob(csv_pattern)

# Check if any CSV files were found
if not csv_files:
    # No CSV files - nothing to process
    print(f"❌ No CSV files found in {INPUT_DIR}")
    raise RuntimeError("No CSV files to process")

# CSV files found - report count
print(f"✅ Found {len(csv_files)} CSV files")

# ============================================================================
# PHASE 3: Sequential File Processing
# ============================================================================

print(f"\n[3] Starting batch processing...")
print("="*80)

# Start timing for entire batch
batch_start_time = time.time()

# Initialize results list to store outcome for each file
results = []

# Iterate through each CSV file with enumeration
# enumerate(csv_files, 1) starts counting from 1 instead of 0
for i, csv_file in enumerate(csv_files, 1):
    # Print file header with position in batch
    print(f"\n\n{'*'*80}")
    print(f"FILE {i} of {len(csv_files)}")
    print(f"{'*'*80}")

    # --------------------------------------------------------------------
    # Checkpoint Check: Skip if already processed
    # --------------------------------------------------------------------
    print(f"\n[Checkpoint] Checking if already processed...")
    
    # Call checkpoint function from Section 5
    # Returns (is_processed: bool, existing_zip: Optional[str])
    is_processed, existing_zip = check_already_processed(csv_file, OUTPUT_DIR)

    if is_processed:
        # File was already processed - ZIP archive exists
        print(f"✅ SKIPPING: {os.path.basename(csv_file)}")
        print(f"  Reason: Already processed (ZIP exists: {existing_zip})")
        print(f"  💡 To reprocess, delete the ZIP file first")

        # Add skip record to results
        results.append({
            'filename': Path(csv_file).stem,  # Filename without extension
            'status': 'skipped',
            'existing_zip': existing_zip,  # Path to existing ZIP
            'reason': 'Already processed'
        })
        
        # Skip to next file (continue statement jumps to next iteration)
        continue

    # File not processed yet - proceed with processing
    print(f"✅ Will process: {os.path.basename(csv_file)}")

    # --------------------------------------------------------------------
    # Process Single File
    # --------------------------------------------------------------------
    try:
        # Call main processing function from Section 7
        # Returns dictionary with status and results
        # This executes all 9 steps for this file
        result = process_single_file(csv_file, pipeline, OUTPUT_DIR)
        
        # Add result to results list
        results.append(result)

        # --------------------------------------------------------------------
        # Memory Cleanup Between Files
        # --------------------------------------------------------------------
        print("\n[Cleanup] Clearing memory...")
        
        # Call lightweight cleanup function from Section 5
        # Does NOT unload models (they're reused via Singleton pattern)
        # Only clears Python garbage and TensorFlow session
        cleanup_memory_light()
        
        print("✅ Memory cleanup complete")

    except Exception as e:
        # Fatal error occurred during processing
        # Log error but continue to next file (don't crash entire batch)
        print(f"\n❌ Fatal error processing {os.path.basename(csv_file)}: {e}")
        
        # Add error record to results
        results.append({
            'filename': Path(csv_file).stem,
            'status': 'fatal_error',
            'error': str(e)  # Store error message
        })
        
        # Clean up memory even after error
        cleanup_memory_light()

# All files processed (or attempted)

# ============================================================================
# PHASE 4: Results Summary
# ============================================================================

# Calculate total batch runtime
batch_runtime = time.time() - batch_start_time

# Print final summary header
print(f"\n\n{'#'*80}")
print("BATCH PROCESSING COMPLETE")
print(f"{'#'*80}\n")

# Display overall statistics
print(f"Total Runtime: {format_runtime(batch_runtime)}")
print(f"Files Processed: {len(results)}")

# Count results by status
successful = sum(1 for r in results if r['status'] == 'success')
failed = sum(1 for r in results if r['status'] in ['failed', 'fatal_error'])
skipped = sum(1 for r in results if r['status'] == 'skipped')

# Display counts breakdown
print(f"Successful: {successful}")
print(f"Failed: {failed}")
print(f"Skipped: {skipped}")

# ----------------------------------------------------------------------------
# Display Detailed Results Table
# ----------------------------------------------------------------------------

print(f"\n{'='*80}")
print("DETAILED RESULTS")
print(f"{'='*80}\n")

# Iterate through results with enumeration
for i, result in enumerate(results, 1):
    # Print file header with index
    print(f"\n[{i}] {result['filename']}")
    print(f"    Status: {result['status']}")

    # Display status-specific details
    if result['status'] == 'success':
        # Successful processing - show metrics
        print(f"    Timestamp: {result['timestamp']}")
        print(f"    Sentences: {result['total_sentences']:,}")  # Thousand separators
        print(f"    Tokens: {result['total_tokens']:,}")
        
        # Only show sentiment if available
        if result['avg_sentiment'] is not None:
            print(f"    Avg Sentiment: {result['avg_sentiment']:.3f}")  # 3 decimals
        
        print(f"    Runtime: {result['runtime_formatted']}")
        print(f"    ZIP: {result['output_files']['zip']}")
        
    elif result['status'] == 'skipped':
        # Skipped file - show existing ZIP
        print(f"    Existing ZIP: {result['existing_zip']}")
        print(f"    Reason: {result['reason']}")
        
    else:
        # Failed or fatal error - show error message
        print(f"    Error: {result.get('error', 'Unknown error')}")

# ----------------------------------------------------------------------------
# Final Confirmation
# ----------------------------------------------------------------------------

print(f"\n\n{'#'*80}")
print("🎉 ALL PROCESSING COMPLETE!")
print(f"Output Location: {OUTPUT_DIR}")
print(f"{'#'*80}")
```
* *Output*:

********************************************************************************
FILE 17 of 18
********************************************************************************

[Checkpoint] Checking if already processed...
✅ Will process: BG-CE_sentence-split-corrected.csv

================================================================================
Processing: BG-CE_sentence-split-corrected
Timestamp: 20251212_185147
================================================================================
[BG-CE_sentence-split-corrected] Validating CSV format:   0%|          | 0/9 [00:00<?, ?step/s]
[1/9] Validating CSV format...
[BG-CE_sentence-split-corrected] Running VNLP analysis:  22%|██▏       | 2/9 [00:00<00:06,  1.10step/s]✓ CSV format valid

[2/9] Pre-processing CSV (dtype + Unicode fixes)...
  Pre-processing CSV: BG-CE_sentence-split-corrected.csv
  ✅ Pre-processed: 3728 valid rows, dtypes fixed, Unicode cleaned

[3/9] Running VNLP pipeline...
100% 3728/3728 [00:00<00:00, 82403.35it/s]100% 3728/3728 [00:00<00:00, 67230.92it/s]Processing Batches:  97% 57/59 [05:02<00:22, 11.01s/it][BG-CE_sentence-split-corrected] Running VNLP analysis:  22%|██▏       | 2/9 [05:04<17:44, 152.04s/step]

❌ Fatal error processing BG-CE_sentence-split-corrected.csv: could not broadcast input array from shape (41,18) into shape (1,18)


********************************************************************************
FILE 18 of 18
********************************************************************************

[Checkpoint] Checking if already processed...
✅ Will process: BR-SM_sentence-split-corrected.csv

================================================================================
Processing: BR-SM_sentence-split-corrected
Timestamp: 20251212_185654
================================================================================
[BR-SM_sentence-split-corrected] Validating CSV format:   0%|          | 0/9 [00:00<?, ?step/s]
[1/9] Validating CSV format...
[BR-SM_sentence-split-corrected] Running VNLP analysis:  22%|██▏       | 2/9 [00:00<00:04,  1.60step/s]✓ CSV format valid

[2/9] Pre-processing CSV (dtype + Unicode fixes)...
  Pre-processing CSV: BR-SM_sentence-split-corrected.csv
  ✅ Pre-processed: 7612 valid rows, dtypes fixed, Unicode cleaned

[3/9] Running VNLP pipeline...
100% 7612/7612 [00:00<00:00, 110506.70it/s]100% 7612/7612 [00:00<00:00, 95200.88it/s]Processing Batches:  99% 118/119 [05:21<00:07,  7.03s/it][BR-SM_sentence-split-corrected] Running VNLP analysis:  22%|██▏       | 2/9 [05:22<18:50, 161.46s/step]

❌ Fatal error processing BR-SM_sentence-split-corrected.csv: could not broadcast input array from shape (41,18) into shape (1,18)


################################################################################
BATCH PROCESSING COMPLETE
################################################################################

Total Runtime: 10.9m
Files Processed: 18
Successful: 0
Failed: 17
Skipped: 1

================================================================================
DETAILED RESULTS
================================================================================


[1] TG-NCK_sentence-split-corrected
    Status: failed
    Error: Expected 5 columns, found 4

[2] KS-ZB_sentence-split-corrected
    Status: failed
    Error: Expected 5 columns, found 4

[3] O-SM_sentence-split-corrected
    Status: failed
    Error: Expected 5 columns, found 4

[4] MN5-AS_sentence-split-corrected
    Status: failed
    Error: Expected 5 columns, found 4

[5] PUM-CE_sentence-split-corrected
    Status: failed
    Error: Expected 5 columns, found 4

[6] N-NCK_sentence-split-corrected
    Status: failed
    Error: Expected 5 columns, found 4

[7] YG-ZB_sentence-split-corrected
    Status: failed
    Error: Expected 5 columns, found 4

[8] SU-ZB_sentence-split-corrected
    Status: failed
    Error: Expected 5 columns, found 4

[9] SB-AS_sentence-split-corrected
    Status: failed
    Error: Expected 5 columns, found 4

[10] KT-OK_sentence-split-corrected
    Status: failed
    Error: Expected 5 columns, found 4

[11] N-DK_sentence-split-corrected
    Status: failed
    Error: Expected 5 columns, found 4

[12] SCS-OK_sentence-split-corrected
    Status: failed
    Error: Expected 5 columns, found 4

[13] KM-SG_sentence-split-corrected
    Status: failed
    Error: Expected 5 columns, found 4

[14] BV-SG_sentence-split-corrected
    Status: failed
    Error: Expected 5 columns, found 4

[15] D_1-DK_sentence-split-corrected
    Status: failed
    Error: Expected 5 columns, found 4

[16] A-ZB_sentence-split-corrected
    Status: skipped
    Existing ZIP: /content/drive/MyDrive/BA_Database/CSV_TR/A-ZB_sentence-split-corrected_20251212_160243.zip
    Reason: Already processed

[17] BG-CE_sentence-split-corrected
    Status: fatal_error
    Error: could not broadcast input array from shape (41,18) into shape (1,18)

[18] BR-SM_sentence-split-corrected
    Status: fatal_error
    Error: could not broadcast input array from shape (41,18) into shape (1,18)


################################################################################
🎉 ALL PROCESSING COMPLETE!
Output Location: /content/drive/MyDrive/BA_Database/CSV_TR
################################################################################

**TASK**: Inject the `tokens_40` processing function in `vnlp_colab_tokens_40` to `vnlp_colab_latest` and make a robust pipeline. The former  `vnlp_colab_tokens_40` `pipeline_colab.py` had the following operation that Created 'tokens_40' and sent it to `dep_colab.py` since this code as TOKEN_LEN_MAX=40 and no such issue arose.

```python
def run_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies all preprocessing steps to the DataFrame."""
        logger.info("Starting preprocessing...")

        df['sentence'] = df['sentence'].progress_apply(
            lambda s: re.sub(r'\s+', ' ', s).strip() if isinstance(s, str) else ""
        )
        logger.info("Step 1/4: Cleaned 'sentence' column.")

        df['no_accents'] = df['sentence'].progress_apply(self.normalizer.remove_accent_marks)
        logger.info("Step 2/4: Created 'no_accents' column.")

        df['tokens'] = df['no_accents'].progress_apply(TreebankWordTokenize)
        logger.info("Step 3/4: Created 'tokens' column.")

        df['tokens_40'] = df['tokens'].progress_apply(
            lambda tokens: [tokens[i:i + 40] for i in range(0, len(tokens), 40)] if tokens else []
        )
        logger.info("Step 4/4: Created 'tokens_40' column for Dependency Parser.")
        logger.info("Preprocessing complete.")
        return df
```

**INFO**: I have confirmed that this issue arose due to `sentence` len(tokens)>40.

---

# TASKS C: Resolve `COLAB_VNLP_automation_pipeline_2.py` ISSUES

CSV files that are parsed by `COLAB_VNLP_automation_pipeline_2.py` are:
* Loaded from Goggle Drive `"/content/drive/MyDrive/BA_Database/CSV_TR"`
* tab-separated, headerless, CSV files that always contain, in order (4 columns):

`ch_no`: chapter number (use int16)
`p_no`: chapter number (use int16)
`s_no`: chapter number (use int32)
`sentence`: string sentence text

* but some begin with the following column (5 columns)

`t_code`: text code

text codes are in <file_name>: If <file_name> is: SU-ZB_sentence-split-corrected.csv, `t_code` is `SU-ZB`.
so simple method to extract `t_code` is:

```python
t_code = <file_name>.replace("_sentence-split-corrected.csv", "")
```

or using *Path*:

```python
t_code = <file_name>.replace("_sentence-split-corrected.csv", "")
```

### ISSUE LEVEL: **MINOR**

## 1 - **CSV FILES SHOULD BE SAVED TO COLAB HARDDISK**:

**TASK:** Change `COLAB_VNLP_automation_pipeline_2.py` so that accessed CSV files from GoogleDrive () to `/content/csv/` on Colab (create the dir) **immediately after** they are accessed

* Save CSV files to /content/csv/




## 2 - **CSV FILES REQUIRE PREPROCESSING**:

### ISSUE DEFINITON: `COLAB_VNLP_automation_pipeline_2.py` corrects the missing `t_code` during parsing VNLP TURKISH NLP BATCH PROCESSING


```COLAB_VNLP_automation_pipeline_2.py
================================================================================
Processing: KS-ZB_sentence-split-corrected
Timestamp: 20251212_185129
================================================================================
[KS-ZB_sentence-split-corrected] Validating CSV format:   0%|          | 0/9 [00:00<?, ?step/s]
[1/9] Validating CSV format...
[KS-ZB_sentence-split-corrected] Validating CSV format:   0%|          | 0/9 [00:00<?, ?step/s]✗ Validation failed: Expected 5 columns, found 4
```

### REQUIRED Input Format for parsing
Tab-separated CSV without headers:
```
t_code\tch_no\tp_no\ts_no\tsentence
novel01\t1\t1\t1\tBu film harikaydı, çok beğendim.
novel01\t1\t1\t2\tBenim adım Melikşah ve İstanbul'da yaşıyorum.
```

**TASK 2:** Correct the column format of CSV files in `/content/csv/` on Colab **immediately after** they are copied there, so that parsing operations are reserved only for batch processing.

## 3 - **CORRECTED CSV FILES SHOULD BE SAVED TO Google Drive**:

**TASK 3:** Copy the reformatted CSV files in `/content/csv/` on Colab`/content/csv/` on Colab **immediately after** they are corrected to the  Goggle Drive `"/content/drive/MyDrive/BA_Database/CSV_TR/reformatted"`

## 4 - **PROCESSED OUTPUT FILES SHOULD BE SAVED TO COLAB PLATFORM AND ZIPPED**:

**TASK 4:** The processed `outputs` of the CSV files via `vnlp_colab` library must be **immediately** saved to Colab `/content/output/` (create) and 'zipped' and timestamped (this operation should not wait the whole parsing operation to finish).

## 5 - **ZIP FILES SHOULD BE SAVED TO COLAB PLATFORM**:
**TASK 5:** The zipped `outputs` of processing (and only the zip files) must be **immediately** saved in Google Drive `"/content/drive/MyDrive/BA_Database/CSV_TR/processed/"`(create) (this operation should not wait the whole parsing operation to finish).
