# Esper [![Build Status](https://travis-ci.org/scanner-research/esper.svg?branch=master)](https://travis-ci.org/scanner-research/esper)

Esper is a tool for exploratory analysis of large video collections.

* [Setup](https://github.com/scanner-research/esper#setup)
* [Creating a dataset](https://github.com/scanner-research/esper#creating-a-dataset)
* [Development](https://github.com/scanner-research/esper#development)


## Setup
First, [install Docker](https://docs.docker.com/engine/installation/#supported-platforms).

If you have a GPU and are running on Linux:
* [Install nvidia-docker.](https://github.com/NVIDIA/nvidia-docker#quick-start)
* For any command below that uses `docker-compose`, use `nvidia-docker-compose` instead.

Next, you will need to configure your Esper installation. If you are using Google Cloud, follow the instructions in [Getting started with Google Cloud](https://github.com/scanner-research/esper/blob/master/guides/google.md) and replace `local.toml` with `google.toml` below. Otherwise, edit any relevant configuration values in `config/local.toml`. Then run:

```
alias dc=docker-compose
pip install -r requirements.txt
python configure.py --config config/local.toml --dataset default
dc pull
dc up -d
dc exec app ./scripts/setup.sh
```

You have successfully setup Esper! Visit [http://localhost](http://localhost) (or whatever server you're running this on) to see the frontend and [http://localhost:8888](http://localhost:8888) to see the Jupyter notebook.


## Creating a dataset
Outside the container:
```
youtube-dl "https://www.youtube.com/watch?v=dQw4w9WgXcQ" -f mp4 -o app/example.mp4
echo "example.mp4" > app/paths
```

Then enter the container with `dc exec app bash`. Within the container, run:
```
esper-run query/datasets/tvnews/ingest.py
```

Then visit [http://localhost](http://localhost).


## Development

While editing the SASS or JSX files, use the Webpack watcher:
```
dc exec app npm run watch
```

This will automatically rebuild all the frontend files into `assets/bundles` when you change a relevant file.
