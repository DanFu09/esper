# Esper

Esper is a tool for exploratory analysis of large video collections.

* [Setup](https://github.com/scanner-research/esper#setup)
  * [Using a proxy](https://github.com/scanner-research/esper#using-a-proxy)
  * [Accessing the local database](https://github.com/scanner-research/esper#accessing-the-local-database)
  * [Accessing the cloud database](https://github.com/scanner-research/esper#accessing-the-cloud-database)
* [Processing videos](https://github.com/scanner-research/esper#processing-videos)
* [Development](https://github.com/scanner-research/esper#development)
* [Running with Tableau](https://github.com/scanner-research/esper#running-with-tableau)

## Setup
First, [install Docker](https://docs.docker.com/engine/installation/#supported-platforms).

If you have a GPU and are running on Linux:
* [Install nvidia-docker.](https://github.com/NVIDIA/nvidia-docker#quick-start)
* `pip install nvidia-docker-compose`
* For any command below that uses `docker-compose`, use `nvidia-docker-compose` instead.

If you do not have a GPU or are not running Linux: `pip install docker-compose`

```
export MYSQL_PASSWORD=<pick a password, save it to your shell .rc>
alias dc=docker-compose
dc build
dc up -d
dc exec esper ./setup.sh
```

Then visit `http://yourserver.com`.

### Using a proxy

If you're behind a proxy (e.g. the CMU PDL cluster), configure the [Docker proxy](https://docs.docker.com/engine/admin/systemd/#http-proxy). Make sure `https_proxy` is set in your environment as well.

Use `docker-compose` for any network operations like pulling or pushing (`nvidia-docker-compose` doesn't properly use a proxy yet). Make sure `http_proxy` is NOT set when you do `ndc up -d`.

You can then use an SSH tunnel to access the webserver from your local machine:
```
ssh -L 8080:127.0.0.1:80 <your_server>
```

Then go to [http://localhost:8080](http://localhost:8080) in your browser.

See [Accessing the cloud database](https://github.com/scanner-research/esper#accessing-the-cloud-database) for setting up the Google Cloud proxy. Make sure to modify your database settings in `esper/esper/settings.py` to change the cloud DB host to `127.0.0.1`.

### Accessing the local database
The default Esper build comes with a local MySQL database, saved to `mysql-db` inside the Esper repository. It is exposed on the default port (3306) and comes with a root user whose password is what you specified in `MYSQL_PASSWORD`. Connect to the database with:
```
mysql -h <your server> -u root -p${MYSQL_PASSWORD} esper
```

### Accessing the cloud database
To access the Google Cloud SQL database, ask Will about getting permissions on the Esper project in Google Cloud. Then, install the [Google Cloud SDK](https://cloud.google.com/sdk/downloads). After that, run:
```
gcloud auth login
gcloud config set project visualdb-1046
gcloud auth application-default login
```

This sets up your machine for accessing any of the Google Cloud services. Then, download the [Google Cloud proxy tool](https://cloud.google.com/sql/docs/mysql/connect-admin-proxy#install) and run:
```
./cloud_sql_proxy -instances=visualdb-1046:us-central1:esper=tcp:3306
```

Then connect to the database with:
```
mysql -h 127.0.0.1 -u <your SQL username> esper
```

## Processing videos

To add videos to the database, add them somewhere in the `esper` directory (the directory containing `manage.py`) and create a file `paths` that contains a newline-separated list of relative paths to your videos. Open a shell in the Docker container by running `docker-compose exec esper bash` and then run:

```
python manage.py ingest paths
python manage.py face_detect paths
python manage.py face_cluster paths
```

## Development
While editing the SASS or JSX files, use the Webpack watcher:
```
./node_modules/.bin/webpack --config webpack.config.js --watch
```

By default, a development instance will use a local database. You can change to use the cloud database by modifying `DJANGO_DB_TYPE` and `DJANGO_DB_USER` in `esper/Dockerfile` and re-running `dc build`.

You can also dump the cloud database into your local instance by running from inside the Esper container:

```
./load-cloud-db.sh
```

## Running with Tableau
Follow the instructions in [Accessing the cloud database](https://github.com/scanner-research/esper#accessing-the-cloud-database) to get the cloud proxy running. Then download the Esper workbook with:

```
gsutil cp gs://esper/Esper.twb .
```

Then inside Tableau, do *File -> Open* and open the `Esper.twb` file. Replace the prompted user name with your own SQL user and click *Sign in*.

To use your own MySQL database:
1. Click *Data Source* in the bottom left.
2. Click the dropdown arrow in box labeled *127.0.0.1* in the top left underneath *Connections*.
3. Select *Edit connection...*.
4. Change the server to wherever your local instance is located.
5. Change the username to `root`.
6. Change the password to the value of your `$MYSQL_PASSWORD`.
7. Click *Sign in*.
8. Click *Update now* in the middle-bottom box.
